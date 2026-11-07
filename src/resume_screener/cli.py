"""Command-line interface: `screener --help`."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import benchmark as bench
from .config import Settings
from .dataset import DEFAULT_PATH
from .db import ResumeRepository, connect
from .metrics import load_requests, summarize
from .schemas import JobDescription

app = typer.Typer(help="AI resume screener: Claude + MongoDB Atlas Vector Search.", no_args_is_help=True)
console = Console()


def _settings() -> Settings:
    return Settings.from_env()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs.")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command("init-index")
def init_index() -> None:
    """Create or update the Atlas Vector Search index and wait until it's ready."""
    from .vector_index import ensure_vector_index

    s = _settings()
    col = connect(s.mongodb_uri)[s.mongodb_db][s.resumes_collection]
    with console.status("Building vector index..."):
        idx = ensure_vector_index(col, s.vector_index, s.embed_dim)
    console.print(f"[green]Index {idx['name']} is {idx['status']}[/green]")


@app.command()
def ingest(
    n: int = typer.Option(500, help="Number of resumes to sample from the dataset."),
    csv_path: Path = typer.Option(DEFAULT_PATH, help="Path to the Kaggle Resume.csv."),
    workers: int = typer.Option(8, help="Parallel Claude parse requests."),
    reingest: bool = typer.Option(False, help="Re-embed and overwrite resumes already stored."),
) -> None:
    """Parse, embed and store resumes in Atlas."""
    from .ingest import run_ingest
    from .metrics import claude_cost

    s = _settings()
    with console.status("Ingesting...") as status:
        r = run_ingest(
            s,
            n=n,
            csv_path=csv_path,
            workers=workers,
            reingest=reingest,
            on_progress=lambda d, t: status.update(f"Parsing {d}/{t}"),
        )
    cost = claude_cost(s.parse_model, r.claude_input_tokens, r.claude_output_tokens)
    console.print(
        f"Sampled {r.sampled}, already stored {r.skipped_existing}, parsed {r.parsed} "
        f"(cached {r.parse_cached}), stored {r.stored}, failed {len(r.parse_failed)}. "
        f"Claude ${cost:.3f}, {r.embed_tokens} embed tokens, {r.seconds:.0f}s."
    )
    for rid, err in r.parse_failed.items():
        console.print(f"[red]  {rid}: {err}[/red]")


@app.command()
def screen(
    title: str = typer.Option(..., help="Job title."),
    description: str = typer.Option(None, help="Job description text."),
    description_file: Path | None = typer.Option(None, help="Read the description from a file."),
    k: int = typer.Option(10, help="Candidates to retrieve and score."),
    top: int = typer.Option(5, help="Candidates to show."),
    category: str | None = typer.Option(None, help="Only search this dataset category."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Rank candidates for a job description."""
    from .screener import Screener

    if description_file:
        description = description_file.read_text()
    if not description:
        raise typer.BadParameter("Give --description or --description-file")
    job = JobDescription(job_id="adhoc", title=title, description=description)
    result = Screener.from_settings(_settings()).screen(job, retrieve_k=k, category=category)
    ranked = result.top(top)

    if as_json:
        console.print_json(json.dumps([c.model_dump() for c in ranked]))
        return
    table = Table(title=f"{title}: top {len(ranked)} of {len(result.candidates)}")
    for col in ("Fit", "Vector", "Resume", "Headline", "Missing", "Why"):
        table.add_column(col, overflow="fold")
    for c in ranked:
        f = c.fit
        table.add_row(
            str(f.score) if f else "-",
            f"{c.vector_score:.3f}",
            c.resume_id,
            c.headline,
            ", ".join(f.missing_skills[:3]) if f else "",
            f.rationale if f else "",
        )
    console.print(table)
    console.print(f"Cost ${result.cost_usd:.4f} | latency {result.latency_s:.2f}s")


@app.command()
def stats() -> None:
    """Cost and latency summary of logged screening requests."""
    s = summarize(load_requests())
    if not s["requests"]:
        console.print("No requests logged yet. Run `screener screen` or `screener eval`.")
        return
    console.print(
        f"{s['requests']} requests | cost mean ${s['cost_mean_usd']:.4f}, p95 ${s['cost_p95_usd']:.4f} | "
        f"latency p50 {s['latency_p50_s']:.2f}s, p95 {s['latency_p95_s']:.2f}s, max {s['latency_max_s']:.2f}s"
    )


@app.command("bench-pool")
def bench_pool(
    n: int = typer.Option(100, help="Pool size."),
    force: bool = typer.Option(False, help="Overwrite an existing pool."),
) -> None:
    """Pick the fixed resume pool the benchmark searches within."""
    if bench.POOL_PATH.exists() and not force:
        console.print(f"{bench.POOL_PATH} already exists; use --force to replace it (labels may go stale).")
        raise typer.Exit(1)
    ids = bench.build_pool(ResumeRepository.from_settings(_settings()), n=n)
    bench.save_pool(ids)
    console.print(f"Saved {len(ids)} resume ids to {bench.POOL_PATH}")


@app.command()
def label(
    per_job: int = typer.Option(12, help="Candidates to label per job."),
    job_id: str | None = typer.Option(None, help="Only label this job."),
) -> None:
    """Hand-label candidates for the benchmark. Progress saves after every answer."""
    from .screener import Screener

    s = _settings()
    jobs = bench.load_jobs()
    if job_id:
        jobs = [j for j in jobs if j.job_id == job_id]
    pool = bench.load_pool()
    labels = bench.load_labels()
    screener = Screener.from_settings(s, metrics_log=None)

    for job in jobs:
        screener.screen(job, retrieve_k=per_job, resume_ids=pool, score=False)
        todo = [r for r in screener.last_retrieved if r.candidate.resume_id not in labels[job.job_id]]
        random.Random(job.job_id).shuffle(todo)  # hide the vector ranking from the labeler
        if not todo:
            continue
        console.rule(f"[bold]{job.title}[/bold]")
        console.print(job.description)
        for i, r in enumerate(todo, 1):
            p = r.parsed
            console.print(f"\n[bold]({i}/{len(todo)}) Resume {r.candidate.resume_id}[/bold]: {p.headline}")
            console.print(f"[dim]{p.total_years_experience:g} yrs[/dim] {p.summary}")
            console.print(f"Skills: {', '.join(p.skills[:20])}")
            for e in p.experience[:4]:
                console.print(f"  - {e.title} ({e.years:g} yrs): {e.summary}")
            answer = typer.prompt(f"Grade [{bench.GRADE_HELP}, s = skip, q = quit]").strip().lower()
            if answer == "q":
                bench.save_labels(labels)
                raise typer.Exit()
            if answer in {"0", "1", "2", "3"}:
                labels[job.job_id][r.candidate.resume_id] = int(answer)
                bench.save_labels(labels)
    pairs, resumes = bench.label_count(labels)
    console.print(f"[green]Done. {pairs} labels covering {resumes} resumes in {bench.LABELS_PATH}[/green]")


@app.command("eval")
def evaluate_cmd(
    k: int = typer.Option(10, help="Candidates retrieved per job before reranking."),
    repeats: int = typer.Option(1, help="Runs per job (more runs = steadier latency p95)."),
) -> None:
    """Score the screener against the hand-labeled benchmark."""
    from .evaluate import evaluate
    from .screener import Screener

    labels = bench.load_labels()
    pairs, resumes = bench.label_count(labels)
    if not pairs:
        console.print("No labels yet. Run `screener bench-pool`, then `screener label`.")
        raise typer.Exit(1)
    report = evaluate(
        Screener.from_settings(_settings()), bench.load_jobs(), labels, bench.load_pool(), k, repeats
    )
    table = Table(title=f"Benchmark: {pairs} labels, {resumes} resumes")
    for col in ("Job", "Vector top-5", "Reranked top-5", "Unlabeled", "Cost", "Latency"):
        table.add_column(col)
    fmt = lambda v: "n/a" if v is None else f"{v:.0%}"
    for j in report.jobs:
        table.add_row(
            j.job_id, fmt(j.vector_agreement), fmt(j.reranked_agreement),
            str(j.unlabeled_in_top5), f"${j.cost_usd:.4f}", f"{j.latency_s:.2f}s",
        )
    console.print(table)
    console.print(
        f"[bold]Top-5 agreement: {report.reranked_agreement:.0%} reranked "
        f"(vector only {report.vector_agreement:.0%})[/bold] | "
        f"cost ${report.cost_mean_usd:.4f}/request | latency p95 {report.latency_p95_s:.2f}s"
    )


if __name__ == "__main__":
    app()
