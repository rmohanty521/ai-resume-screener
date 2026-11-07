"""Hand-labeled benchmark: fixed jobs, a fixed 100-resume pool, and human grades.

Files in data/benchmark/:
- jobs.json   job descriptions (checked in)
- pool.txt    the 100 resume ids searches are restricted to (checked in)
- labels.csv  job_id,resume_id,grade written by `screener label` (checked in)

Grades: 0 = no fit, 1 = weak, 2 = good, 3 = strong. Grade >= 2 counts as a match.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .db import ResumeRepository
from .schemas import JobDescription

BENCH_DIR = Path("data/benchmark")
JOBS_PATH = BENCH_DIR / "jobs.json"
POOL_PATH = BENCH_DIR / "pool.txt"
LABELS_PATH = BENCH_DIR / "labels.csv"

RELEVANT_GRADE = 2
GRADE_HELP = "0 = no fit, 1 = weak, 2 = good, 3 = strong"


def load_jobs(path: Path | str = JOBS_PATH) -> list[JobDescription]:
    return [JobDescription(**j) for j in json.loads(Path(path).read_text())]


def build_pool(repo: ResumeRepository, n: int = 100, seed: int = 7) -> list[str]:
    """Pick `n` stored resumes spread evenly across categories (deterministic)."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for d in repo.col.find({}, {"resume_id": 1, "category": 1, "_id": 0}):
        by_cat[d.get("category", "")].append(d["resume_id"])
    rng = random.Random(seed)
    for ids in by_cat.values():
        ids.sort()
        rng.shuffle(ids)
    pool: list[str] = []
    while len(pool) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(pool) < n:
                pool.append(by_cat[cat].pop())
    return sorted(pool, key=lambda x: (len(x), x))


def save_pool(ids: list[str], path: Path | str = POOL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(ids) + "\n")


def load_pool(path: Path | str = POOL_PATH) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `screener bench-pool` first.")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


Labels = dict[str, dict[str, int]]  # job_id -> resume_id -> grade


def load_labels(path: Path | str = LABELS_PATH) -> Labels:
    labels: Labels = defaultdict(dict)
    path = Path(path)
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                labels[row["job_id"]][row["resume_id"]] = int(row["grade"])
    return labels


def save_labels(labels: Labels, path: Path | str = LABELS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "resume_id", "grade"])
        for job_id in sorted(labels):
            for resume_id, grade in sorted(labels[job_id].items()):
                w.writerow([job_id, resume_id, grade])
    tmp.replace(path)


def label_count(labels: Labels) -> tuple[int, int]:
    """(labeled pairs, distinct resumes labeled)."""
    pairs = sum(len(v) for v in labels.values())
    resumes = len({r for v in labels.values() for r in v})
    return pairs, resumes
