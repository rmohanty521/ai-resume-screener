"""End-to-end ingest: load -> sample -> parse (Claude) -> embed (Voyage) -> store (Atlas)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .batch import ParseCache, parse_batch
from .config import Settings
from .dataset import DEFAULT_PATH, load_resumes, stratified_sample
from .db import ResumeRepository, connect
from .embeddings import Embedder
from .parser import ResumeParser
from .vector_index import ensure_vector_index

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    sampled: int = 0
    skipped_existing: int = 0
    parsed: int = 0
    parse_cached: int = 0
    parse_failed: dict[str, str] = field(default_factory=dict)
    embedded: int = 0
    stored: int = 0
    claude_input_tokens: int = 0
    claude_output_tokens: int = 0
    embed_tokens: int = 0
    seconds: float = 0.0


def run_ingest(
    settings: Settings,
    n: int = 500,
    csv_path: Path | str = DEFAULT_PATH,
    workers: int = 8,
    seed: int = 42,
    reingest: bool = False,
    embed_chunk: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
) -> IngestReport:
    start = time.perf_counter()
    report = IngestReport()

    col = connect(settings.mongodb_uri)[settings.mongodb_db][settings.resumes_collection]
    repo = ResumeRepository(col)
    repo.ensure_indexes()

    sample = stratified_sample(load_resumes(csv_path), n, seed=seed)
    report.sampled = len(sample)
    if not reingest:
        existing = repo.existing_ids()
        report.skipped_existing = sum(r.resume_id in existing for r in sample)
        sample = [r for r in sample if r.resume_id not in existing]

    # 1. Parse with Claude (cached on disk, so re-runs are cheap).
    parser = ResumeParser(model=settings.parse_model)
    docs, stats = parse_batch(sample, parser, ParseCache(), workers=workers, on_progress=on_progress)
    report.parsed, report.parse_cached = stats.parsed, stats.cached
    report.parse_failed = stats.failed
    report.claude_input_tokens, report.claude_output_tokens = stats.input_tokens, stats.output_tokens

    # 2-3. Embed each resume's structured summary and store it in Atlas.
    # Work in chunks so progress is saved even if a later chunk fails.
    embedder = Embedder(
        model=settings.embed_model,
        dimension=settings.embed_dim,
        api_key=settings.voyage_api_key,
    )
    for i in range(0, len(docs), embed_chunk):
        chunk = docs[i : i + embed_chunk]
        result = embedder.embed_documents([d.embedding_text() for d in chunk])
        for doc, vec in zip(chunk, result.vectors):
            doc.embedding = vec
        report.embedded += len(chunk)
        report.embed_tokens += result.total_tokens
        report.stored += repo.upsert_many(chunk)
        log.info("Stored %s/%s resumes", report.stored, len(docs))

    # 4. Make sure the vector index exists and is queryable.
    ensure_vector_index(col, settings.vector_index, settings.embed_dim)

    report.seconds = time.perf_counter() - start
    log.info("Ingest done: %s", report)
    return report
