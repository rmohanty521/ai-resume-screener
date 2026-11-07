"""Parse many resumes concurrently, caching each result on disk.

Re-running is cheap: resumes already in the cache are skipped, so an
interrupted run picks up where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .dataset import RawResume
from .parser import ParseResult, ResumeParser
from .schemas import ParsedResume, ResumeDocument

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/cache/parsed")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class ParseCache:
    """One JSON file per resume, keyed by id; invalidated if the text changes."""

    def __init__(self, root: Path | str = DEFAULT_CACHE) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, resume_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in resume_id)
        return self.root / f"{safe}.json"

    def get(self, raw: RawResume) -> dict | None:
        path = self._path(raw.resume_id)
        if not path.exists():
            return None
        entry = json.loads(path.read_text())
        return entry if entry.get("text_hash") == _text_hash(raw.text) else None

    def put(self, raw: RawResume, result: ParseResult) -> dict:
        entry = {
            "resume_id": raw.resume_id,
            "category": raw.category,
            "text_hash": _text_hash(raw.text),
            "parsed": result.parsed.model_dump(),
            "meta": {
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_s": round(result.latency_s, 3),
                "truncated": result.truncated,
            },
        }
        tmp = self._path(raw.resume_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, indent=2))
        tmp.replace(self._path(raw.resume_id))  # atomic: no half-written files
        return entry


@dataclass
class BatchStats:
    parsed: int = 0
    cached: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


def _to_document(raw: RawResume, entry: dict) -> ResumeDocument:
    return ResumeDocument(
        resume_id=raw.resume_id,
        category=raw.category,
        raw_text=raw.text,
        parsed=ParsedResume.model_validate(entry["parsed"]),
    )


def parse_batch(
    resumes: Iterable[RawResume],
    parser: ResumeParser,
    cache: ParseCache | None = None,
    workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ResumeDocument], BatchStats]:
    """Parse resumes in parallel. Failures are recorded in stats, not raised."""
    cache = cache or ParseCache()
    resumes = list(resumes)
    stats = BatchStats()
    docs: dict[str, ResumeDocument] = {}

    todo = []
    for raw in resumes:
        entry = cache.get(raw)
        if entry:
            docs[raw.resume_id] = _to_document(raw, entry)
            stats.cached += 1
        else:
            todo.append(raw)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parser.parse, raw.text): raw for raw in todo}
        for fut in as_completed(futures):
            raw = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # keep going; one bad resume shouldn't stop the batch
                log.warning("Failed to parse %s: %s", raw.resume_id, exc)
                stats.failed[raw.resume_id] = str(exc)
            else:
                docs[raw.resume_id] = _to_document(raw, cache.put(raw, result))
                stats.parsed += 1
                stats.input_tokens += result.input_tokens
                stats.output_tokens += result.output_tokens
            done += 1
            if on_progress:
                on_progress(done, len(todo))

    # Preserve input order.
    ordered = [docs[r.resume_id] for r in resumes if r.resume_id in docs]
    return ordered, stats
