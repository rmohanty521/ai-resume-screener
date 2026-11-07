"""Screen candidates for a job: vector retrieval, then Claude fit scoring."""

from __future__ import annotations

import time
from pathlib import Path

from .config import Settings
from .db import connect
from .embeddings import Embedder
from .metrics import DEFAULT_LOG, RequestMetrics, log_request
from .retrieval import RetrievedResume, Retriever
from .schemas import JobDescription, ScreeningResult
from .scorer import FitScorer


class Screener:
    def __init__(
        self,
        retriever: Retriever,
        scorer: FitScorer,
        embed_model: str = "voyage-3.5",
        metrics_log: Path | str | None = DEFAULT_LOG,
    ) -> None:
        self.retriever = retriever
        self.scorer = scorer
        self.embed_model = embed_model
        self.metrics_log = metrics_log
        self.last_retrieved: list[RetrievedResume] = []

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> Screener:
        col = connect(settings.mongodb_uri)[settings.mongodb_db][settings.resumes_collection]
        embedder = Embedder(
            model=settings.embed_model,
            dimension=settings.embed_dim,
            api_key=settings.voyage_api_key,
        )
        return cls(
            Retriever(col, embedder, settings.vector_index),
            FitScorer(model=settings.claude_model),
            embed_model=settings.embed_model,
            **kwargs,
        )

    def screen(
        self,
        job: JobDescription,
        retrieve_k: int = 10,
        category: str | None = None,
        resume_ids: list[str] | None = None,
        score: bool = True,
    ) -> ScreeningResult:
        """Retrieve `retrieve_k` candidates and (optionally) score each with Claude.

        Use `ScreeningResult.top(k)` for the final ranking.
        """
        metrics = RequestMetrics(
            job_id=job.job_id, claude_model=self.scorer.model, embed_model=self.embed_model
        )
        start = time.perf_counter()
        retrieved = self.retriever.search(
            job, limit=retrieve_k, category=category, resume_ids=resume_ids
        )
        metrics.retrieval_s = time.perf_counter() - start
        metrics.embed_tokens = retrieved.embed_tokens
        metrics.candidates = len(retrieved.resumes)
        self.last_retrieved = retrieved.resumes

        candidates = [r.candidate.model_copy() for r in retrieved.resumes]
        if score and retrieved.resumes:
            scored = self.scorer.score(job, retrieved.resumes)
            metrics.scoring_s = scored.latency_s
            metrics.claude_input_tokens = scored.input_tokens
            metrics.claude_output_tokens = scored.output_tokens
            for c in candidates:
                c.fit = scored.scores.get(c.resume_id)

        metrics.total_s = time.perf_counter() - start
        if self.metrics_log:
            log_request(metrics, self.metrics_log)
        return ScreeningResult(
            job_id=job.job_id,
            candidates=candidates,
            cost_usd=metrics.cost_usd,
            latency_s=metrics.total_s,
        )
