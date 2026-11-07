"""Measure screener quality against the hand-labeled benchmark.

Top-5 agreement for one job = share of the screener's top-5 picks that the human
labeled a match (grade >= 2), divided by min(5, number of human matches) so a
job with only 3 good candidates in the pool can still score 100%.
Reported for vector search alone and for vector search + Claude reranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .benchmark import RELEVANT_GRADE, Labels
from .metrics import percentile
from .schemas import JobDescription
from .screener import Screener


def top_k_agreement(ranked_ids: list[str], grades: dict[str, int], k: int = 5) -> float | None:
    """None when the human found no match for this job (nothing to agree on)."""
    relevant = {rid for rid, g in grades.items() if g >= RELEVANT_GRADE}
    if not relevant:
        return None
    hits = sum(rid in relevant for rid in ranked_ids[:k])
    return hits / min(k, len(relevant))


@dataclass
class JobEval:
    job_id: str
    vector_top5: list[str]
    reranked_top5: list[str]
    vector_agreement: float | None
    reranked_agreement: float | None
    unlabeled_in_top5: int
    cost_usd: float
    latency_s: float


@dataclass
class EvalReport:
    jobs: list[JobEval] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        vals = [getattr(j, attr) for j in self.jobs if getattr(j, attr) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def vector_agreement(self) -> float:
        return self._mean("vector_agreement")

    @property
    def reranked_agreement(self) -> float:
        return self._mean("reranked_agreement")

    @property
    def cost_mean_usd(self) -> float:
        return sum(j.cost_usd for j in self.jobs) / max(1, len(self.jobs))

    @property
    def latency_p95_s(self) -> float:
        return percentile([j.latency_s for j in self.jobs], 95)


def evaluate(
    screener: Screener,
    jobs: list[JobDescription],
    labels: Labels,
    pool: list[str],
    retrieve_k: int = 10,
    repeats: int = 1,
) -> EvalReport:
    """Screen each labeled job within the pool and compare with human grades.

    `repeats` > 1 re-runs each job to get a steadier latency p95.
    """
    report = EvalReport()
    for job in jobs:
        grades = labels.get(job.job_id, {})
        if not grades:
            continue
        for _ in range(repeats):
            result = screener.screen(job, retrieve_k=retrieve_k, resume_ids=pool)
            vector_top = [c.resume_id for c in result.candidates[:5]]  # already in vector order
            reranked_top = [c.resume_id for c in result.top(5)]
            report.jobs.append(
                JobEval(
                    job_id=job.job_id,
                    vector_top5=vector_top,
                    reranked_top5=reranked_top,
                    vector_agreement=top_k_agreement(vector_top, grades),
                    reranked_agreement=top_k_agreement(reranked_top, grades),
                    unlabeled_in_top5=sum(r not in grades for r in reranked_top),
                    cost_usd=result.cost_usd,
                    latency_s=result.latency_s,
                )
            )
    return report
