"""Score how well retrieved candidates fit a job, with Claude.

Candidates are split into small groups scored in parallel. Latency is driven
by output length, so several short calls finish much sooner than one long one.
`chunk_size=None` scores everyone in one call (slower, slightly cheaper).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, Field

from .parser import fallback_kwargs
from .retrieval import RetrievedResume
from .schemas import FitScore, JobDescription

SYSTEM_PROMPT = """\
You are an experienced technical recruiter screening candidates for one job.

For each candidate, judge role fit using only the job description and the \
candidate's resume data. Scoring guide:
- 85-100: meets nearly all core requirements, with directly relevant experience
- 65-84: meets most core requirements; gaps are learnable
- 40-64: partial or adjacent fit (related field, missing key skills)
- 0-39: different field or missing most requirements

Judge skills and experience only. Ignore names, schools' prestige, gaps in \
employment, and anything that could indicate age, gender or ethnicity.
Return exactly one entry per candidate, using its candidate_id."""


class ScoredCandidate(FitScore):
    candidate_id: str = Field(description="The candidate_id given in the input")


class ScoreBatch(BaseModel):
    scores: list[ScoredCandidate]


class ScoreError(RuntimeError):
    pass


@dataclass
class ScoreResult:
    scores: dict[str, FitScore]  # resume_id -> score
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str


def _candidate_payload(r: RetrievedResume) -> dict:
    p = r.parsed
    return {
        "candidate_id": r.candidate.resume_id,
        "headline": p.headline,
        "summary": p.summary,
        "total_years_experience": p.total_years_experience,
        "skills": p.skills,
        "experience": [
            {"title": e.title, "years": e.years, "summary": e.summary} for e in p.experience
        ],
        "education": [e.degree for e in p.education],
    }


def build_prompt(job: JobDescription, resumes: list[RetrievedResume]) -> str:
    candidates = json.dumps([_candidate_payload(r) for r in resumes], indent=1)
    return (
        f"<job>\nTitle: {job.title}\n{job.description}\n</job>\n\n"
        f"<candidates>\n{candidates}\n</candidates>"
    )


class FitScorer:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-haiku-5-5",
        effort: str = "low",
        chunk_size: int | None = 2,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.chunk_size = chunk_size

    def score(self, job: JobDescription, resumes: list[RetrievedResume]) -> ScoreResult:
        if not resumes:
            return ScoreResult({}, 0, 0, 0.0, self.model)

        size = self.chunk_size or len(resumes)
        chunks = [resumes[i : i + size] for i in range(0, len(resumes), size)]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            parts = list(pool.map(lambda c: self._score_chunk(job, c), chunks))

        # The model occasionally skips a candidate; give those one more try together.
        scored = {k for p in parts for k in p.scores}
        missing = [r for r in resumes if r.candidate.resume_id not in scored]
        if missing:
            parts.append(self._score_chunk(job, missing))

        merged = ScoreResult({}, 0, 0, time.perf_counter() - start, parts[0].model)
        for part in parts:
            merged.scores.update(part.scores)
            merged.input_tokens += part.input_tokens
            merged.output_tokens += part.output_tokens
        return merged

    def _score_chunk(self, job: JobDescription, resumes: list[RetrievedResume]) -> ScoreResult:
        start = time.perf_counter()
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(job, resumes)}],
            output_format=ScoreBatch,
            output_config={"effort": self.effort},
            **fallback_kwargs(self.model),
        )
        latency = time.perf_counter() - start

        if response.stop_reason == "refusal":
            raise ScoreError(f"Model declined to score: {response.stop_details}")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise ScoreError(f"Incomplete scoring output (stop_reason={response.stop_reason})")

        wanted = {r.candidate.resume_id for r in resumes}
        scores: dict[str, FitScore] = {}
        for s in response.parsed_output.scores:
            if s.candidate_id in wanted:  # ignore any id the model invented
                fit = FitScore(**s.model_dump(exclude={"candidate_id"}))
                fit.score = max(0, min(100, fit.score))
                scores[s.candidate_id] = fit

        return ScoreResult(
            scores=scores,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            model=response.model,
        )
