import pytest

from resume_screener.evaluate import top_k_agreement
from resume_screener.metrics import RequestMetrics, claude_cost, percentile, summarize
from resume_screener.schemas import Candidate, FitScore, ScreeningResult


def test_costs():
    assert claude_cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    m = RequestMetrics("j", "claude-haiku-5-5", "voyage-3.5", claude_input_tokens=10_000,
                       claude_output_tokens=2_000, embed_tokens=1_000_000)
    assert m.cost_usd == pytest.approx(0.001 + 0.001 + 0.06)


def test_percentile_nearest_rank():
    vals = list(range(1, 21))
    assert percentile(vals, 95) == 19 and percentile(vals, 50) == 10 and percentile([], 95) == 0


def test_summarize():
    s = summarize([{"cost_usd": 0.01, "total_s": t} for t in (1, 2, 3, 10)])
    assert s["requests"] == 4 and s["latency_max_s"] == 10


def test_top_k_agreement():
    grades = {"a": 3, "b": 2, "c": 0, "d": 1}
    assert top_k_agreement(["a", "c", "b", "d", "e"], grades) == 1.0  # both matches found
    assert top_k_agreement(["c", "d", "a", "e", "f"], grades) == 0.5
    assert top_k_agreement(["a"], {"a": 0}) is None


def test_top_ranks_by_fit_then_vector():
    fit = lambda s: FitScore(score=s, matched_skills=[], missing_skills=[], rationale="")
    r = ScreeningResult(job_id="j", candidates=[
        Candidate(resume_id="a", headline="", vector_score=0.9),
        Candidate(resume_id="b", headline="", vector_score=0.5, fit=fit(80)),
        Candidate(resume_id="c", headline="", vector_score=0.6, fit=fit(80)),
    ])
    assert [c.resume_id for c in r.top(3)] == ["c", "b", "a"]
