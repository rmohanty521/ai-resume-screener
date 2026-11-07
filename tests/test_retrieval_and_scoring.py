from types import SimpleNamespace as NS

from resume_screener.retrieval import build_pipeline
from resume_screener.schemas import JobDescription
from resume_screener.scorer import FitScorer, ScoreBatch, ScoredCandidate

from .conftest import fake_response, make_retrieved

JOB = JobDescription(job_id="j", title="Engineer", description="Python")


def test_pipeline_filters():
    stage = build_pipeline([0.1], "idx", 5, 100)[0]["$vectorSearch"]
    assert "filter" not in stage and stage["limit"] == 5
    stage = build_pipeline([0.1], "idx", 5, 100, category="IT", resume_ids=["1"])[0]["$vectorSearch"]
    assert stage["filter"] == {"$and": [{"category": "IT"}, {"resume_id": {"$in": ["1"]}}]}


def scorer_client(drop_first=True):
    calls = []

    def parse(**kw):
        content = kw["messages"][0]["content"]
        ids = [line.split('"')[3] for line in content.splitlines() if '"candidate_id"' in line]
        calls.append(ids)
        keep = ids[:-1] if (drop_first and len(calls) == 1) else ids
        out = ScoreBatch(scores=[
            ScoredCandidate(candidate_id=i, score=150 if i == "0" else 50, matched_skills=[],
                            missing_skills=[], rationale="r")
            for i in keep + ["invented"]
        ])
        return fake_response(out)

    return NS(beta=NS(messages=NS(parse=parse))), calls


def test_scorer_retries_skipped_and_ignores_invented_ids():
    client, calls = scorer_client()
    rs = [make_retrieved(str(i)) for i in range(3)]
    out = FitScorer(client=client, chunk_size=None).score(JOB, rs)
    assert calls == [["0", "1", "2"], ["2"]]
    assert set(out.scores) == {"0", "1", "2"}
    assert out.scores["0"].score == 100  # clamped


def test_scorer_chunks_in_parallel():
    client, calls = scorer_client(drop_first=False)
    out = FitScorer(client=client, chunk_size=2).score(JOB, [make_retrieved(str(i)) for i in range(5)])
    assert sorted(len(c) for c in calls) == [1, 2, 2]
    assert len(out.scores) == 5 and out.input_tokens == 300
