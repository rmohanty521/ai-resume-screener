from types import SimpleNamespace as NS

import pytest

from resume_screener.retrieval import RetrievedResume
from resume_screener.schemas import Candidate, Experience, ParsedResume


def make_parsed(headline="Backend engineer", skills=("Python", "SQL")) -> ParsedResume:
    return ParsedResume(
        headline=headline,
        summary="Builds web services.",
        skills=list(skills),
        total_years_experience=4,
        experience=[Experience(title="Engineer", company="Acme", years=4, summary="APIs")],
        education=[],
    )


def make_retrieved(resume_id: str, score: float = 0.5) -> RetrievedResume:
    return RetrievedResume(
        candidate=Candidate(resume_id=resume_id, headline="h", vector_score=score),
        parsed=make_parsed(),
        category="INFORMATION-TECHNOLOGY",
    )


def fake_response(parsed_output, stop_reason="end_turn", input_tokens=100, output_tokens=20):
    return NS(
        stop_reason=stop_reason,
        stop_details=None,
        parsed_output=parsed_output,
        usage=NS(input_tokens=input_tokens, output_tokens=output_tokens),
        model="claude-haiku-5-5",
    )


@pytest.fixture
def parsed():
    return make_parsed()
