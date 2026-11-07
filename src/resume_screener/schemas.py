"""Data models shared across the pipeline.

`ParsedResume` and `FitScore` double as Claude structured-output schemas,
so keep their fields simple (strings, numbers, lists) and well described.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Parsed by Claude from raw resume text ---------------------------------


class Experience(BaseModel):
    title: str = Field(description="Job title as written on the resume")
    company: str = Field(description="Employer name, or empty string if not given")
    years: float = Field(description="Approximate years in this role; 0 if unknown")
    summary: str = Field(description="One-sentence summary of responsibilities and impact")


class Education(BaseModel):
    degree: str = Field(description="Degree or certificate, e.g. 'B.S. Computer Science'")
    institution: str = Field(description="School name, or empty string if not given")


class ParsedResume(BaseModel):
    """Structured view of one resume, produced by the parser."""

    headline: str = Field(description="Short professional headline, e.g. 'Backend engineer'")
    summary: str = Field(description="2-3 sentence neutral summary of the candidate")
    skills: list[str] = Field(description="Distinct hard skills, tools and technologies")
    total_years_experience: float = Field(description="Total years of work experience; 0 if unknown")
    experience: list[Experience]
    education: list[Education]


# --- Stored in MongoDB -----------------------------------------------------


class ResumeDocument(BaseModel):
    """One resume as stored in the `resumes` collection."""

    resume_id: str
    category: str = ""  # label from the public dataset, e.g. "INFORMATION-TECHNOLOGY"
    raw_text: str
    parsed: ParsedResume
    embedding: list[float] = Field(default_factory=list)

    def embedding_text(self) -> str:
        """Text that gets embedded for vector search."""
        p = self.parsed
        roles = "; ".join(f"{e.title} at {e.company}".strip() for e in p.experience)
        return (
            f"{p.headline}\n{p.summary}\n"
            f"Skills: {', '.join(p.skills)}\n"
            f"Experience ({p.total_years_experience:g} yrs): {roles}"
        )


# --- Screening -------------------------------------------------------------


class JobDescription(BaseModel):
    job_id: str
    title: str
    description: str


class FitScore(BaseModel):
    """Claude's judgment of how well one resume fits one job."""

    score: int = Field(description="Role fit from 0 (no fit) to 100 (excellent fit)")
    matched_skills: list[str] = Field(description="Required skills the candidate clearly has")
    missing_skills: list[str] = Field(description="Important required skills not evidenced")
    rationale: str = Field(description="2-3 sentence justification for the score")


class Candidate(BaseModel):
    """A retrieved resume plus its scores."""

    resume_id: str
    headline: str
    vector_score: float
    fit: FitScore | None = None


class ScreeningResult(BaseModel):
    job_id: str
    candidates: list[Candidate]
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def top(self, k: int = 5) -> list[Candidate]:
        """Candidates ranked by fit score (falling back to vector score)."""
        return sorted(
            self.candidates,
            key=lambda c: (c.fit.score if c.fit else -1, c.vector_score),
            reverse=True,
        )[:k]
