"""Find the resumes closest to a job description with Atlas `$vectorSearch`."""

from __future__ import annotations

from dataclasses import dataclass

from pymongo.collection import Collection

from .embeddings import Embedder
from .schemas import Candidate, JobDescription, ParsedResume


@dataclass
class RetrievedResume:
    candidate: Candidate
    parsed: ParsedResume
    category: str


@dataclass
class RetrievalResult:
    resumes: list[RetrievedResume]
    embed_tokens: int


def job_query_text(job: JobDescription) -> str:
    return f"{job.title}\n{job.description}"


def build_pipeline(
    query_vector: list[float],
    index: str,
    limit: int,
    num_candidates: int,
    category: str | None = None,
    resume_ids: list[str] | None = None,
) -> list[dict]:
    search: dict = {
        "index": index,
        "path": "embedding",
        "queryVector": query_vector,
        # ANN search: consider more candidates than we return for better recall.
        "numCandidates": num_candidates,
        "limit": limit,
    }
    filters = []
    if category:
        filters.append({"category": category})
    if resume_ids is not None:
        filters.append({"resume_id": {"$in": resume_ids}})
    if filters:
        search["filter"] = filters[0] if len(filters) == 1 else {"$and": filters}
    return [
        {"$vectorSearch": search},
        {
            "$project": {
                "_id": 0,
                "resume_id": 1,
                "category": 1,
                "parsed": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]


class Retriever:
    def __init__(self, collection: Collection, embedder: Embedder, index: str) -> None:
        self.col = collection
        self.embedder = embedder
        self.index = index

    def search(
        self,
        job: JobDescription,
        limit: int = 10,
        category: str | None = None,
        resume_ids: list[str] | None = None,
    ) -> RetrievalResult:
        vector, tokens = self.embedder.embed_query(job_query_text(job))
        pipeline = build_pipeline(
            vector,
            self.index,
            limit,
            num_candidates=max(limit * 15, 100),
            category=category,
            resume_ids=resume_ids,
        )
        resumes = []
        for row in self.col.aggregate(pipeline):
            parsed = ParsedResume.model_validate(row["parsed"])
            resumes.append(
                RetrievedResume(
                    candidate=Candidate(
                        resume_id=row["resume_id"],
                        headline=parsed.headline,
                        vector_score=row["score"],
                    ),
                    parsed=parsed,
                    category=row.get("category", ""),
                )
            )
        return RetrievalResult(resumes, tokens)
