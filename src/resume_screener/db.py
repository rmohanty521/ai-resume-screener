"""MongoDB Atlas access: connection plus a small repository for resumes."""

from __future__ import annotations

from typing import Iterable

import certifi
from pymongo import MongoClient, ReplaceOne
from pymongo.collection import Collection

from .schemas import ResumeDocument


def connect(uri: str, timeout_ms: int = 10_000) -> MongoClient:
    """Connect to Atlas and fail fast if it is unreachable.

    `certifi` supplies CA certificates, which python.org builds on macOS lack
    (otherwise: "CERTIFICATE_VERIFY_FAILED").
    """
    client: MongoClient = MongoClient(
        uri,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=timeout_ms,
        appname="resume-screener",
    )
    client.admin.command("ping")
    return client


class ResumeRepository:
    """CRUD for the `resumes` collection. `resume_id` is the unique key."""

    def __init__(self, collection: Collection) -> None:
        self.col = collection

    @classmethod
    def from_settings(cls, settings) -> "ResumeRepository":
        client = connect(settings.mongodb_uri)
        return cls(client[settings.mongodb_db][settings.resumes_collection])

    def ensure_indexes(self) -> None:
        self.col.create_index("resume_id", unique=True)
        self.col.create_index("category")

    def upsert_many(self, docs: Iterable[ResumeDocument]) -> int:
        """Insert or replace resumes by `resume_id`; returns how many were written."""
        ops = [
            ReplaceOne({"resume_id": d.resume_id}, d.model_dump(), upsert=True) for d in docs
        ]
        if not ops:
            return 0
        result = self.col.bulk_write(ops, ordered=False)
        return result.upserted_count + result.matched_count

    def get(self, resume_id: str) -> ResumeDocument | None:
        raw = self.col.find_one({"resume_id": resume_id}, {"_id": 0})
        return ResumeDocument.model_validate(raw) if raw else None

    def get_many(self, resume_ids: list[str]) -> list[ResumeDocument]:
        found = {
            d["resume_id"]: ResumeDocument.model_validate(d)
            for d in self.col.find({"resume_id": {"$in": resume_ids}}, {"_id": 0})
        }
        return [found[i] for i in resume_ids if i in found]

    def existing_ids(self) -> set[str]:
        return set(self.col.distinct("resume_id"))

    def count(self) -> int:
        return self.col.count_documents({})

    def delete_all(self) -> int:
        return self.col.delete_many({}).deleted_count
