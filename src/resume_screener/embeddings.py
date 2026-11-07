"""Text embeddings with Voyage AI.

Resumes are embedded as "document" and job descriptions as "query", which
Voyage tunes separately for better retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import voyageai

# Voyage accepts up to 1,000 texts per request; smaller batches keep each
# request well under the per-request token limit for long resumes.
BATCH_SIZE = 64


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    total_tokens: int


class Embedder:
    def __init__(
        self,
        client: voyageai.Client | None = None,
        model: str = "voyage-3.5",
        dimension: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.client = client or voyageai.Client(api_key=api_key, max_retries=3)
        self.model = model
        self.dimension = dimension

    def _embed(self, texts: Sequence[str], input_type: str) -> EmbedResult:
        vectors: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[i : i + BATCH_SIZE])
            res = self.client.embed(
                batch,
                model=self.model,
                input_type=input_type,
                output_dimension=self.dimension,
            )
            vectors.extend(res.embeddings)
            tokens += res.total_tokens
        if any(len(v) != self.dimension for v in vectors):
            raise ValueError(f"Expected {self.dimension}-dim vectors from {self.model}")
        return EmbedResult(vectors, tokens)

    def embed_documents(self, texts: Sequence[str]) -> EmbedResult:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> tuple[list[float], int]:
        res = self._embed([text], "query")
        return res.vectors[0], res.total_tokens
