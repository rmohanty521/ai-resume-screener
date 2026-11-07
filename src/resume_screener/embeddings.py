"""Text embeddings with Voyage AI.

Resumes are embedded as "document" and job descriptions as "query", which
Voyage tunes separately for better retrieval.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import voyageai
from voyageai.error import RateLimitError

log = logging.getLogger(__name__)

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

    def _embed_batch(self, batch: list[str], input_type: str, max_attempts: int = 8):
        """One request, waiting out rate limits.

        Voyage's free tier (no payment method) allows only 3 requests/min and
        10K tokens/min, so waits are long enough to clear a full minute window.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                return self.client.embed(
                    batch,
                    model=self.model,
                    input_type=input_type,
                    output_dimension=self.dimension,
                )
            except RateLimitError:
                if attempt == max_attempts:
                    raise
                wait = min(20 * attempt, 65)
                log.warning("Voyage rate limit; retrying in %ss (attempt %s)", wait, attempt)
                time.sleep(wait)

    def _embed(self, texts: Sequence[str], input_type: str) -> EmbedResult:
        vectors: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), BATCH_SIZE):
            res = self._embed_batch(list(texts[i : i + BATCH_SIZE]), input_type)
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
