"""Create and check the Atlas Vector Search index on resume embeddings."""

from __future__ import annotations

import time

from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel


def index_definition(dimensions: int, similarity: str = "cosine") -> dict:
    """Vector field plus filters: `category` (narrow by field) and `resume_id`
    (restrict search to a fixed pool, used by the benchmark)."""
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimensions,
                "similarity": similarity,
            },
            {"type": "filter", "path": "category"},
            {"type": "filter", "path": "resume_id"},
        ]
    }


def get_index(col: Collection, name: str) -> dict | None:
    return next(iter(col.list_search_indexes(name)), None)


def ensure_vector_index(
    col: Collection, name: str, dimensions: int, wait: bool = True, timeout_s: int = 300
) -> dict:
    """Create the index if missing (or update it if the definition changed).

    Atlas builds search indexes in the background; with `wait=True` this
    blocks until the index is queryable.
    """
    try:  # search indexes can only be created on an existing collection
        col.database.create_collection(col.name)
    except CollectionInvalid:
        pass

    definition = index_definition(dimensions)
    existing = get_index(col, name)
    if existing is None:
        col.create_search_index(SearchIndexModel(definition=definition, name=name, type="vectorSearch"))
    elif existing.get("latestDefinition") != definition:
        col.update_search_index(name, definition)

    if not wait:
        return get_index(col, name) or {}

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        index = get_index(col, name) or {}
        if index.get("queryable") and index.get("status") == "READY":
            return index
        if index.get("status") == "FAILED":
            raise RuntimeError(f"Vector index {name!r} failed to build: {index}")
        time.sleep(5)
    raise TimeoutError(f"Vector index {name!r} not ready after {timeout_s}s")
