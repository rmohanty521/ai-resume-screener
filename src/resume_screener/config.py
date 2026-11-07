"""Settings loaded from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when a required setting is missing."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name} (see .env.example)")
    return value


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    claude_model: str  # fit scoring (needs judgment)
    parse_model: str  # resume extraction (simple, high volume)
    voyage_api_key: str
    embed_model: str
    embed_dim: int
    mongodb_uri: str
    mongodb_db: str
    resumes_collection: str
    vector_index: str

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            anthropic_api_key=_require("ANTHROPIC_API_KEY"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"),
            parse_model=os.getenv("PARSE_MODEL", "claude-haiku-5-5"),
            voyage_api_key=_require("VOYAGE_API_KEY"),
            embed_model=os.getenv("EMBED_MODEL", "voyage-3.5"),
            embed_dim=int(os.getenv("EMBED_DIM", "1024")),
            mongodb_uri=_require("MONGODB_URI"),
            mongodb_db=os.getenv("MONGODB_DB", "resume_screener"),
            resumes_collection=os.getenv("RESUMES_COLLECTION", "resumes"),
            vector_index=os.getenv("VECTOR_INDEX", "resume_vector_index"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings.from_env()
