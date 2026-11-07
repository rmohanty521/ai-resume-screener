"""Per-request cost and latency tracking, logged to JSONL."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_LOG = Path("data/metrics/requests.jsonl")

# USD per 1M tokens: (input, output). Prompts under 100K tokens for Haiku 5.5.
CLAUDE_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
# USD per 1M tokens. Voyage gives 200M free tokens, so real spend is usually $0.
VOYAGE_PRICES: dict[str, float] = {"voyage-3.5": 0.06, "voyage-3.5-lite": 0.02}


def claude_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = CLAUDE_PRICES.get(model, (0.0, 0.0))
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


def voyage_cost(model: str, tokens: int) -> float:
    return tokens / 1e6 * VOYAGE_PRICES.get(model, 0.0)


@dataclass
class RequestMetrics:
    job_id: str
    claude_model: str
    embed_model: str
    candidates: int = 0
    claude_input_tokens: int = 0
    claude_output_tokens: int = 0
    embed_tokens: int = 0
    retrieval_s: float = 0.0
    scoring_s: float = 0.0
    total_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def cost_usd(self) -> float:
        return claude_cost(
            self.claude_model, self.claude_input_tokens, self.claude_output_tokens
        ) + voyage_cost(self.embed_model, self.embed_tokens)

    def to_dict(self) -> dict:
        return {**asdict(self), "cost_usd": round(self.cost_usd, 6)}


def log_request(m: RequestMetrics, path: Path | str = DEFAULT_LOG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(m.to_dict()) + "\n")


def load_requests(path: Path | str = DEFAULT_LOG) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0-100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"requests": 0}
    costs = [r["cost_usd"] for r in rows]
    latencies = [r["total_s"] for r in rows]
    return {
        "requests": len(rows),
        "cost_mean_usd": sum(costs) / len(costs),
        "cost_p95_usd": percentile(costs, 95),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_max_s": max(latencies),
    }
