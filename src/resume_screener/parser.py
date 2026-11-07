"""Turn raw resume text into a structured `ParsedResume` with Claude."""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic

from .schemas import ParsedResume
from .text import clean_resume

SYSTEM_PROMPT = """\
You extract structured data from resumes for a recruiting search index.

Rules:
- Use only facts stated in the resume. Do not guess or embellish.
- Skills: list concrete hard skills, tools, languages and certifications. \
Deduplicate and use common names (e.g. "PostgreSQL", not "postgres db").
- Years: estimate from date ranges when given; use 0 when there is no evidence.
- Personal details were already redacted as [EMAIL], [PHONE], [URL]; ignore them.
- The summary must be neutral and must not mention name, age, gender or ethnicity."""

# Fall back to another model if Claude declines a request (rare for resumes).
# Only some models support server-side fallback; Haiku does not.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


def fallback_kwargs(model: str) -> dict:
    """Request params that enable refusal fallback, if the model supports it."""
    if model in FALLBACK_MODELS:
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    return {}


class ParseError(RuntimeError):
    """Claude did not return a usable parsed resume."""


@dataclass
class ParseResult:
    parsed: ParsedResume
    input_tokens: int
    output_tokens: int
    latency_s: float
    truncated: bool
    model: str


class ResumeParser:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-haiku-5-5",
        effort: str = "low",
        max_chars: int | None = 12_000,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort  # extraction is simple; low effort keeps cost and latency down
        self.max_chars = max_chars

    def parse(self, raw_text: str) -> ParseResult:
        text, truncated = clean_resume(raw_text, self.max_chars)
        if not text:
            raise ParseError("Resume is empty after cleaning")

        start = time.perf_counter()
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"<resume>\n{text}\n</resume>"}],
            output_format=ParsedResume,
            output_config={"effort": self.effort},
            **fallback_kwargs(self.model),
        )
        latency = time.perf_counter() - start

        if response.stop_reason == "refusal":
            raise ParseError(f"Model declined to parse resume: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise ParseError("Output hit max_tokens before the JSON was complete")
        if response.parsed_output is None:
            raise ParseError("No parsed output returned")

        return ParseResult(
            parsed=response.parsed_output,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            truncated=truncated,
            model=response.model,
        )
