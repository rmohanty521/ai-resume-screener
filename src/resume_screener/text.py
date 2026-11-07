"""Clean raw resume text before parsing and embedding."""

from __future__ import annotations

import re
import unicodedata

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# US-style (555) 123-4567 / 555.123.4567, or international +CC 98765 43210.
_PHONE = re.compile(
    r"(?<![\w+])(?:"
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}"
    r"|\+\d{1,3}[\s.-]?\d{5}[\s.-]?\d{5}"
    r")(?!\w)"
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def normalize(text: str) -> str:
    """Normalize unicode, drop control characters and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()


def redact_pii(text: str) -> str:
    """Mask emails, phone numbers and URLs so they never reach the model or the DB."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    return _PHONE.sub("[PHONE]", text)


def clean_resume(text: str, max_chars: int | None = 12_000) -> tuple[str, bool]:
    """Return (cleaned_text, was_truncated).

    `max_chars` caps cost per resume. Truncation is reported to the caller
    instead of happening silently; pass None to keep the full text.
    """
    cleaned = redact_pii(normalize(text))
    if max_chars is not None and len(cleaned) > max_chars:
        cut = cleaned.rfind("\n", 0, max_chars)
        return cleaned[: cut if cut > max_chars // 2 else max_chars], True
    return cleaned, False
