"""Load resumes from the public Kaggle "Resume Dataset" CSV.

Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
Download it and place `Resume.csv` at `data/raw/Resume.csv`.
Columns used: `ID`, `Resume_str`, `Category`.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_PATH = Path("data/raw/Resume.csv")

# Resume text fields can be larger than csv's default 128 KB limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass(frozen=True)
class RawResume:
    resume_id: str
    category: str
    text: str


def iter_resumes(
    path: Path | str = DEFAULT_PATH,
    id_col: str = "ID",
    text_col: str = "Resume_str",
    category_col: str = "Category",
) -> Iterator[RawResume]:
    """Yield resumes from the CSV, skipping rows with empty text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle Resume Dataset and put Resume.csv there."
        )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {id_col, text_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for row in reader:
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            yield RawResume(
                resume_id=str(row[id_col]).strip(),
                category=(row.get(category_col) or "").strip(),
                text=text,
            )


def load_resumes(path: Path | str = DEFAULT_PATH, **kwargs) -> list[RawResume]:
    return list(iter_resumes(path, **kwargs))


def stratified_sample(resumes: list[RawResume], n: int, seed: int = 42) -> list[RawResume]:
    """Pick about `n` resumes spread evenly across categories (deterministic)."""
    if n >= len(resumes):
        return list(resumes)
    rng = random.Random(seed)
    by_cat: dict[str, list[RawResume]] = defaultdict(list)
    for r in resumes:
        by_cat[r.category].append(r)
    for group in by_cat.values():
        rng.shuffle(group)

    # Round-robin across categories so small ones are still represented.
    picked: list[RawResume] = []
    cats = sorted(by_cat)
    while len(picked) < n:
        for cat in cats:
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop())
    return picked
