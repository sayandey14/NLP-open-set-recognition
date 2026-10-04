"""Loading the example datasets.

Two files live in data/:

- buckets.csv       (text, label)         labeled seed examples for the KNOWN buckets.
- unknown_pool.csv  (text, latent_topic)  texts that belong to NO known bucket.

The `latent_topic` column is hidden ground truth. The system must never use it
to make decisions; we only use it later to *evaluate* whether clustering
recovered sensible groups (cluster purity, Phase 9).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Label used when the system decides a text fits no known bucket.
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LabeledText:
    text: str
    label: str


def _read_csv(path: Path, label_column: str) -> list[LabeledText]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            LabeledText(text=row["text"].strip(), label=row[label_column].strip())
            for row in csv.DictReader(f)
        ]


def load_buckets(path: Path = DATA_DIR / "buckets.csv") -> list[LabeledText]:
    """Seed examples for the known buckets."""
    return _read_csv(path, "label")


def load_unknown_pool(path: Path = DATA_DIR / "unknown_pool.csv") -> list[LabeledText]:
    """Texts outside every known bucket; `label` holds the hidden latent topic."""
    return _read_csv(path, "latent_topic")


def group_by_label(items: list[LabeledText]) -> dict[str, list[str]]:
    """{label: [text, ...]} preserving file order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        groups[item.label].append(item.text)
    return dict(groups)
