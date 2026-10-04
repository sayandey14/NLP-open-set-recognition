"""Experiment 00 — look at the data before modeling it.

Question: how much do buckets share *words*? Phase 1 (TF-IDF) can only see
word overlap, so this tells us in advance where it will struggle.

Run:  .venv/bin/python experiments/00_dataset_overview.py
"""

import re
from itertools import combinations

from textbuckets.data import group_by_label, load_buckets, load_unknown_pool


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


groups = group_by_label(load_buckets())
unknown = group_by_label(load_unknown_pool())

print("Known buckets:")
for label, texts in groups.items():
    print(f"  {label:<10} {len(texts)} examples   e.g. {texts[0]!r}")

print("\nUnknown pool (latent topics are hidden from the system):")
for topic, texts in unknown.items():
    print(f"  {topic:<12} {len(texts)} texts")

# Jaccard overlap = |A ∩ B| / |A ∪ B| between the vocabularies of two buckets.
vocab = {label: set().union(*(words(t) for t in texts)) for label, texts in groups.items()}
print("\nVocabulary overlap between buckets (Jaccard), highest first:")
pairs = sorted(
    ((len(vocab[a] & vocab[b]) / len(vocab[a] | vocab[b]), a, b) for a, b in combinations(vocab, 2)),
    reverse=True,
)
for score, a, b in pairs[:8]:
    shared = sorted(vocab[a] & vocab[b])
    print(f"  {score:.2f}  {a} / {b}   shared: {shared[:8]}")
