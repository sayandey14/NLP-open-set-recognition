"""Similarity-based classification.

The classifier doesn't care how texts become vectors: any object with
fit(texts) and transform(texts) works. TF-IDF now, sentence embeddings in Phase 2.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from textbuckets.similarity import cosine_similarity


class Encoder(Protocol):
    def fit(self, texts: list[str]) -> "Encoder": ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class Prediction:
    label: str
    scores: dict[str, float]  # bucket -> similarity to that bucket's closest example


class NearestExampleClassifier:
    """Predict the bucket of the single most similar labeled example (1-nearest-neighbor).

    Note: always returns *some* bucket (argmax). Phase 4 adds the option of UNKNOWN.
    """

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def fit(self, texts: list[str], labels: list[str]) -> "NearestExampleClassifier":
        self.encoder.fit(texts)
        self.vectors_ = self.encoder.transform(texts)
        self.labels_ = np.array(labels)
        self.buckets_ = list(dict.fromkeys(labels))  # unique, in first-seen order
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        sims = cosine_similarity(self.encoder.transform(texts), self.vectors_)  # (queries, examples)
        predictions = []
        for row in sims:
            scores = {b: float(row[self.labels_ == b].max()) for b in self.buckets_}
            predictions.append(Prediction(label=max(scores, key=scores.get), scores=scores))
        return predictions
