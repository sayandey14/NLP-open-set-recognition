import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from textbuckets.classification import NearestExampleClassifier
from textbuckets.data import load_buckets
from textbuckets.embeddings import DEFAULT_MODEL, SentenceEncoder


@pytest.fixture(scope="module")
def encoder():
    return SentenceEncoder()  # loading the model takes a few seconds; share it


def test_embeddings_match_sentence_transformers(encoder):
    texts = [item.text for item in load_buckets()]
    theirs = SentenceTransformer(DEFAULT_MODEL).encode(texts, normalize_embeddings=True)
    np.testing.assert_allclose(encoder.transform(texts), theirs, atol=1e-5)


def test_embeddings_are_unit_length(encoder):
    X = encoder.transform(["short", "a somewhat longer sentence, so the batch needs padding"])
    assert X.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)


def test_padding_does_not_change_embedding(encoder):
    # The mask must stop padding tokens from leaking into the mean.
    alone = encoder.transform(["hi"])
    batched = encoder.transform(["hi", "this sentence is much longer and forces padding on the first"])[:1]
    np.testing.assert_allclose(alone, batched, atol=1e-5)


def test_paraphrase_beats_word_overlap(encoder):
    # The Phase 1 failure: same meaning, no shared words.
    a, b, c = encoder.transform([
        "I should probably start looking for another position.",
        "I've been considering switching jobs.",
        "I should probably start looking for my keys.",
    ])
    assert a @ b > a @ c


def test_classifier_recognizes_its_own_examples(encoder):
    items = load_buckets()
    clf = NearestExampleClassifier(encoder).fit([i.text for i in items], [i.label for i in items])
    preds = clf.predict([i.text for i in items])
    assert [p.label for p in preds] == [i.label for i in items]
