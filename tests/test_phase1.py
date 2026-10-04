import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from textbuckets.classification import NearestExampleClassifier
from textbuckets.data import load_buckets
from textbuckets.preprocessing import tokenize
from textbuckets.similarity import cosine_similarity
from textbuckets.tfidf import TfidfEncoder


def test_tokenize():
    assert tokenize("I've been SO tired, honestly!") == ["i've", "been", "so", "tired", "honestly"]


def test_tfidf_matches_sklearn():
    texts = [item.text for item in load_buckets()]
    ours = TfidfEncoder().fit(texts)
    theirs = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False).fit(texts)
    assert ours.vocabulary_ == theirs.vocabulary_
    np.testing.assert_allclose(ours.idf_, theirs.idf_)
    np.testing.assert_allclose(ours.transform(texts), theirs.transform(texts).toarray())


def test_cosine_properties():
    a = np.array([[1.0, 2.0, 0.0]])
    assert np.isclose(cosine_similarity(a, a)[0, 0], 1.0)  # same direction
    assert np.isclose(cosine_similarity(a, 10 * a)[0, 0], 1.0)  # length doesn't matter
    assert np.isclose(cosine_similarity(a, np.array([[0.0, 0.0, 5.0]]))[0, 0], 0.0)  # no shared dims
    assert cosine_similarity(a, np.zeros((1, 3)))[0, 0] == 0.0  # zero vector


def test_classifier_recognizes_its_own_examples():
    items = load_buckets()
    clf = NearestExampleClassifier(TfidfEncoder()).fit([i.text for i in items], [i.label for i in items])
    preds = clf.predict([i.text for i in items])
    assert [p.label for p in preds] == [i.label for i in items]
