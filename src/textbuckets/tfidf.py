"""Hand-written TF-IDF encoder.

Matches sklearn's TfidfVectorizer defaults so we can check our understanding
against it (tests/test_tfidf.py):

    tf(t, d)  = number of times term t appears in document d
    idf(t)    = ln((1 + n) / (1 + df(t))) + 1      n = number of documents
                                                   df(t) = documents containing t
    tfidf     = tf * idf, then each row scaled to length 1 (L2 norm)

The "1 +" terms ("smoothing") avoid division by zero, and the trailing "+ 1"
keeps words that appear in every document from getting weight exactly 0.

Vectors are dense numpy arrays here because our data is tiny. Real TF-IDF
matrices are mostly zeros, so sklearn stores them as sparse matrices.
"""

from collections.abc import Callable

import numpy as np

from textbuckets.preprocessing import tokenize
from textbuckets.similarity import l2_normalize


class TfidfEncoder:
    def __init__(self, tokenizer: Callable[[str], list[str]] = tokenize):
        self.tokenizer = tokenizer

    def fit(self, texts: list[str]) -> "TfidfEncoder":
        """Learn the vocabulary and each word's IDF from a corpus."""
        docs = [self.tokenizer(t) for t in texts]
        vocab = sorted({word for doc in docs for word in doc})
        self.vocabulary_ = {word: i for i, word in enumerate(vocab)}

        df = np.zeros(len(vocab))
        for doc in docs:
            for word in set(doc):  # set: count each document once per word
                df[self.vocabulary_[word]] += 1
        n = len(docs)
        self.idf_ = np.log((1 + n) / (1 + df)) + 1
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Texts -> (n_texts, vocab_size) matrix. Words not seen in fit() are ignored."""
        X = np.zeros((len(texts), len(self.vocabulary_)))
        for row, text in enumerate(texts):
            for word in self.tokenizer(text):
                col = self.vocabulary_.get(word)
                if col is not None:
                    X[row, col] += 1
        return l2_normalize(X * self.idf_)
