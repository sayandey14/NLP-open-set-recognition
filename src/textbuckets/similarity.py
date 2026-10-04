"""Vector similarity.

cos(A, B) = (A · B) / (||A|| ||B||)

If every row is first scaled to length 1, the denominator is 1 and cosine
similarity is just a dot product, which is how it is computed here.
"""

import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit length. All-zero rows stay all-zero."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity: result[i, j] = cos(A[i], B[j]).

    A zero vector (e.g. a text with no known words) has similarity 0 to everything.
    """
    return l2_normalize(A) @ l2_normalize(B).T
