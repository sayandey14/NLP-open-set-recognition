"""Sentence embeddings from a pretrained transformer, with the pooling written out by hand.

TF-IDF gives every *word* its own dimension, so two texts are similar only if
they share words. A sentence embedding model maps a whole text to a dense
vector (384 numbers for MiniLM) whose dimensions have no individual meaning;
the model was trained so that texts with similar meaning land close together.

How a text becomes one vector:

    "I feel anxious"
      -> tokenizer   -> [CLS] i feel anxious [SEP]           (word pieces -> ids)
      -> transformer -> one 384-d vector per token            (contextual: each token
                                                               vector depends on its neighbors)
      -> mean pool   -> average of the token vectors          (one vector per text)
      -> L2 normalize                                         (length 1, so cosine = dot)

Mean pooling, with padding masked out (a batch pads short texts to equal length):

    e = sum_i(mask_i * h_i) / sum_i(mask_i)      h_i = token i's vector, mask_i in {0, 1}

The `sentence-transformers` library does exactly these steps in model.encode();
tests/test_phase2.py checks that ours gives the same numbers.

Unlike TF-IDF, fit() learns nothing: the model is already trained on ~1B
sentence pairs. fit() exists only so this matches the Encoder protocol.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from textbuckets.similarity import l2_normalize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()  # eval(): turn off dropout

    def fit(self, texts: list[str]) -> "SentenceEncoder":
        return self  # nothing to learn; the model is pretrained

    def token_vectors(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, tokens, dim) contextual token vectors and the (batch, tokens) padding mask."""
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():  # inference only: don't track gradients
            out = self.model(**batch)
        return out.last_hidden_state, batch["attention_mask"]

    def transform(self, texts: list[str]) -> np.ndarray:
        """Texts -> (n_texts, dim) matrix of unit-length embeddings."""
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            hidden, mask = self.token_vectors(texts[start : start + self.batch_size])
            mask = mask.unsqueeze(-1).float()  # (batch, tokens, 1) so it broadcasts over dim
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled.numpy())
        return l2_normalize(np.vstack(chunks))
