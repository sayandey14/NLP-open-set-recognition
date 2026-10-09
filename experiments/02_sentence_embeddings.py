"""Experiment 02 — sentence embeddings vs TF-IDF.

Same questions as experiment 01, asked of a pretrained transformer (all-MiniLM-L6-v2):
  A. What does an embedding look like, and are token vectors really contextual?
  B. Leave-one-out accuracy: TF-IDF vs embeddings, side by side.
  C. Do the paraphrase / shared-word pairs from 01 now come out right?
  D. What does argmax do with text that fits no bucket? (Spoiler: still picks one.)

Run:  .venv/bin/python experiments/02_sentence_embeddings.py
(First run downloads the model, ~90MB, to ~/.cache/huggingface; after that it's offline.)
"""

import numpy as np

from textbuckets.classification import NearestExampleClassifier
from textbuckets.data import load_buckets
from textbuckets.embeddings import SentenceEncoder
from textbuckets.similarity import cosine_similarity
from textbuckets.tfidf import TfidfEncoder

items = load_buckets()
texts = [i.text for i in items]
labels = [i.label for i in items]
enc = SentenceEncoder()


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --- A. What an embedding is --------------------------------------------------
section("A. A dense vector, built from contextual token vectors")
example = "I feel really anxious and I can't pin down why."
vec = enc.transform([example])[0]
print(f"Embedding of {example!r}")
print(f"  {vec.shape[0]} dimensions, {np.count_nonzero(vec)} non-zero  (dense: compare TF-IDF's handful)")
print(f"  first 8 values: {np.round(vec[:8], 3)}   (no single dimension 'means' anything)")

# The same word gets a different vector depending on its sentence.
# TF-IDF can't do this: "bank" is one column no matter what surrounds it.
bank_sentences = [
    "I need to go to the bank to deposit money.",
    "The bank approved my loan application.",
    "We had a picnic on the river bank.",
]
bank_id = enc.tokenizer.convert_tokens_to_ids("bank")
hidden, _ = enc.token_vectors(bank_sentences)
input_ids = enc.tokenizer(bank_sentences, padding=True, return_tensors="pt")["input_ids"]
bank_vecs = np.vstack([hidden[i][input_ids[i] == bank_id][0].numpy() for i in range(len(bank_sentences))])
print("\nCosine between the token vectors of 'bank' in different sentences:")
sims = cosine_similarity(bank_vecs, bank_vecs)
print(f"  money bank vs loan bank:  {sims[0, 1]:.2f}")
print(f"  money bank vs river bank: {sims[0, 2]:.2f}")


# --- B. Leave-one-out accuracy, both encoders ---------------------------------
section("B. Leave-one-out accuracy: TF-IDF vs embeddings")


def leave_one_out_tfidf():
    errors = []
    for i in range(len(texts)):
        clf = NearestExampleClassifier(TfidfEncoder()).fit(texts[:i] + texts[i + 1 :], labels[:i] + labels[i + 1 :])
        pred = clf.predict([texts[i]])[0]
        if pred.label != labels[i]:
            errors.append((labels[i], pred.label, texts[i]))
    return errors


def leave_one_out_embeddings():
    # The embedding model learns nothing from our data, so "leaving one out" just means
    # not letting an example match itself: embed once, then blank the diagonal.
    sims = cosine_similarity(enc.transform(texts), enc.transform(texts))
    np.fill_diagonal(sims, -np.inf)
    label_arr = np.array(labels)
    errors = []
    for i, row in enumerate(sims):
        pred = label_arr[row.argmax()]
        if pred != labels[i]:
            errors.append((labels[i], pred, texts[i]))
    return errors


results = {"TF-IDF": leave_one_out_tfidf(), "Embeddings": leave_one_out_embeddings()}
for name, errors in results.items():
    print(f"  {name:<11} {len(texts) - len(errors)}/{len(texts)} = {1 - len(errors) / len(texts):.0%}")

print("\nEmbedding mistakes  (true -> predicted):")
for true, pred, text in results["Embeddings"]:
    print(f"  {true:>8} -> {pred:<8} {text}")


# --- C. The pairs that broke TF-IDF -----------------------------------------
section("C. Lexical similarity vs meaning, revisited")
should_be_similar = [
    ("I should probably start looking for another position.", "I've been considering switching jobs."),
    ("I am frustrated with my job.", "Work has been really annoying lately."),
    ("The meeting got pushed to next week.", "We rescheduled the sync for later."),
]
should_be_different = [
    ("I need to go to the bank to deposit money.", "We had a picnic on the river bank."),
    ("I am happy with my job.", "I am not happy with my job."),
    ("The dog bit the man.", "The man bit the dog."),
]
pair_texts = [t for pair in should_be_similar + should_be_different for t in pair]
tfidf = TfidfEncoder().fit(texts + pair_texts)


def show(pairs):
    print(f"  {'TF-IDF':>6}  {'Embed':>6}")
    for a, b in pairs:
        t = cosine_similarity(tfidf.transform([a]), tfidf.transform([b]))[0, 0]
        e = cosine_similarity(enc.transform([a]), enc.transform([b]))[0, 0]
        print(f"  {t:>6.2f}  {e:>6.2f}   {a!r}\n                  {b!r}")


print("Same meaning, different words (we WANT high similarity):")
show(should_be_similar)
print("\nShared words, different meaning (we WANT low similarity):")
show(should_be_different)


# --- D. Argmax on new text ----------------------------------------------------
section("D. Classifying new text (argmax still always picks something)")
clf = NearestExampleClassifier(enc).fit(texts, labels)
new_texts = [
    "I've been thinking about building a tool that automatically organizes all my notes. "
    "I could probably use embeddings and cluster the stuff that doesn't fit existing categories.",
    "Everything feels heavy and pointless lately.",
    "I need to book my dentist, renew my passport, and schedule my driving test.",
    "Quantum chromodynamics describes the strong interaction.",
]
for text, pred in zip(new_texts, clf.predict(new_texts)):
    print(f"\n{text!r}")
    for bucket, score in sorted(pred.scores.items(), key=lambda kv: -kv[1]):
        print(f"    {bucket:<10} {score:.2f}")
    print(f"  -> {pred.label}")
