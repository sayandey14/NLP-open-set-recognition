"""Experiment 01 — TF-IDF + cosine similarity + nearest-neighbor classification.

Questions:
  A. Which words does IDF consider important vs unimportant?
  B. How accurate is a TF-IDF nearest-neighbor classifier on our buckets?
  C. Where does lexical (word-overlap) similarity fail?
  D. What does argmax do with text that fits no bucket?

Run:  .venv/bin/python experiments/01_tfidf_baseline.py
"""

from textbuckets.classification import NearestExampleClassifier
from textbuckets.data import load_buckets
from textbuckets.similarity import cosine_similarity
from textbuckets.tfidf import TfidfEncoder

items = load_buckets()
texts = [i.text for i in items]
labels = [i.label for i in items]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --- A. IDF weights ---------------------------------------------------------
section("A. IDF: rare words get high weight, common words get low weight")
enc = TfidfEncoder().fit(texts)
by_idf = sorted(enc.vocabulary_, key=lambda w: enc.idf_[enc.vocabulary_[w]])
print(f"{len(texts)} documents, {len(by_idf)} distinct words")
print("Lowest IDF (most common):", [(w, round(enc.idf_[enc.vocabulary_[w]], 2)) for w in by_idf[:8]])
print("Highest IDF (rarest):    ", [(w, round(enc.idf_[enc.vocabulary_[w]], 2)) for w in by_idf[-5:]])

example = "I feel really anxious and I can't pin down why."
vec = enc.transform([example])[0]
nonzero = sorted(((vec[i], w) for w, i in enc.vocabulary_.items() if vec[i] > 0), reverse=True)
print(f"\nTF-IDF vector of {example!r}")
print(f"  {len(nonzero)} non-zero entries out of {len(vec)}  (that's what 'sparse' means)")
for weight, word in nonzero:
    print(f"  {word:<10} {weight:.3f}")


# --- B. Leave-one-out accuracy ----------------------------------------------
section("B. Leave-one-out accuracy (each example classified by a model that never saw it)")
errors = []
for i in range(len(texts)):
    train_texts, train_labels = texts[:i] + texts[i + 1 :], labels[:i] + labels[i + 1 :]
    clf = NearestExampleClassifier(TfidfEncoder()).fit(train_texts, train_labels)
    pred = clf.predict([texts[i]])[0]
    if pred.label != labels[i]:
        errors.append((labels[i], pred.label, pred.scores[pred.label], texts[i]))

print(f"Accuracy: {len(texts) - len(errors)}/{len(texts)} = {1 - len(errors) / len(texts):.0%}")
print(f"(Random guessing among {len(set(labels))} buckets would get ~{1 / len(set(labels)):.0%})")
print("\nMistakes  (true -> predicted, similarity of the winning match):")
for true, pred, score, text in errors:
    print(f"  {true:>8} -> {pred:<8} {score:.2f}  {text}")


# --- C. Where lexical similarity fails --------------------------------------
section("C. Lexical similarity vs meaning")
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
# Fit IDF on the buckets plus these sentences so every word has a weight.
pair_enc = TfidfEncoder().fit(texts + pair_texts)


def show(pairs):
    for a, b in pairs:
        sim = cosine_similarity(pair_enc.transform([a]), pair_enc.transform([b]))[0, 0]
        print(f"  {sim:.2f}   {a!r}\n         {b!r}")


print("Same meaning, different words (we WANT high similarity):")
show(should_be_similar)
print("\nShared words, different meaning (we WANT low similarity):")
show(should_be_different)


# --- D. Argmax on new text ----------------------------------------------------
section("D. Classifying new text (argmax always picks something)")
clf = NearestExampleClassifier(TfidfEncoder()).fit(texts, labels)
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
