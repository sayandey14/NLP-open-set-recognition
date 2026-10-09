# NLP-open-set-recognition

A learning project: organize free-form text into user-defined buckets, say **UNKNOWN** when nothing fits,
cluster the unknowns, and propose new buckets, all with local open-source models (no paid LLM APIs).

```
raw text → chunks → vectors → compare to bucket representations
                                   ├── confident → existing bucket
                                   └── not confident → UNKNOWN → clustering → topic candidate → user confirms → new bucket
```

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python experiments/00_dataset_overview.py
```

(Without uv: `python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.)

## Layout

```
data/
  buckets.csv        labeled seed examples for the known buckets (text, label)
  unknown_pool.csv   texts that fit no bucket (text, latent_topic) — latent_topic is for evaluation only
src/textbuckets/     the library; one module per component, added as each phase needs it
experiments/         numbered, runnable scripts, one per concept
tests/
```

## Buckets

Buckets are data, not code: add a bucket by adding rows to `data/buckets.csv`.

| Bucket   | Meant for                                         |
|----------|---------------------------------------------------|
| Thoughts | abstract musings and reflections                  |
| Feelings | emotional states                                  |
| Journal  | narration of what happened                        |
| Opinions | judgments about things in the world               |
| Work     | job tasks, meetings, colleagues, status           |
| Ideas    | things one could build, make, or try              |
| To-dos   | short actionable tasks                            |

The unknown pool holds hidden topics (appointments, recipes, fitness, plus noise) that later phases should
discover. Note that some of them, like appointments versus To-dos, are deliberately close to an existing bucket.

## Dependencies

Each one is added only in the phase that needs it.

| Package      | Why                                                                   | Since   |
|--------------|-----------------------------------------------------------------------|---------|
| numpy        | vectors, dot products, norms                                          | Phase 0 |
| scikit-learn | TF-IDF, metrics, K-Means/DBSCAN/HDBSCAN (HDBSCAN is built in since 1.3) | Phase 0 |
| sentence-transformers | pretrained embedding models (MiniLM); brings PyTorch + transformers. We call the model through `transformers` and pool by hand; the library itself is used to check our numbers | Phase 2 |
| pytest (dev) | tests                                                                 | Phase 0 |

Planned: `matplotlib` once we need plots.
