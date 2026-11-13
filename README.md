# Graph ML Platform

A graph machine-learning platform built **from scratch** — no torch, no
numpy, no sklearn. A deterministic synthetic community graph, node
classification (label propagation + from-scratch logistic regression
over aggregated neighbourhood features), link prediction (heuristic
scorers + a learned scorer), node embeddings via random walks and a
small skip-gram trainer with negative sampling, and the full production
stack around them: versioned registry with gated champion promotion,
latency-budgeted serving with baseline fallback, and embedding-drift
monitoring.

## The enterprise problem

Graphs are how enterprises model their real world — customers and
products, accounts and transactions, services and dependencies. The
questions are always the same: classify the nodes you only partly
label, predict the links that do not exist yet, and embed the graph so
downstream systems can use it. Shipping those models is the easy part;
shipping them with leakage-safe splits, promotion gates, fallback
serving and drift monitoring is the part enterprises actually ask about
in Director-level interviews. This repo is that stack, end to end, on a
fully synthetic graph so anyone can run it without real data.

## Architecture

```
synthetic graph (seeded, community-structured, node features + labels)
        │
        ▼
graph.py     — generation, quality checks, leakage screen
splits.py    — transductive-safe node split; edge split (test edges
               removed from the training graph)
features.py  — own + 1-hop mean + 2-hop mean neighbourhood features
        │
        ├─► models.py     — label propagation + softmax logistic
        │                   regression (from scratch) + majority baseline
        ├─► embeddings.py — random walks + skip-gram, negative sampling
        └─► linkpred.py   — common neighbours / Adamic–Adar / Jaccard
                            + learned logistic scorer
        │
        ▼
evaluation.py — accuracy, macro F1, ROC-AUC, Hits@K, MRR
pipeline.py   — train_version(): end-to-end training + stability check
registry.py   — versioned registry, 5 promotion gates, audit history,
                rollback
serving.py    — champion/canary arms, latency budget, baseline fallback
drift.py      — embedding drift: centroid / norm / cosine shift
service.py    — FastAPI: /health /predict-node /predict-link /embed
                /models /stats /promote
```

See [docs/architecture.md](docs/architecture.md) for the full design,
including the data flow, gate definitions, failure modes and
trade-offs.

## How to run

```bash
# Install (core library is pure stdlib; FastAPI/pytest are test+API deps)
pip install -r requirements.txt

# Run the test suite
make test

# End-to-end demo: train v1 + v2, gate the promotion, serve requests,
# score link predictions, show drift on a shifted graph
make demo

# CLI
PYTHONPATH=src python -m graphml.cli train --name v1 --seed 42
PYTHONPATH=src python -m graphml.cli predict-node --node 0 --seed 42
PYTHONPATH=src python -m graphml.cli predict-link --u 0 --v 1 --seed 42
PYTHONPATH=src python -m graphml.cli embed --node 0 --seed 42
PYTHONPATH=src python -m graphml.cli drift --seed 42
PYTHONPATH=src python -m graphml.cli serve --port 8000

# Docker
docker build -t graphml . && docker run -p 8000:8000 graphml
```

API examples (once serving):

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict-node \
  -H 'Content-Type: application/json' \
  -d '{"node": 0, "request_id": "api-1"}'
curl -X POST localhost:8000/predict-link \
  -H 'Content-Type: application/json' \
  -d '{"u": 0, "v": 1}'
curl -X POST localhost:8000/embed \
  -H 'Content-Type: application/json' \
  -d '{"node": 0}'
```

## What it demonstrates

- **From-scratch graph ML, honestly:** label propagation, softmax
  logistic regression with manual gradients, skip-gram with negative
  sampling over random walks, Adamic–Adar / Jaccard / common-neighbour
  link scoring — all pure stdlib, all unit-tested.
- **Leakage-safe graph splits.** Node classification is transductive
  (only labels are partitioned; a screen proves it), link prediction
  is inductive over edges (test edges are removed from the training
  graph). The leakage screen is an audited, tested unit — the classic
  graph-ML bug, made explicit.
- **Real learned improvement.** Logistic regression over aggregated
  neighbourhood features beats the majority-class baseline by a wide
  margin; the learned link scorer beats both random and the best
  single heuristic on Hits@10.
- **Model governance:** versioned JSON snapshots, five promotion gates
  (accuracy floor, must-beat-baseline, per-class recall floor, seed
  stability, link-quality floor), audited promotion history, rollback
  to the previous champion.
- **Safe serving:** hash-bucketed canary traffic, latency budgets with
  a majority-baseline fallback so the service always answers, serving
  stats (arms, fallbacks, latency).
- **Observability:** embedding-drift monitoring (centroid shift, mean
  norm shift, per-node cosine distance) that flags a graph whose
  community structure has dissolved — and scores exactly zero drift on
  a deterministic re-train of the same graph.

## Design decisions and trade-offs

- **Synthetic community graph, not a real network.** Real graphs carry
  privacy and licensing constraints; a seeded stochastic-block-style
  graph gives a controlled, reproducible signal where the ground truth
  is known exactly. The ML and serving stack is identical either way —
  swap `graph.py` for a real loader and the rest is unchanged.
- **Neighbourhood aggregation, not a GNN.** Concatenated own + 1-hop +
  2-hop mean features give a logistic model the graph signal it needs
  without message-passing machinery; a two-layer GCN is the natural
  next step and the embedding + link-scorer modules are where it
  would plug in.
- **Learned link scorer over heuristics + embeddings.** Interpretable,
  fast, and far above random — and it keeps the learned-vs-heuristic
  comparison honest instead of hiding behind a black box. Embedding
  similarity is a valid scorer input only because the pipeline trains
  embeddings on a nested split that excludes the scorer's training
  positives; without it, similarity would leak training edges.
- **Deterministic embeddings.** Same graph + same seed reproduces
  identical embeddings, which is what makes the drift monitor
  meaningful: a zero-drift re-train is the control, flagged drift is
  real structural change.
- **In-memory registry.** Deliberately in-memory with JSON snapshots on
  disk; a production deployment would back it with a database and an
  object store. The gate logic and audit history are unchanged by that
  swap.
- **Majority baseline as fallback *and* gate.** A trivial classifier is
  the degradation path when the model is slow or the input is bad, and
  the "must-beat" bar a challenger has to clear — a model that cannot
  beat a one-line heuristic is not an improvement.

## Project layout

```
src/graphml/
  graph.py       deterministic synthetic graph + quality/leakage checks
  splits.py      transductive-safe node split + edge split
  features.py    neighbourhood-aggregated node features
  models.py      Standardizer, logistic regression, label propagation,
                 majority baseline
  embeddings.py  random walks + skip-gram with negative sampling
  linkpred.py    heuristic scorers + learned link scorer
  evaluation.py  accuracy, macro F1, ROC-AUC, Hits@K, MRR
  pipeline.py    train_version(), JSON model snapshots
  registry.py    versioned registry, promotion gates, rollback
  drift.py       embedding-drift monitoring
  serving.py     champion/canary, latency budgets, fallback
  service.py     FastAPI app
  cli.py         train / predict-node / predict-link / embed / drift /
                 serve / demo
tests/          pytest suite (API tests auto-skip without FastAPI)
scripts/demo.py end-to-end demo
docs/           architecture.md, github-actions-ci.yml
data/           notes on the synthetic corpus
```

## Clean-room note

All code is original, all data is generated in-process from seeded
random draws, and no employer code, data, architecture, or proprietary
design appears anywhere in this repository.
