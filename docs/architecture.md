# Graph ML Platform — Architecture

## Components

```
synthetic graph (seeded SBM-style: 4 communities, node features, labels)
        │
        ▼
graph.py        — generation + quality checks (isolation, degree skew,
                  label balance) + leakage screen
        │
        ├─► splits.py — stratified node split (transductive-safe:
        │               only *labels* are partitioned) and edge split
        │               (held-out edges removed from the training graph)
        ▼
features.py     — neighbourhood aggregation: own + 1-hop mean + 2-hop mean
        │
        ├─► models.py — label propagation (clamps train labels only)
        │               + multinomial logistic regression (from-scratch
        │               softmax GD) + majority-class baseline
        │
        ├─► embeddings.py — random walks + skip-gram with negative
        │                   sampling (from scratch, deterministic)
        │
        └─► linkpred.py — common neighbours / Adamic–Adar / Jaccard /
                          preferential attachment + a learned logistic
                          scorer over the heuristic features
        │
        ▼
evaluation.py   — accuracy, macro F1, ROC-AUC (one-vs-rest), Hits@K, MRR
        │
        ▼
pipeline.py     — train_version(): graph → splits → models → metrics →
                  JSON-serialisable ModelVersion snapshot + stability
                  check across re-training seeds
        │
        ▼
registry.py     — versioned registry; promotion gates (accuracy floor,
                  must-beat-baseline, per-class recall floor, seed
                  stability, link Hits@K floor) + audited history +
                  rollback
        │
        ▼
serving.py      — champion / canary (hash-bucketed) arms, latency
                  budget with majority-baseline fallback, serving stats
drift.py        — embedding-drift monitor: centroid shift, mean-norm
                  shift, per-node cosine distance vs. the champion's
                  reference embeddings (no labels needed)
        │
        ▼
service.py      — FastAPI: /health /predict-node /predict-link /embed
                  /models /stats /promote
cli.py          — train / predict-node / predict-link / embed / drift /
                  serve / demo
```

## Data flow

1. **Generate.** `generate_graph(seed)` produces the community graph,
   node features (community centre + Gaussian noise) and labels
   (community id, 5% label noise). `check_quality` rejects graphs with
   isolated nodes or empty classes.
2. **Split.** Node ids are stratified by label into train/val/test;
   the leakage screen proves train/test disjointness and that no
   feature column encodes the label. For link prediction, 15% of edges
   are removed from the training graph — no scorer input can see a
   test edge.
3. **Train.** Label propagation and logistic regression learn node
   classification; skip-gram embeddings are learned on the training
   graph's random walks; the learned link scorer is trained on observed
   edges vs. sampled non-edges.
4. **Evaluate.** Held-out metrics (accuracy, macro F1, ROC-AUC,
   Hits@10, MRR) plus a multi-seed stability re-train.
5. **Gate & promote.** The registry promotes a challenger only if all
   five gates pass; the audit history records every decision.
6. **Serve & monitor.** The serving layer enforces a latency budget
   with baseline fallback; the drift monitor scores every new
   embedding set against the champion's reference.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| Isolated nodes / degenerate graph | `check_quality` | training refuses to start |
| Test-label leakage | `leakage_screen` | training refuses to start |
| Challenger below accuracy floor | registry gate | promotion rejected, audit entry kept |
| Challenger worse than baseline | registry gate | promotion rejected |
| One class sacrificed | per-class recall gate | promotion rejected |
| Unstable across seeds | seed-std gate | promotion rejected |
| Link quality at random | Hits@K gate | promotion rejected |
| Serving latency over budget | per-request timer | majority-baseline fallback |
| Unknown node / bad pair | serving validation | error arm, counted in stats |
| Graph structure shifts | drift monitor | flagged report; no auto-rollback (human decision) |
| Champion regression discovered later | audit history | `rollback()` to previous champion |

## Design decisions and trade-offs

- **Pure stdlib core, no torch/numpy/sklearn.** Every model is
  implemented by hand (softmax GD, label propagation, skip-gram with
  negative sampling) so the mathematics is inspectable and the test
  suite can verify it directly. The cost is speed: graphs are kept at
  a few hundred nodes, which is the right scale for a demonstration
  platform; a production deployment would swap in a compiled backend
  behind the same interfaces.
- **Transductive node split, inductive edge split.** Node
  classification follows the standard transductive setup (the graph is
  visible, labels are partitioned); link prediction is strictly
  inductive over edges (test edges are removed). Mixing the two up is
  the classic graph-ML leakage bug, so the split code and the leakage
  screen are separate, tested units.
- **Learned link scorer over heuristics + embeddings, not a GNN.** A
  two-layer GCN is the obvious next step, but a logistic scorer on
  interpretable features is explainable, fast, and already far above
  random — and it keeps the learned-vs-heuristic comparison honest.
  Embedding similarity is a scorer input only because the pipeline
  trains embeddings on a nested split that excludes the scorer's
  training positives; without that split the similarity would leak
  training edges and the scorer would overfit to it. That nested split
  is the difference between a model that looks good and one that is
  good.
- **Deterministic embeddings make drift monitoring meaningful.**
  Same graph + same seed reproduces identical embeddings, so a
  zero-drift re-train is the control and any flagged drift is real
  structural change, not training noise.
- **In-memory registry with JSON snapshots.** Deliberate: the gate
  logic and audit history are the point, and they survive unchanged
  when the registry is backed by a database and object store.
- **Latency budget with baseline fallback, not a timeout error.** A
  slow model still has to answer; the majority baseline is always
  available and the fallback is counted, so degraded mode is visible
  in `/stats` rather than silent.
