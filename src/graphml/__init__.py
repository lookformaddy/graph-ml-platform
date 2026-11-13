"""Graph ML platform — from-scratch graph machine learning, pure stdlib.

Modules:
    graph       deterministic synthetic graph generation + quality checks
    splits      transductive-safe node splits and edge splits
    features    neighbourhood-aggregated node features
    models      label propagation, from-scratch logistic regression, baselines
    embeddings  random walks + skip-gram with negative sampling
    linkpred    heuristic and learned link-prediction scorers
    evaluation  accuracy, macro F1, ROC-AUC, Hits@K, MRR
    pipeline    train_version(): end-to-end versioned training
    registry    versioned registry with gated champion promotion
    drift       embedding-drift monitoring (centroid / norm shift)
    serving     latency-budgeted serving layer with baseline fallback
    service     FastAPI service
    cli         command-line interface
"""

from graphml.graph import Graph, check_quality, generate_graph, leakage_screen

__all__ = ["Graph", "check_quality", "generate_graph", "leakage_screen"]
__version__ = "1.0.0"
