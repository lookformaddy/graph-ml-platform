"""Command-line interface for the graph ML platform.

Subcommands:
    train        train a version and print its report
    predict-node classify one node id
    predict-link score one node pair for link existence
    embed        print one node's embedding (first 8 dims)
    drift        score embedding drift of a shifted graph
    serve        start the FastAPI service
    demo         run the end-to-end demo
"""

from __future__ import annotations

import argparse
import json
import sys

from graphml.drift import DriftMonitor
from graphml.embeddings import SkipGramEmbeddings, random_walks
from graphml.graph import check_quality, generate_graph
from graphml.pipeline import train_version
from graphml.registry import ModelRegistry
from graphml.serving import ServingConfig, ServingLayer
from graphml.splits import edge_split


def _build_layer(seed: int = 42, n_nodes: int = 200) -> ServingLayer:
    version, _report = train_version("v1-seed42", seed=seed, n_nodes=n_nodes)
    registry = ModelRegistry()
    registry.register(version, role="champion")
    graph = generate_graph(n_nodes=n_nodes, seed=seed)
    monitor = DriftMonitor(
        {int(k): list(v) for k, v in version.embeddings.items()})
    return ServingLayer(registry, ServingConfig(), graph=graph,
                        drift_monitor=monitor)


def cmd_train(args: argparse.Namespace) -> int:
    version, report = train_version(args.name, seed=args.seed,
                                    n_nodes=args.nodes)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(version.to_json())
        print(f"snapshot written to {args.out}")
    return 0


def cmd_predict_node(args: argparse.Namespace) -> int:
    layer = _build_layer(seed=args.seed)
    pred = layer.predict_node(args.node, request_id=args.request_id)
    print(json.dumps(pred.to_dict(), indent=2))
    return 0


def cmd_predict_link(args: argparse.Namespace) -> int:
    layer = _build_layer(seed=args.seed)
    pred = layer.predict_link(args.u, args.v, request_id=args.request_id)
    print(json.dumps(pred.to_dict(), indent=2))
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    layer = _build_layer(seed=args.seed)
    pred = layer.embed(args.node)
    data = pred.to_dict()
    if isinstance(data["value"], list):
        data["value"] = data["value"][:8]
        data["value_note"] = "first 8 of 16 dims shown"
    print(json.dumps(data, indent=2))
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    layer = _build_layer(seed=args.seed)
    shifted = generate_graph(n_nodes=200, p_in=args.p_in, p_out=args.p_out,
                             seed=args.seed)
    train_graph, _edges, _negs = edge_split(shifted, seed=args.seed)
    walks = random_walks(train_graph, seed=args.seed)
    emb = SkipGramEmbeddings(epochs=2, seed=args.seed)
    emb.fit(train_graph, walks=walks)
    report = layer.drift_monitor.score(emb.all_embeddings())
    print(json.dumps(report, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    layer = _build_layer(seed=args.seed)
    from graphml.service import build_app
    uvicorn.run(build_app(layer), host=args.host, port=args.port)
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    from scripts.demo import run_demo
    run_demo()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphml",
                                     description="Graph ML platform CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a version")
    p_train.add_argument("--name", default="v1")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--nodes", type=int, default=240)
    p_train.add_argument("--out", default=None)
    p_train.set_defaults(func=cmd_train)

    p_node = sub.add_parser("predict-node", help="classify one node")
    p_node.add_argument("--node", type=int, required=True)
    p_node.add_argument("--seed", type=int, default=42)
    p_node.add_argument("--request-id", default="cli")
    p_node.set_defaults(func=cmd_predict_node)

    p_link = sub.add_parser("predict-link", help="score one node pair")
    p_link.add_argument("--u", type=int, required=True)
    p_link.add_argument("--v", type=int, required=True)
    p_link.add_argument("--seed", type=int, default=42)
    p_link.add_argument("--request-id", default="cli")
    p_link.set_defaults(func=cmd_predict_link)

    p_emb = sub.add_parser("embed", help="print one node's embedding")
    p_emb.add_argument("--node", type=int, required=True)
    p_emb.add_argument("--seed", type=int, default=42)
    p_emb.set_defaults(func=cmd_embed)

    p_drift = sub.add_parser("drift", help="score drift of a shifted graph")
    p_drift.add_argument("--seed", type=int, default=42)
    p_drift.add_argument("--p-in", type=float, default=0.03)
    p_drift.add_argument("--p-out", type=float, default=0.05)
    p_drift.set_defaults(func=cmd_drift)

    p_serve = sub.add_parser("serve", help="start the FastAPI service")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--seed", type=int, default=42)
    p_serve.set_defaults(func=cmd_serve)

    p_demo = sub.add_parser("demo", help="run the end-to-end demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
