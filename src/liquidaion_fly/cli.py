"""Command line entry point for data preparation and real experiments."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw", type=Path, default=Path("data/raw"))
    prep.add_argument("--output", type=Path, default=Path("data/cx512"))
    prep.add_argument("--neurons", type=int, default=512)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/cpu.json"))
    run.add_argument("--graph", type=Path, default=Path("data/cx512"))
    run.add_argument("--output", type=Path, default=Path("results/cpu"))
    run.add_argument("--model-path", help="Use downloaded local pretrained weights")
    sub.add_parser("inspect")
    view = sub.add_parser("visualize")
    view.add_argument("--results", type=Path, default=Path("results/cpu"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parser.parse_args()
    if args.command == "prepare":
        from .connectome import prepare
        print(json.dumps(prepare(args.raw, args.output, args.neurons), indent=2))
    elif args.command == "inspect":
        from .environment import inspect_environment
        print(json.dumps(inspect_environment(), indent=2))
    elif args.command == "visualize":
        from .visualization import render
        render(args.results)
    else:
        from .experiments import run_experiments
        config = json.loads(args.config.read_text())
        if args.model_path:
            config["model_source"] = config["model"]
            config["model"] = args.model_path
        summary = run_experiments(config, args.graph, args.output)
        print(json.dumps({"status": summary["status"], "total_seconds": summary["total_seconds"]}))


if __name__ == "__main__":
    main()
