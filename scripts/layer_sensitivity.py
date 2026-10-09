"""Short steering-layer sensitivity check on identical token contexts (no training, fixed projections).

For each steering layer, the same contexts and 16 forced continuation tokens are scored with the
reservoir started from the same recorded state, and compared with Clean. The intervention size
is `strength x RMS(hidden)` at the steered layer, so the relative perturbation is matched across
layers by construction.

  PYTHONPATH=src python scripts/layer_sensitivity.py --run results/continuous/sample-t1.0-p0.95-seed42/fly \
      --output results/continuous/layer-sensitivity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from liquidaion_fly.continuous import load_run
from liquidaion_fly.llm import compare_logits
from liquidaion_fly.observatory import engine_config
from liquidaion_fly.observatory_engine import MeasuredCoupling, ObservatoryEngine
from liquidaion_fly.reservoir import Reservoir


def score(engine, window, forced, state, steer_layer, strength, feedback):
    coupling = None
    if steer_layer is not None:
        reservoir = Reservoir(engine.graph, features=engine.config.features, seed=engine.config.seed)
        reservoir.state = state.copy()
        coupling = MeasuredCoupling(reservoir, engine.llm.model.config.hidden_size, engine.config.capture_layer,
                                    steer_layer, strength, feedback, mode="fly", seed=engine.config.seed)
    rows, deltas = [], []
    for token in forced:
        ids = torch.tensor([window], device=engine.llm.device)
        if coupling:
            coupling.attach(engine.llm.model)
        try:
            with torch.inference_mode():
                rows.append(engine.llm.model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().numpy())
        finally:
            if coupling:
                coupling.detach()
        if coupling:
            deltas.append(coupling.records[-1]["delta_relative_rms"])
            coupling.clear_trace()
        window = (window + [token])[-len(window):]
    return np.stack(rows), float(np.mean(deltas)) if deltas else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="A saved Fly condition directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="8,12,14,18,22,26")
    parser.add_argument("--steps", default="512,1024,2048,4096", help="Contexts and recorded states taken at these steps")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--feedback", type=float, default=0.25)
    parser.add_argument("--forced", type=int, default=16)
    args = parser.parse_args()
    settings = json.loads(Path("configs/observatory.json").read_text())
    engine = ObservatoryEngine(settings["model"], Path("data/cx512"), device=settings["device"],
                               config=engine_config(settings))
    run = load_run(args.run)
    window_size = run["metadata"]["config"]["window"]
    history = run["tokens"]["seed"] + run["tokens"]["generated"]
    seed_len = len(run["tokens"]["seed"])
    states = dict(zip(run["states"]["steps"], run["states"]["states"]))
    rows = []
    for step in (int(x) for x in args.steps.split(",")):
        if step not in states or seed_len + step + args.forced > len(history):
            continue
        window = history[:seed_len + step][-window_size:]
        forced = history[seed_len + step:seed_len + step + args.forced]
        clean, _ = score(engine, list(window), forced, states[step], None, 0, 0)
        for layer in (int(x) for x in args.layers.split(",")):
            fly, delta = score(engine, list(window), forced, states[step], layer, args.strength, args.feedback)
            metrics = compare_logits(clean, fly)
            rows.append({"step": step, "steer_layer": layer, "mean_delta_relative_rms": delta, **metrics})
            print(json.dumps(rows[-1]))
    args.output.mkdir(parents=True, exist_ok=True)
    layers = sorted({row["steer_layer"] for row in rows})
    summary = [{"steer_layer": layer,
                "mean_kl_clean_to_fly": float(np.mean([r["mean_kl_baseline_to_condition"] for r in rows if r["steer_layer"] == layer])),
                "mean_argmax_disagreement": float(np.mean([r["argmax_disagreement_fraction"] for r in rows if r["steer_layer"] == layer])),
                "mean_delta_relative_rms": float(np.mean([r["mean_delta_relative_rms"] for r in rows if r["steer_layer"] == layer]))}
               for layer in layers]
    (args.output / "layer_sensitivity.json").write_text(json.dumps(
        {"run": str(args.run), "strength": args.strength, "feedback": args.feedback, "forced_tokens": args.forced,
         "capture_layer": engine.config.capture_layer, "rows": rows, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
