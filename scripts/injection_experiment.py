"""Identical external injection with and without the external neural cue (real model).

Runs the same sampled stream three times with one scheduled injection: cue off, cue on the
16 DA-annotated neurons, and cue on a size-matched random control group. Before the injection
the three runs are identical by construction (same seed, same random numbers, same tokens).

Measured:
- reservoir state difference from the cue-off run at every stream position;
- DA-group activity in four phases: before, during, post-injection, after leaving the window;
- next-token effect on identical text: each run's reservoir state at (a) the end of the injection
  and (b) the moment the injection left the window is transplanted into the cue-off run's own
  window and scored on the same 16 following tokens; KL relative to the cue-off state.

  PYTHONPATH=src python scripts/injection_experiment.py --output results/injection/da-cue
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from liquidaion_fly.continuous import ContinuousConfig, ContinuousRun, load_run
from liquidaion_fly.llm import compare_logits
from liquidaion_fly.observatory import engine_config
from liquidaion_fly.observatory_engine import ObservatoryEngine
from liquidaion_fly.reservoir import Reservoir

TEXT = " Из-за деревьев неожиданно появился огромный зелёный мамонт."


def phase_means(rows, start, end, left):
    phases = {"before": [r for r in rows if start - 32 <= r["position"] < start],
              "during": [r for r in rows if start <= r["position"] < end],
              "post_injection": [r for r in rows if r["category"] == "post_injection"],
              "after_left_window": [r for r in rows if left is not None and left <= r["position"] < left + 32]}
    return {name: {"tokens": len(part),
                   "state_rms": float(np.mean([r["state_rms"] for r in part])) if part else None,
                   "da_mean": float(np.mean([r["da_mean"] for r in part])) if part else None,
                   "da_abs_mean": float(np.mean([r["da_abs_mean"] for r in part])) if part else None}
            for name, part in phases.items()}


def transplant(run: ContinuousRun, state, context, sources, forced):
    """Score `forced` stream tokens after `context` with a clone carrying `state` (cue rule as in `run`)."""
    reservoir = Reservoir(run.graph, features=run.engine.config.features, seed=run.engine.config.seed)
    reservoir.set_leak(run.config.leak_rate)
    reservoir.state = np.asarray(state, dtype=float).copy()
    coupling = run._coupling(reservoir)
    window, sources = list(context), list(sources)
    rows = []
    for token, source in forced:
        coupling.cue = run.cue_vector if (run.cue_vector is not None and sources[-1] is not None) else None
        rows.append(run._forward_logits(window, coupling).numpy())
        coupling.cue = None
        coupling.clear_trace()
        window = (window + [token])[-run.config.window:]
        sources = (sources + [source])[-run.config.window:]
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position", type=int, default=250)
    parser.add_argument("--tokens", type=int, default=600)
    parser.add_argument("--cue-strength", type=float, default=0.5)
    parser.add_argument("--text", default=TEXT)
    args = parser.parse_args()
    settings = json.loads(Path("configs/observatory.json").read_text())
    engine = ObservatoryEngine(settings["model"], Path("data/cx512"), device=settings["device"],
                               config=engine_config(settings))
    base = ContinuousConfig(window=128, total_tokens=args.tokens, decoding="sample", temperature=1.0, top_p=0.95,
                            sampling_seed=42, eos_policy="suppress", checkpoints=(), cue_strength=args.cue_strength,
                            injections=((args.position, args.text),))
    runs = {}
    for name, target in (("cue_off", "off"), ("cue_da", "da"), ("cue_random", "random")):
        run = ContinuousRun(engine, replace(base, cue_target=target), args.output / name)
        run.run()
        runs[name] = run
    off = runs["cue_off"]
    saved = {name: load_run(args.output / name) for name in runs}
    history = [(t, None) for t in off.seed_ids] + off.stream
    record = off.injections[1]
    end, left = record["end_position"], record["left_window_position"]
    result = {"text": args.text, "position": args.position, "cue_strength": args.cue_strength,
              "injected_tokens": record["inserted"], "end_position": end, "left_window_position": left,
              "conditions": {}}
    for name, run in runs.items():
        rows = saved[name]["telemetry"]
        divergence = next((i for i, (a, b) in enumerate(zip(off.stream, run.stream)) if a != b), None)
        entry = {"cue_neurons": len(run.cue_indices), "cue_applied_positions": [r["position"] for r in rows if r["cue_applied"]],
                 "first_token_divergence_from_cue_off": divergence,
                 "phases": phase_means(rows, args.position, end, run.injections[1]["left_window_position"]),
                 "left_window_position": run.injections[1]["left_window_position"]}
        events = saved[name]["states"]["events"]
        off_events = saved["cue_off"]["states"]["events"]
        for moment, key in (("end_of_injection", "injection1_end"), ("left_window", "injection1_left_window")):
            if key not in events or key not in off_events:
                continue
            at = end if moment == "end_of_injection" else left
            prefix = history[:len(off.seed_ids) + at]
            context = [t for t, _ in prefix][-128:]
            sources = [s for _, s in prefix][-128:]
            forced = off.stream[at:at + 16]
            reference = transplant(off, off_events[key], context, sources, forced)
            condition = transplant(run, events[key], context, sources, forced)
            entry[moment] = {"state_l2_difference_from_cue_off": float(np.linalg.norm(events[key] - off_events[key])),
                             "da_group_l2_difference": float(np.linalg.norm(events[key][off.da_indices] -
                                                                             off_events[key][off.da_indices])),
                             "logits_vs_cue_off_state_same_text": compare_logits(reference, condition)}
        result["conditions"][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "cue_applied_positions"}, ensure_ascii=False)[:900])
    (args.output / "injection_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.edgecolor": "#e4e3df", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    lo, hi = args.position - 60, (left or end) + 80
    for color, name in zip(("#2a78d6", "#eb6834", "#1baf7a"), runs):
        rows = [r for r in saved[name]["telemetry"] if lo <= r["position"] < hi]
        axes[0].plot([r["position"] for r in rows], [r["da_mean"] for r in rows], color=color, lw=1.4, label=name)
        axes[1].plot([r["position"] for r in rows], [r["state_rms"] for r in rows], color=color, lw=1.2, label=name)
    for ax in axes:
        ax.axvspan(args.position, end, color="#fde8de", lw=0)
        if left is not None:
            ax.axvline(left, color="#52514e", lw=0.8, ls="--")
        ax.grid(color="#e4e3df", lw=0.5)
        ax.legend(frameon=False)
    axes[0].set_title("Mean signed state of the 16 DA-annotated neurons (shaded: injected tokens; dashed: injection "
                      "left the 128-token window)", loc="left")
    axes[0].axhline(0, color="#52514e", lw=0.6)
    axes[1].set_title("Reservoir state RMS", loc="left")
    axes[1].set_xlabel("stream position (tokens after the seed)")
    fig.tight_layout()
    fig.savefig(args.output / "injection_da_cue.png", dpi=140)


if __name__ == "__main__":
    main()
