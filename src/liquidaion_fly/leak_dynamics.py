"""Leak-rate dynamics: repeated steering probes at several leak rates, plus idle retention.

Used by plugins/examples/leak_rate_dynamics.py through the plugin API only. Nothing here
changes the reservoir equation, the coupling or decoding; it sets the leak rate the GUI
also sets, runs read-only probes (steering_diagnostics.run_probe) and advances idle steps.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .continuous import repetition_metrics
from .steering_diagnostics import UNDEFINED, ZERO_NORM, cosine, spectrum

SUMMARY_KEYS = ("state_rms_after", "state_change_norm", "state_cos_prev", "steering_norm", "applied_relative_norm",
                "steering_cos_prev", "steering_cos_ref", "steering_orth_ratio", "kl_clean_fly",
                "kl_off_fly_immediate", "kl_clean_off_history")


def _median(rows: list[dict], key: str) -> float:
    values = np.array([r.get(key, math.nan) for r in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def _last(rows: list[dict], key: str) -> float:
    values = [r.get(key, math.nan) for r in rows]
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(values[-1]) if values else math.nan


def idle_retention(api: Any, steps: int) -> list[dict[str, Any]]:
    """After the measurement: idle reservoir updates with no input (NOT tokens).

    Records RMS, norm ratio and cosine to the state at the end of the measurement period."""
    start = api.get_reservoir_state()
    reference, t0 = np.array(start["state"]), start["simulation_time"]
    rows = []
    for step in range(1, steps + 1):
        state = api.advance_idle(1)
        x = np.array(state["state"])
        rows.append({"idle_step": step, "simulation_time_since_stop": state["simulation_time"] - t0,
                     "state_rms": state["rms"],
                     "norm_ratio": float(np.linalg.norm(x) / np.linalg.norm(reference))
                     if np.linalg.norm(reference) > ZERO_NORM else math.nan,
                     "cos_to_stop_state": cosine(x, reference)})
    return rows


def half_life(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if math.isfinite(row["norm_ratio"]) and row["norm_ratio"] <= 0.5:
            return {"idle_steps": row["idle_step"], "simulation_time": row["simulation_time_since_stop"]}
    return {"idle_steps": None, "simulation_time": None,
            "note": f"norm did not fall to half within {len(rows)} idle steps" if rows else "no idle steps"}


def matched_amplitude(runs: dict[float, dict[str, Any]], forced: bool) -> dict[str, Any]:
    """OFFLINE geometry of unit-normalized recorded steering vectors (not a causal experiment).

    Rescaling recorded vectors cannot reproduce a coupled run at matched amplitude: the applied
    delta also drives feedback and therefore the future reservoir states and tokens."""
    units, result = {}, {"note": "Offline analysis of recorded vectors scaled to unit norm. Directions are "
                                 "unchanged by scaling, so these metrics are amplitude-independent; they are NOT "
                                 "the outcome of a coupled run at matched amplitude, which would need a separate "
                                 "controlled intervention mechanism in the coupling (not implemented).",
                         "conditions": {}, "cross_condition_same_step_cosine": {}}
    for leak, run in runs.items():
        vectors = run["vectors"].get("steering")
        if vectors is None or not len(vectors):
            continue
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > ZERO_NORM
        unit = np.where(valid[:, None], vectors / np.where(valid, norms, 1)[:, None], np.nan)
        units[leak] = unit
        geometry = spectrum(unit[valid]) if valid.sum() else {}
        result["conditions"][str(leak)] = {"valid_steps": int(valid.sum()), "unit_vector_geometry": geometry,
                                           "median_norm": float(np.median(norms[valid])) if valid.any() else None}
    leaks = sorted(units)
    for i, a in enumerate(leaks):
        for b in leaks[i + 1:]:
            n = min(len(units[a]), len(units[b]))
            values = [cosine(units[a][k], units[b][k]) for k in range(n)
                      if np.isfinite(units[a][k]).all() and np.isfinite(units[b][k]).all()]
            result["cross_condition_same_step_cosine"][f"{a}|{b}"] = {
                "median": float(np.median(values)) if values else None, "steps": len(values),
                "meaningful": forced,
                "note": "same token ids at each step" if forced else
                        "free mode: different token histories, step-aligned comparison is descriptive only"}
    return result


def run_leak_dynamics(api: Any, leak_rates: list[float], prompt: str, max_tokens: int, mode: str = "forced",
                      forced_tokens: list[int] | None = None, warmup_prompt: str | None = None,
                      warmup_tokens: int = 0, idle_steps: int = 0, reset_between_runs: bool = True,
                      counterfactual: bool = True, save_vectors: bool = True, save_plots: bool = True) -> dict:
    if mode not in {"forced", "free"}:
        raise ValueError("mode must be 'forced' or 'free'")
    output = api.output_dir
    runs: dict[float, dict[str, Any]] = {}
    token_rows, idle_rows, summary_rows = [], [], []
    tokens, warm_ids = forced_tokens, None
    for leak in leak_rates:
        api.check_cancelled()
        if reset_between_runs or not runs:
            api.new_session()  # zero reservoir, empty histories and caches: independent runs
        api.set_coupling(leak_rate=float(leak))
        directory = output / f"leak-{leak:.4f}"
        warm_rows = []
        if warmup_tokens > 0:
            warm = api.steering_probe(warmup_prompt or prompt, warmup_tokens, name="warmup", forced_tokens=warm_ids,
                                      counterfactual=False, save_vectors=False, save_plots=save_plots,
                                      directory=directory)
            warm_ids = warm["metadata"]["token_ids"]  # the same warm-up sequence for every leak rate
            warm_rows = warm["rows"]
        result = api.steering_probe(prompt, max_tokens, name="measure", forced_tokens=tokens if mode == "forced" else None,
                                    counterfactual=counterfactual, fly_tokens=mode, save_vectors=False,
                                    save_plots=save_plots, directory=directory)
        if mode == "forced":
            tokens = result["metadata"]["token_ids"]  # the same measurement sequence for every leak rate
        retention = idle_retention(api, idle_steps) if idle_steps > 0 else []
        runs[leak] = result
        for phase, rows in (("warmup", warm_rows), ("measure", result["rows"])):
            for row in rows:
                index = row["step"] - len(warm_rows) if phase == "warmup" else row["step"]
                token_rows.append({"leak": leak, "phase": phase, "index": index,
                                   "reservoir_updates_per_token": result["metadata"]["reservoir"]["updates_per_token"],
                                   **{k: v for k, v in row.items() if k != "token_text"}})
        idle_rows += [{"leak": leak, **row} for row in retention]
        rows = result["rows"]
        repetition = repetition_metrics(result["metadata"]["token_ids"], chunk=max(1, len(rows)))
        summary_rows.append({
            "leak": leak, "mode": mode, "tokens": len(rows), "warmup_tokens": len(warm_rows),
            "reservoir_updates_per_token": result["metadata"]["reservoir"]["updates_per_token"],
            "configured_strength": result["metadata"]["settings"]["strength"],
            **{f"median_{key}": _median(rows, key) for key in SUMMARY_KEYS},
            "last_steering_cos_ref": _last(rows, "steering_cos_ref"),
            "normalization_active_steps": sum(bool(r.get("normalization_active")) for r in rows),
            "top1_changed": sum(bool(r.get("top1_changed")) for r in rows),
            "top1_changed_immediate": sum(bool(r.get("top1_changed_immediate")) for r in rows),
            # Descriptive: for small perturbations KL grows ~ quadratically with the perturbation size.
            "median_kl_immediate_per_rel2": _median([{"x": r["kl_off_fly_immediate"] / r["applied_relative_norm"] ** 2}
                                                     for r in rows if r.get("applied_relative_norm", 0) > 1e-9
                                                     and "kl_off_fly_immediate" in r], "x"),
            "idle_half_life_steps": half_life(retention)["idle_steps"] if retention else None,
            "idle_half_life_simulation_time": half_life(retention)["simulation_time"] if retention else None,
            "distinct_1": repetition["distinct_1"], "distinct_2": repetition["distinct_2"],
            # Largest fraction of tokens equal to the token p steps earlier (exact periodic repetition).
            "self_repeat_rate": repetition["chunks"][0]["self_repeat_rate"] if repetition["chunks"] else None,
            "best_period": repetition["chunks"][0]["best_period"] if repetition["chunks"] else None,
            "fly_text": result["metadata"]["fly_text"][:400]})
        api.log(f"leak {leak:.4f}: relative intervention {summary_rows[-1]['median_applied_relative_norm']:.3g}, "
                f"steering cos prev {summary_rows[-1]['median_steering_cos_prev']:.4f}, "
                f"immediate KL {summary_rows[-1]['median_kl_off_fly_immediate']:.3g}")
    matched = matched_amplitude(runs, mode == "forced")
    first = next(iter(runs.values()))["metadata"] if runs else {}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "leak_rates": list(leak_rates),
                "prompt": prompt, "max_tokens": max_tokens, "warmup_prompt": warmup_prompt or prompt,
                "warmup_tokens": warmup_tokens, "warmup_token_ids": warm_ids, "measurement_token_ids": tokens,
                "idle_steps": idle_steps, "reset_between_runs": reset_between_runs, "counterfactual": counterfactual,
                "model": first.get("model"), "coupling": first.get("coupling"), "settings": first.get("settings"),
                "reservoir": {key: value for key, value in first.get("reservoir", {}).items()
                              if key not in {"leak", "initial_state_rms"}},
                "projection": first.get("projection"), "step_alignment": first.get("step_alignment"),
                "state_alignment": first.get("state_alignment"),
                "phases": {"warmup": "fixed forced token sequence (same ids for every leak), index < 0, not "
                                     "counted as measurement",
                           "measure": "index >= 0; forced = same ids for every leak, free = Fly's own greedy tokens",
                           "idle": "reservoir updates with no input after the measurement; each idle step is "
                                   "dt simulation time, not a token"},
                "independence": "new_session() before every leak rate" if reset_between_runs else
                                "runs continue from the previous run's reservoir state (NOT independent)",
                "free_mode_caveat": "free mode: Clean and Fly histories differ; only Fly-off vs Fly (same Fly "
                                    "history) is a direct current-token effect" if mode == "free" else None,
                "configured_vs_actual": "configured steering strength is not the applied size: with RMS(centered) "
                                        "< 1 the normalization is inactive and the applied delta scales with the "
                                        "reservoir state; see applied_relative_norm",
                "undefined_values": UNDEFINED, "runs": {str(k): v["output"] for k, v in runs.items()}}
    _write(output, metadata, token_rows, summary_rows, idle_rows, matched, runs, save_vectors, save_plots)
    return {"output": str(output), "summary": summary_rows, "matched_amplitude": matched}


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()})


def _write(output: Path, metadata: dict, token_rows: list, summary_rows: list, idle_rows: list, matched: dict,
           runs: dict, save_vectors: bool, save_plots: bool) -> None:
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=float) + "\n",
                                          encoding="utf-8")
    _csv(output / "token_metrics.csv", token_rows)
    _csv(output / "summary.csv", summary_rows)
    _csv(output / "idle_metrics.csv", idle_rows)
    (output / "matched_amplitude.json").write_text(json.dumps(matched, indent=2, default=float) + "\n")
    if save_vectors:
        states, steering = {}, {}
        for leak, run in runs.items():
            key = f"leak_{leak:.4f}".replace(".", "_")
            for name in ("state_before", "state_used", "state_after"):
                if name in run["vectors"]:
                    states[f"{key}__{name}"] = run["vectors"][name]
            for name in ("compact", "steering", "applied_delta"):
                if name in run["vectors"]:
                    steering[f"{key}__{name}"] = run["vectors"][name]
        np.savez_compressed(output / "reservoir_states.npz", **states)
        np.savez_compressed(output / "steering_vectors.npz", **steering)
    if save_plots:
        plot_comparison(output / "plots", token_rows, idle_rows, metadata)


def plot_comparison(directory: Path, token_rows: list, idle_rows: list, metadata: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    leaks = metadata["leak_rates"]
    coupling = metadata.get("coupling") or {}
    title = (f"{(metadata.get('model') or {}).get('display', '')} · {coupling.get('coupling_mode', '')} "
             f"{coupling.get('active_bypasses', '')} · {metadata['mode']}")
    panels = {"01_reservoir_rms.png": ("state_rms_after", "reservoir state RMS", False),
              "02_relative_intervention.png": ("applied_relative_norm", "applied ‖Δ‖ / ‖h‖", False),
              "03_steering_cos_prev.png": ("steering_cos_prev", "cos(v_t, v_t−1)", False),
              "04_steering_cos_ref.png": ("steering_cos_ref", "cos(v_t, v_ref)", False),
              "05_state_change.png": ("state_change_norm", "‖x_after − x_before‖ per token", False),
              "06_kl_clean_fly.png": ("kl_clean_fly", "KL(Clean‖Fly), same tokens", True),
              "07_kl_immediate.png": ("kl_off_fly_immediate", "KL(Fly-off‖Fly), current-token steering", True)}
    for name, (key, label, log) in panels.items():
        fig, axis = plt.subplots(figsize=(8, 3.6))
        drawn = False
        for leak in leaks:
            rows = [r for r in token_rows if r["leak"] == leak]
            x = np.array([r["index"] for r in rows])
            y = np.array([r.get(key, math.nan) for r in rows], dtype=float)
            if np.isfinite(y).any():
                axis.plot(x, y, linewidth=1, marker=".", markersize=3, label=f"leak {leak:g}")
                drawn = True
        if metadata.get("warmup_tokens"):
            axis.axvline(0, color="gray", linestyle="--", linewidth=1)
            axis.text(0, 1.0, " measurement →", transform=axis.get_xaxis_transform(), fontsize=7, color="gray",
                      va="top")
        if log and drawn:
            axis.set_yscale("symlog", linthresh=1e-5)
        axis.set_xlabel("token index (forward pass; < 0 = warm-up, 0 = prompt prefill of the measurement)")
        axis.set_ylabel(label)
        axis.set_title(title, fontsize=8)
        if drawn:
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, "not measured in this mode", ha="center", transform=axis.transAxes)
        fig.tight_layout()
        fig.savefig(directory / name, dpi=110)
        plt.close(fig)
    if idle_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for leak in leaks:
            rows = [r for r in idle_rows if r["leak"] == leak]
            axes[0].plot([r["idle_step"] for r in rows], [r["norm_ratio"] for r in rows], label=f"leak {leak:g}")
            axes[1].plot([r["idle_step"] for r in rows], [r["cos_to_stop_state"] for r in rows], label=f"leak {leak:g}")
        axes[0].set_ylabel("‖x‖ / ‖x_stop‖")
        axes[1].set_ylabel("cos(x, x_stop)")
        for axis in axes:
            axis.set_xlabel("idle reservoir step after the measurement (no input; not tokens)")
            axis.legend(fontsize=7)
        fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / "08_idle_retention.png", dpi=110)
        plt.close(fig)
