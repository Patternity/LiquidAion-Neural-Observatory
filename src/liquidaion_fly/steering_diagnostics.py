"""Read-only per-token diagnostics of the reservoir -> steering pathway (measurement, not a mechanism).

The steering rule being observed (llm.Coupling._intervene), for reservoir state x (512):

    compact   = x @ readout                (64;   readout 512 x 64, fixed, seed 42)
    raw       = compact @ decoder          (hidden; decoder 64 x hidden, fixed, seed 42)
    centered  = raw - mean(raw)            (removes the uniform component)
    direction = centered / max(RMS(centered), 1)
    delta     = strength * RMS(h_last) * direction     (added to the last position only)

"Steering vector" v_t below is ``centered``: every later step only rescales it, so its direction
is the direction of the applied delta (up to the model dtype rounding, measured separately).

A probe run (``run_probe``) uses the engine's own forward path (ObservatoryEngine._forward)
and the engine's live coupling and reservoir:

    1. Clean: the prompt plus a token sequence, uncoupled. The sequence is Clean's own greedy
       continuation unless ``forced_tokens`` is given.
    2. Fly: the SAME token ids, coupled; the reservoir evolves legitimately through them.
    3. Immediate counterfactual (optional): at every step, the same input token is also run
       uncoupled on a copy of the Fly KV cache (Fly history, no steering at this token, no
       reservoir update). Fly-off vs Fly-on isolates the immediate effect of this token's
       steering; Clean vs Fly-off is the effect carried by earlier interventions in the cache.

Step k is forward pass k: pass 0 is the prompt prefill; pass k >= 1 feeds token k-1 of the
sequence. Pass k's logits predict token k; its hooks act on the last input position.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import torch

# Vectors with a norm at or below this are treated as zero: their cosine is undefined.
ZERO_NORM = 1e-12
UNDEFINED = "Cosines involving a vector with norm <= 1e-12 are undefined and stored as NaN (empty in CSV)."


def _cpu(tensor: Any) -> np.ndarray:
    """A float64 CPU copy; never a view of an engine-owned tensor, never part of a graph."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().to("cpu", torch.float64).numpy().reshape(-1).copy()
    return np.array(tensor, dtype=np.float64).reshape(-1)


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity, NaN when either vector is missing or has (near-)zero norm."""
    if a is None or b is None:
        return math.nan
    na, nb = norm(a), norm(b)
    if na <= ZERO_NORM or nb <= ZERO_NORM:
        return math.nan
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def decompose(vector: np.ndarray, reference: np.ndarray | None) -> tuple[float, float, float]:
    """Signed parallel component, orthogonal norm and orthogonal/total ratio relative to reference."""
    total = norm(vector)
    if reference is None or norm(reference) <= ZERO_NORM or total <= ZERO_NORM:
        return math.nan, math.nan, math.nan
    unit = reference / norm(reference)
    parallel = float(np.dot(vector, unit))
    orthogonal = norm(vector - parallel * unit)
    return parallel, orthogonal, orthogonal / total


def spectrum(vectors: np.ndarray, top: int = 10) -> dict[str, Any]:
    """Offline geometry of a trajectory (rows = token steps).

    centered: PCA of the rows after subtracting their mean -- how many directions the
      variation spans (explained-variance ratios; effective rank = exp(entropy of the ratios);
      participation ratio = (sum l)^2 / sum l^2).
    uncentered: share of total energy sum ||v_t||^2 along the top singular direction -- close
      to 1 means the vectors stay near ONE fixed direction (a constant bias). A trajectory can
      be low-dimensional (centered) without being fixed-direction (uncentered), and vice versa.
    """
    rows = np.asarray([v for v in vectors if norm(v) > ZERO_NORM], dtype=np.float64)
    result: dict[str, Any] = {"samples": int(len(rows)), "dimension": int(vectors.shape[1]) if len(vectors) else 0}
    if len(rows) < 3:
        result["note"] = "fewer than 3 nonzero samples; not computed"
        return result
    centered = rows - rows.mean(axis=0)
    eigen = np.linalg.svd(centered, compute_uv=False) ** 2
    if eigen.sum() > 0:
        ratio = eigen / eigen.sum()
        nonzero = ratio[ratio > 0]
        result["centered"] = {"explained_variance_ratio": ratio[:top].tolist(),
                              "effective_rank": float(np.exp(-(nonzero * np.log(nonzero)).sum())),
                              "participation_ratio": float(eigen.sum() ** 2 / (eigen ** 2).sum())}
    energy = np.linalg.svd(rows, compute_uv=False) ** 2
    result["uncentered"] = {"top1_energy_fraction": float(energy[0] / energy.sum()),
                            "top3_energy_fraction": float(energy[:3].sum() / energy.sum())}
    result["note"] = (f"rank estimates from {len(rows)} samples are bounded by min(samples - 1, dimension) "
                      "and are unreliable when samples are few")
    return result


class SteeringObserver:
    """Collects copies of the tensors the coupling hooks actually compute, grouped per forward pass."""

    def __init__(self) -> None:
        self.passes: list[dict[str, Any]] = []
        self.current: dict[str, Any] = self._empty()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"captures": {}, "steering": {}, "feedback": {}, "aggregate": None}

    def capture(self, layer: int, hidden: torch.Tensor, features: np.ndarray) -> None:
        h = _cpu(hidden)
        self.current["captures"][layer] = {"hidden_norm": norm(h), "features": np.array(features, dtype=np.float64)}

    def steering(self, layer: int | None, state: np.ndarray, compact: torch.Tensor, raw: torch.Tensor,
                 centered: torch.Tensor, direction: torch.Tensor, baseline: torch.Tensor,
                 modified: torch.Tensor) -> None:
        before, after = _cpu(baseline), _cpu(modified)
        self.current["steering"][layer] = {
            "state_used": np.array(state, dtype=np.float64), "compact": _cpu(compact), "raw": _cpu(raw),
            "centered": _cpu(centered), "direction": _cpu(direction), "hidden_before": before,
            "applied_delta": after - before}

    def feedback(self, layer: int | None, scaled_input: np.ndarray) -> None:
        self.current["feedback"][layer] = np.array(scaled_input, dtype=np.float64)

    def aggregate(self, captured: np.ndarray, feedback_input: np.ndarray) -> None:
        self.current["aggregate"] = {"captured": np.array(captured, dtype=np.float64),
                                     "feedback": np.array(feedback_input, dtype=np.float64)}

    def end_pass(self, state_before: np.ndarray, state_after: np.ndarray) -> None:
        self.current["state_before"] = np.array(state_before, dtype=np.float64)
        self.current["state_after"] = np.array(state_after, dtype=np.float64)
        self.passes.append(self.current)
        self.current = self._empty()


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def distribution_metrics(reference: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    """KL(reference || condition), Jensen-Shannon, top-1 identity, confidence and margin, logit L2."""
    lp, lq = _log_softmax(reference), _log_softmax(condition)
    p, q = np.exp(lp), np.exp(lq)
    m = 0.5 * (p + q)
    lm = np.log(np.maximum(m, 1e-300))
    top_p, top_q = np.argsort(p)[-2:], np.argsort(q)[-2:]
    return {"kl": float((p * (lp - lq)).sum()), "js": float(0.5 * (p * (lp - lm)).sum() + 0.5 * (q * (lq - lm)).sum()),
            "top1_reference": int(top_p[-1]), "top1_condition": int(top_q[-1]),
            "top1_changed": bool(top_p[-1] != top_q[-1]),
            "top1_prob_reference": float(p[top_p[-1]]), "top1_prob_condition": float(q[top_q[-1]]),
            "margin_reference": float(p[top_p[-1]] - p[top_p[-2]]),
            "margin_condition": float(q[top_q[-1]] - q[top_q[-2]]),
            "logit_diff_l2": float(np.linalg.norm(np.asarray(condition, np.float64) - np.asarray(reference, np.float64)))}


def _logits(result: Any) -> np.ndarray:
    return result.logits[0, -1].detach().to("cpu", torch.float64).numpy().copy()


def _readout_basis(readout: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the reservoir directions the readout can see (its column space)."""
    q, r = np.linalg.qr(readout)
    rank = int((np.abs(np.diag(r)) > 1e-10 * np.abs(np.diag(r)).max()).sum())
    return q[:, :rank]


def projection_report(coupling: Any) -> dict[str, Any]:
    readout = np.asarray(coupling.readout, dtype=np.float64)
    decoder = coupling.decoder.numpy().astype(np.float64)
    singular_readout = np.linalg.svd(readout, compute_uv=False)
    singular_decoder = np.linalg.svd(decoder, compute_uv=False)
    centering = decoder - decoder.mean(axis=1, keepdims=True)
    singular_centered = np.linalg.svd(centering, compute_uv=False)
    return {
        "readout_shape": list(readout.shape), "decoder_shape": list(decoder.shape),
        "readout_rank": int((singular_readout > 1e-10 * singular_readout.max()).sum()),
        "decoder_rank": int((singular_decoder > 1e-10 * singular_decoder.max()).sum()),
        "centered_decoder_rank": int((singular_centered > 1e-10 * singular_centered.max()).sum()),
        "readout_singular_values": {"max": float(singular_readout.max()), "min": float(singular_readout.min())},
        "decoder_singular_values": {"max": float(singular_decoder.max()), "min": float(singular_decoder.min())},
        "centered_decoder_condition_number": float(singular_centered.max() / singular_centered.min()),
        "reservoir_null_space_dimension": int(readout.shape[0] - np.linalg.matrix_rank(readout)),
        "fixed": "readout and decoder are drawn once from seed 42 and never change",
        "nonlinearity": "none between state and delta: linear projections, mean removal, then one scalar "
                        "per token (1 / max(RMS, 1) and strength * RMS(h)); direction is set by the linear part",
        "steering_subspace": "every steering vector lies in the image of the centered decoder "
                             "(at most 64 dimensions of the hidden space)"}


def _forced_loop(engine: Any, prompt_ids: torch.Tensor, tokens: list[int] | None, steps: int, coupling: Any,
                 observer: SteeringObserver | None, counterfactual: bool, cancel: Event) -> dict[str, Any]:
    """Prefill plus forced (or greedy, when tokens is None) decode through engine._forward."""
    cache, current = None, prompt_ids
    logits, off_logits, chosen = [], [], []
    reservoir = coupling.reservoir if coupling is not None else None
    for step in range(steps):
        if cancel.is_set():
            break
        if counterfactual and coupling is not None:
            # Same Fly history (copied cache), this token uncoupled: no steering, no reservoir update.
            off_logits.append(_logits(engine._forward(current, copy.deepcopy(cache), None)))
        before = reservoir.state.copy() if reservoir is not None else None
        result = engine._forward(current, cache, coupling)
        if observer is not None:
            observer.end_pass(before, reservoir.state.copy())
        logits.append(_logits(result))
        cache = result.past_key_values
        token = int(np.argmax(logits[-1])) if tokens is None else int(tokens[step])
        chosen.append(token)
        current = torch.tensor([[token]], device=engine.llm.device)
    return {"logits": logits, "off_logits": off_logits, "tokens": chosen}


def run_probe(engine: Any, prompt: str, max_tokens: int = 48, output: Path | None = None,
              forced_tokens: list[int] | None = None, counterfactual: bool = True,
              cancel: Event | None = None, label: str = "probe") -> dict[str, Any]:
    """One controlled Clean / Fly / Fly-off run on the engine's live coupling and reservoir.

    The reservoir starts from its current state (call new_session() or reset_state() first for a
    zero start). Its final state is the state after the Fly pass, as in any Fly generation.
    """
    cancel = cancel or Event()
    from .observatory_engine import ChatSession
    with engine._lock:
        plan = engine._build_context(ChatSession([]), prompt, max_tokens)
        ids = plan.ids.to(engine.llm.device)
        clean = _forced_loop(engine, ids, forced_tokens, max_tokens, None, None, False, cancel)
        tokens = clean["tokens"]
        coupling, observer = engine.coupling, SteeringObserver()
        coupling.clear_trace()
        coupling.observer = observer
        initial_state = engine.reservoir.state.copy()
        try:
            fly = _forced_loop(engine, ids, tokens, len(tokens), coupling, observer, counterfactual, cancel)
        finally:
            coupling.observer = None
        info = engine.coupling_info()
        model = {key: value for key, value in engine.model_info().items() if not key.startswith("gpu_")}
        metadata = {"label": label, "created_utc": datetime.now(timezone.utc).isoformat(), "prompt": prompt,
                    "prompt_token_ids": plan.ids[0].tolist(), "token_ids": tokens,
                    "token_source": "forced (given)" if forced_tokens is not None else "Clean greedy continuation",
                    "steps": len(fly["logits"]), "cancelled": cancel.is_set(), "model": model, "coupling": info,
                    "settings": {key: getattr(engine.config, key) for key in
                                 ("strength", "feedback", "steering_enabled", "feedback_enabled", "leak_rate")},
                    "reservoir": {"leak": engine.reservoir.leak, "dt": engine.reservoir.dynamics.dt,
                                  "gain": engine.reservoir.dynamics.gain,
                                  "input_strength": engine.reservoir.dynamics.input_strength,
                                  "initial_state_rms": float(np.sqrt(np.mean(initial_state ** 2))),
                                  "updates_per_token": info["reservoir_updates_per_token"],
                                  "note": "Legacy single bypass advances the reservoir twice per token (capture, "
                                          "feedback) and synchronous multi-bypass once, so equal leak rates are "
                                          "not equal time scales per token across the two modes."},
                    "projection": projection_report(coupling), "counterfactual": counterfactual,
                    "step_alignment": "step k = forward pass k; pass 0 is the prompt prefill, pass k >= 1 feeds "
                                      "token k-1; pass k's logits predict token k; hooks act on the last input position",
                    "state_alignment": {"state_before": "reservoir state at the start of pass k",
                                        "state_used_for_steering": "the state passed to the steering rule in pass k "
                                        "(legacy forward: after this token's capture update; legacy backward and "
                                        "synchronous: state_before)",
                                        "state_after": "reservoir state at the end of pass k (input to pass k+1)"},
                    "steering_vector": "centered = compact @ decoder - mean; the applied delta is a positive "
                                       "multiple of it (strength * RMS(h) / max(RMS(centered), 1))",
                    "undefined_values": UNDEFINED}
        analysis = analyse(observer.passes, clean["logits"], fly["logits"], fly["off_logits"], coupling)
        metadata["reference_steps"] = analysis["reference_steps"]
        metadata["geometry"] = analysis["geometry"]
        metadata["tokens_text"] = [engine.llm.tokenizer.decode([t]) for t in tokens]
        if output is not None:
            save(Path(output), metadata, analysis)
        return {"metadata": metadata, "rows": analysis["rows"], "summary": analysis["summary"],
                "output": str(output) if output is not None else None}


def analyse(passes: list[dict[str, Any]], clean_logits: list, fly_logits: list, off_logits: list,
            coupling: Any) -> dict[str, Any]:
    basis = _readout_basis(np.asarray(coupling.readout, dtype=np.float64))
    rows, vectors = [], {"state_before": [], "state_after": [], "state_used": [], "compact": [], "steering": [],
                         "applied_delta": [], "input": []}
    references: dict[str, np.ndarray | None] = {"state": None, "compact": None, "steering": None}
    reference_steps: dict[str, int | None] = {"state": None, "compact": None, "steering": None}
    previous: dict[str, np.ndarray | None] = {"state": None, "compact": None, "steering": None, "input": None}

    def tracked(key: str, vector: np.ndarray, step: int, row: dict[str, Any], prefix: str) -> None:
        if references[key] is None and norm(vector) > ZERO_NORM:
            references[key], reference_steps[key] = vector.copy(), step
        row[f"{prefix}_cos_prev"] = cosine(vector, previous[key])
        row[f"{prefix}_cos_ref"] = cosine(vector, references[key])
        parallel, orthogonal, ratio = decompose(vector, references[key])
        row[f"{prefix}_parallel"], row[f"{prefix}_orthogonal"], row[f"{prefix}_orth_ratio"] = parallel, orthogonal, ratio
        before = previous[key]
        row[f"{prefix}_rel_norm_change"] = (norm(vector) - norm(before)) / norm(before) \
            if before is not None and norm(before) > ZERO_NORM else math.nan
        previous[key] = vector

    for step, data in enumerate(passes):
        row: dict[str, Any] = {"step": step}
        before, after = data["state_before"], data["state_after"]
        steering = data["steering"]
        first = steering[min(steering, key=lambda k: -1 if k is None else k)] if steering else None
        used = first["state_used"] if first else before
        change = after - before
        row.update({"state_rms_before": float(np.sqrt(np.mean(before ** 2))),
                    "state_rms_after": float(np.sqrt(np.mean(after ** 2))), "state_norm_after": norm(after),
                    "state_change_norm": norm(change),
                    "state_readout_fraction": norm(basis.T @ after) / norm(after) if norm(after) > ZERO_NORM else math.nan,
                    "change_readout_fraction": norm(basis.T @ change) / norm(change) if norm(change) > ZERO_NORM else math.nan})
        tracked("state", after, step, row, "state")
        vectors["state_before"].append(before)
        vectors["state_after"].append(after)
        vectors["state_used"].append(used)
        captures = data["captures"]
        row["capture_layers"] = ";".join(str(k) for k in sorted(captures))
        row["capture_hidden_norm"] = float(np.mean([c["hidden_norm"] for c in captures.values()])) if captures else math.nan
        row["capture_feature_norm"] = float(np.mean([norm(c["features"]) for c in captures.values()])) if captures else math.nan
        aggregate = data["aggregate"]
        if aggregate is not None:
            # Synchronous: the one aggregate input u of this token's single update.
            reservoir_input = aggregate["captured"] + aggregate["feedback"]
        else:
            # Legacy: capture and feedback enter in two separate updates; their sum is shown.
            parts = [c["features"] for c in captures.values()] + list(data["feedback"].values())
            reservoir_input = np.sum(parts, axis=0) if parts else np.zeros(1)
        row["input_feature_norm"] = norm(reservoir_input)
        row["input_cos_prev"] = cosine(reservoir_input, previous["input"])
        previous["input"] = reservoir_input
        row["feedback_input_norm"] = float(sum(norm(v) for v in data["feedback"].values()))
        vectors["input"].append(reservoir_input)
        if first is not None:
            vector = first["centered"]
            row.update({"compact_norm": norm(first["compact"]), "steering_norm": norm(vector),
                        "steering_rms": float(np.sqrt(np.mean(vector ** 2))),
                        "normalization_active": bool(np.sqrt(np.mean(vector ** 2)) > 1.0),
                        "steering_layers": ";".join(str(k) for k in sorted(steering, key=lambda k: -1 if k is None else k)),
                        "direction_max_dev_across_layers": float(max(np.abs(s["direction"] - first["direction"]).max()
                                                                     for s in steering.values())),
                        "state_used_minus_before_norm": norm(used - before)})
            tracked("compact", first["compact"], step, row, "compact")
            tracked("steering", vector, step, row, "steering")
            deltas = [s["applied_delta"] for s in steering.values()]
            hidden = [s["hidden_before"] for s in steering.values()]
            row["applied_delta_norm"] = float(np.mean([norm(d) for d in deltas]))
            row["hidden_norm_before"] = float(np.mean([norm(h) for h in hidden]))
            row["applied_relative_norm"] = float(np.mean([norm(d) / norm(h) for d, h in zip(deltas, hidden)
                                                          if norm(h) > ZERO_NORM]))
            row["applied_vs_steering_cos"] = cosine(deltas[0], vector)
            vectors["compact"].append(first["compact"])
            vectors["steering"].append(vector)
            vectors["applied_delta"].append(deltas[0])
        if step < len(fly_logits) and step < len(clean_logits):
            m = distribution_metrics(clean_logits[step], fly_logits[step])
            row.update({"kl_clean_fly": m["kl"], "js_clean_fly": m["js"], "top1_clean": m["top1_reference"],
                        "top1_fly": m["top1_condition"], "top1_changed": m["top1_changed"],
                        "top1_prob_clean": m["top1_prob_reference"], "top1_prob_fly": m["top1_prob_condition"],
                        "margin_clean": m["margin_reference"], "margin_fly": m["margin_condition"],
                        "logit_diff_l2": m["logit_diff_l2"]})
        if step < len(off_logits):
            immediate = distribution_metrics(off_logits[step], fly_logits[step])
            history = distribution_metrics(clean_logits[step], off_logits[step])
            row.update({"kl_off_fly_immediate": immediate["kl"], "top1_changed_immediate": immediate["top1_changed"],
                        "logit_diff_l2_immediate": immediate["logit_diff_l2"],
                        "kl_clean_off_history": history["kl"], "top1_changed_history": history["top1_changed"]})
        rows.append(row)
    stacked = {key: np.asarray(value) for key, value in vectors.items() if value}
    geometry = {"steering": spectrum(stacked["steering"]) if "steering" in stacked else None,
                "compact": spectrum(stacked["compact"]) if "compact" in stacked else None,
                "reservoir_state": spectrum(stacked["state_after"]) if "state_after" in stacked else None,
                "reservoir_change": spectrum(np.diff(stacked["state_after"], axis=0))
                if "state_after" in stacked and len(stacked["state_after"]) > 1 else None}
    return {"rows": rows, "vectors": stacked, "reference_steps": reference_steps, "geometry": geometry,
            "summary": summarize(rows, geometry)}


def _stats(rows: list[dict], key: str, skip: int = 0) -> dict[str, Any]:
    values = np.array([r.get(key, math.nan) for r in rows[skip:]], dtype=float)
    valid = values[np.isfinite(values)]
    if not len(valid):
        return {"n": 0}
    return {"n": int(len(valid)), "undefined": int(len(values) - len(valid)), "min": float(valid.min()),
            "median": float(np.median(valid)), "mean": float(valid.mean()), "max": float(valid.max()),
            "first": float(valid[0]), "last": float(valid[-1])}


def summarize(rows: list[dict], geometry: dict) -> dict[str, Any]:
    keys = ("state_rms_after", "state_change_norm", "state_cos_prev", "state_cos_ref", "change_readout_fraction",
            "compact_cos_prev", "compact_cos_ref", "steering_norm", "steering_cos_prev", "steering_cos_ref",
            "steering_orth_ratio", "applied_relative_norm", "input_cos_prev", "kl_clean_fly", "js_clean_fly",
            "kl_off_fly_immediate", "kl_clean_off_history", "direction_max_dev_across_layers",
            "applied_vs_steering_cos")
    summary = {key: _stats(rows, key) for key in keys}
    half = len(rows) // 2
    for key in ("kl_clean_fly", "kl_off_fly_immediate", "kl_clean_off_history"):
        summary[f"{key}_first_half"] = _stats(rows[:half], key)
        summary[f"{key}_second_half"] = _stats(rows[half:], key)
    for key in ("top1_changed", "top1_changed_immediate", "top1_changed_history"):
        flags = [bool(r[key]) for r in rows if key in r]
        summary[key] = {"count": int(sum(flags)), "of": len(flags)}
    summary["geometry_uncentered_top1"] = {k: (v or {}).get("uncentered", {}).get("top1_energy_fraction")
                                           for k, v in geometry.items()}
    return summary


COLUMNS_FIRST = ("step", "token_id", "token_text")


def save(output: Path, metadata: dict[str, Any], analysis: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"{output} exists; probe outputs are never overwritten")
    (output / "plots").mkdir(parents=True)
    rows = analysis["rows"]
    for row in rows:
        step = row["step"]
        row["token_id"] = metadata["token_ids"][step] if step < len(metadata["token_ids"]) else None
        row["token_text"] = metadata["tokens_text"][step] if step < len(metadata["tokens_text"]) else ""
    columns = list(COLUMNS_FIRST) + [key for key in dict.fromkeys(k for r in rows for k in r) if key not in COLUMNS_FIRST]
    with (output / "token_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()})
    vectors = analysis["vectors"]
    np.savez_compressed(output / "reservoir_states.npz",
                        **{key: vectors[key] for key in ("state_before", "state_used", "state_after", "input")
                           if key in vectors})
    np.savez_compressed(output / "steering_vectors.npz",
                        **{key: vectors[key] for key in ("compact", "steering", "applied_delta") if key in vectors})
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=float) + "\n",
                                          encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(analysis["summary"], indent=2) + "\n")
    plot(output / "plots", rows, metadata)
    (output / "summary.md").write_text(markdown(metadata, analysis["summary"]), encoding="utf-8")


def plot(directory: Path, rows: list[dict], metadata: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = np.array([r["step"] for r in rows])

    def series(key):
        return np.array([r.get(key, math.nan) for r in rows], dtype=float)
    title = (f"{metadata['model']['display']} · {metadata['coupling']['coupling_mode']} "
             f"{metadata['coupling']['active_bypasses']} · leak {metadata['reservoir']['leak']:.3f}")
    figures = {
        "01_reservoir_rms.png": [("state_rms_after", "reservoir state RMS (after pass)")],
        "02_steering_norm.png": [("steering_norm", "‖centered steering vector‖"), ("applied_delta_norm", "‖applied delta‖")],
        "03_steering_cos_prev.png": [("steering_cos_prev", "cos(v_t, v_t-1)")],
        "04_steering_cos_ref.png": [("steering_cos_ref", "cos(v_t, v_ref)")],
        "05_parallel_orthogonal.png": [("steering_parallel", "parallel to v_ref"), ("steering_orthogonal", "orthogonal")],
        "06_state_vs_steering_direction.png": [("state_cos_prev", "reservoir cos(x_t, x_t-1)"),
                                                ("compact_cos_prev", "64-d readout cos"),
                                                ("steering_cos_prev", "steering cos(v_t, v_t-1)")],
        "07_kl.png": [("kl_clean_fly", "KL(Clean‖Fly)"), ("kl_off_fly_immediate", "KL(Fly-off‖Fly) immediate"),
                      ("kl_clean_off_history", "KL(Clean‖Fly-off) history")],
    }
    written = []
    for name, lines in figures.items():
        fig, axis = plt.subplots(figsize=(8, 3.6))
        drawn = False
        for key, label in lines:
            values = series(key)
            if np.isfinite(values).any():
                axis.plot(steps, values, marker=".", linewidth=1, label=label)
                drawn = True
        undefined = [int(s) for s, v in zip(steps, series(lines[0][0])) if not np.isfinite(v)]
        if undefined:
            axis.text(0.01, 0.02, f"undefined at steps {undefined[:12]}{'…' if len(undefined) > 12 else ''}",
                      transform=axis.transAxes, fontsize=7, color="gray")
        if name == "07_kl.png" and drawn:
            axis.set_yscale("symlog", linthresh=1e-5)
        axis.set_xlabel("step (forward pass; 0 = prompt prefill)")
        axis.set_title(title, fontsize=8)
        if drawn:
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, "not measured", ha="center", transform=axis.transAxes)
        fig.tight_layout()
        fig.savefig(directory / name, dpi=110)
        plt.close(fig)
        written.append(name)
    return written


def _fmt(stat: dict[str, Any], key: str = "median") -> str:
    return "n/a" if not stat or stat.get("n", 0) == 0 else f"{stat[key]:.4g}"


def markdown(metadata: dict[str, Any], summary: dict[str, Any]) -> str:
    s = summary
    lines = [f"# Steering dynamics probe: {metadata['label']}", "",
             f"- Model: {metadata['model']['display']} (rev {metadata['model']['revision'][:10]})",
             f"- Coupling: {metadata['coupling']['coupling_mode']} {metadata['coupling']['active_bypasses']}, "
             f"{metadata['reservoir']['updates_per_token']} reservoir update(s) per token",
             f"- Strength {metadata['settings']['strength']}, feedback {metadata['settings']['feedback']}, "
             f"leak {metadata['reservoir']['leak']:.4f}",
             f"- Steps: {metadata['steps']} ({metadata['token_source']}); reference steps {metadata['reference_steps']}",
             "", "| Quantity | median | min | max | first | last |", "|---|---|---|---|---|---|"]
    for key in ("state_rms_after", "state_change_norm", "state_cos_prev", "state_cos_ref", "change_readout_fraction",
                "compact_cos_prev", "steering_cos_prev", "steering_cos_ref", "steering_orth_ratio", "steering_norm",
                "applied_relative_norm", "input_cos_prev", "kl_clean_fly", "kl_off_fly_immediate",
                "kl_clean_off_history"):
        stat = s[key]
        lines.append(f"| {key} | {_fmt(stat)} | {_fmt(stat, 'min')} | {_fmt(stat, 'max')} | "
                     f"{_fmt(stat, 'first')} | {_fmt(stat, 'last')} |")
    lines += ["", f"- Top-1 changed (Clean vs Fly): {s['top1_changed']['count']}/{s['top1_changed']['of']}; "
              f"immediate (Fly-off vs Fly): {s['top1_changed_immediate']['count']}/{s['top1_changed_immediate']['of']}",
              f"- KL(Clean‖Fly) median, first vs second half: {_fmt(s['kl_clean_fly_first_half'])} → "
              f"{_fmt(s['kl_clean_fly_second_half'])}",
              f"- Uncentered top-1 energy fraction (1 = one fixed direction): {s['geometry_uncentered_top1']}",
              f"- {UNDEFINED}", ""]
    return "\n".join(lines)
