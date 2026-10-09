"""Paired fixed-context measurements, controls, and cross-prompt persistence."""
from __future__ import annotations
import hashlib
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import torch
from .connectome import load_graph, sha256
from .environment import inspect_environment
from .llm import Coupling, LocalLLM, compare_logits
from .reservoir import Dynamics, Reservoir, random_control
LOG = logging.getLogger(__name__)


def run_experiments(config: dict, graph_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if config.get("device") == "cuda":
        # Deterministic cuBLAS kernels require this workspace setting before the first CUDA GEMM.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    graph, ids = load_graph(graph_dir)
    provenance = json.loads((graph_dir / "provenance.json").read_text())
    for name, key in [("edges.csv", "edge_sha256"), ("nodes.csv", "node_sha256")]:
        if sha256(graph_dir / name) != provenance[key]:
            raise ValueError("Data do not match recorded provenance")
    initial_environment = inspect_environment()
    LOG.info(json.dumps({"startup_environment": initial_environment}))
    start = time.perf_counter()
    llm = LocalLLM(config["model"], config["device"], config.get("revision"), config["threads"])
    summary = {"status": "running", "configuration": config, "provenance": provenance,
               "model_config": llm.model.config.to_dict(), "model_load_seconds": time.perf_counter() - start,
               "initial_environment": initial_environment, "conditions": [], "scope": "Untrained adapters; causal pathway demonstration, not semantic control"}
    shuffled = random_control(graph, config["seed"] + 1)
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")

    def make(mode: str, strength: float | None = None, feedback: float | None = None) -> Coupling:
        r = Reservoir(shuffled if mode == "random" else graph, config["features"], config["seed"], Dynamics(**config["dynamics"]))
        return Coupling(r, llm.model.config.hidden_size, config["capture_layer"], config["steer_layer"],
                        config["strength"] if strength is None else strength,
                        config["feedback"] if feedback is None else feedback, mode, config["seed"])

    def save(name: str, result: dict) -> None:
        serial = {k: v for k, v in result.items() if k not in {"logits", "states"}}
        serial["logits_sha256"] = hashlib.sha256(result["logits"].tobytes()).hexdigest()
        serial["last_step_top10_token_ids"] = np.argsort(result["logits"][-1])[-10:][::-1].tolist()
        (output / f"{name}.json").write_text(json.dumps(serial, indent=2) + "\n")
        if result["states"].size:
            np.savez_compressed(output / f"{name}_states.npz", states=result["states"], neuron_ids=np.asarray(ids))

    baselines = []
    for i, prompt in enumerate(config["prompts"]):
        baseline = llm.run(prompt, max_tokens=config["max_tokens"])
        baselines.append(baseline)
        save(f"p{i}_baseline", baseline)
        conditions = {}
        for mode in ("none", "static", "random", "fly"):
            result = llm.run(prompt, make(mode), forced_tokens=baseline["token_ids"])
            conditions[mode] = result
            save(f"p{i}_{mode}_fixed", result)
            metrics = compare_logits(baseline["logits"], result["logits"])
            metrics.update({"prompt_index": i, "mode": mode,
                            "mean_delta_relative_rms": float(np.mean([r["delta_relative_rms"] for r in result["trace"]])),
                            "max_delta_relative_rms": max(r["delta_relative_rms"] for r in result["trace"]),
                            "tokens_per_second": result["tokens_per_second"]})
            summary["conditions"].append(metrics)
            if mode == "none" and not np.array_equal(baseline["logits"], result["logits"]):
                raise AssertionError("Disabled intervention changed the baseline")
            LOG.info(json.dumps(metrics))
        generated = llm.run(prompt, make("fly"), max_tokens=config["max_tokens"])
        save(f"p{i}_fly_generated", generated)
        summary.setdefault("generated_examples", []).append({"prompt": prompt,
                           "baseline": baseline["text"], "fly": generated["text"],
                           "same_tokens": baseline["token_ids"] == generated["token_ids"]})
        if i == 0:
            repeated = llm.run(prompt, make("fly"), forced_tokens=baseline["token_ids"])
            save("p0_fly_repeat", repeated)
            summary["repeat"] = {"bitwise_equal_logits": bool(np.array_equal(conditions["fly"]["logits"], repeated["logits"])),
                                 "bitwise_equal_states": bool(np.array_equal(conditions["fly"]["states"], repeated["states"]))}
            no_feedback = llm.run(prompt, make("fly", feedback=0), forced_tokens=baseline["token_ids"])
            save("p0_fly_feedback_off", no_feedback)
            summary["feedback_comparison"] = compare_logits(no_feedback["logits"], conditions["fly"]["logits"])
            zero = llm.run(prompt, make("fly", strength=0), forced_tokens=baseline["token_ids"])
            save("p0_zero_strength", zero)
            summary["zero_strength_equal_baseline"] = bool(np.array_equal(zero["logits"], baseline["logits"]))
            summary["fly_vs_random_same_context"] = compare_logits(conditions["random"]["logits"], conditions["fly"]["logits"])
    if len(baselines) < 2:
        raise ValueError("Persistence experiment requires two prompts")
    c = make("fly")
    first = llm.run(config["prompts"][0], c, forced_tokens=baselines[0]["token_ids"])
    save("persistence_prompt_a", first)
    c.reservoir.snapshot(output / "state_after_a.npz")
    before_idle = c.reservoir.state.copy()
    c.reservoir.advance(steps=config["idle_steps"])
    after_idle = c.reservoir.state.copy()
    c.reservoir.snapshot(output / "state_after_idle.npz")
    persistent = llm.run(config["prompts"][1], c, forced_tokens=baselines[1]["token_ids"])
    save("persistence_b_retained", persistent)
    c.reservoir.reset()
    reset = llm.run(config["prompts"][1], c, forced_tokens=baselines[1]["token_ids"])
    save("persistence_b_reset", reset)
    c.reservoir.restore(output / "state_after_idle.npz")
    restored = llm.run(config["prompts"][1], c, forced_tokens=baselines[1]["token_ids"])
    save("persistence_b_restored", restored)
    summary["persistence"] = {**compare_logits(reset["logits"], persistent["logits"]),
                              "state_rms_before_idle": float(np.sqrt(np.mean(before_idle**2))),
                              "state_rms_after_idle": float(np.sqrt(np.mean(after_idle**2))),
                              "state_l2_idle_change": float(np.linalg.norm(before_idle - after_idle)),
                              "restored_bitwise_equal_logits": bool(np.array_equal(restored["logits"], persistent["logits"])),
                              "new_text_context_and_kv_cache_for_b": True}
    passed = all([summary["zero_strength_equal_baseline"], summary["repeat"]["bitwise_equal_logits"],
                  summary["repeat"]["bitwise_equal_states"], summary["persistence"]["restored_bitwise_equal_logits"],
                  summary["persistence"]["logit_rms_difference"] > 0,
                  summary["fly_vs_random_same_context"]["logit_rms_difference"] > 0,
                  all(r["logit_rms_difference"] > 0 for r in summary["conditions"] if r["mode"] == "fly")])
    summary["status"] = "cpu_pathway_verified" if passed and config["device"] == "cpu" else "pathway_verified" if passed else "verification_failed"
    summary["environment"] = inspect_environment()
    summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    summary["total_seconds"] = time.perf_counter() - start
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not passed:
        raise AssertionError("Real end-to-end checks failed; see summary.json")
    return summary
