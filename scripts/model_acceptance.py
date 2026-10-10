"""Real-GPU acceptance of one language-model profile (weights, architecture, coupling, speed, memory).

  PYTHONPATH=src python scripts/model_acceptance.py --profile qwen3-4b-base --output results/model-acceptance
  PYTHONPATH=src python scripts/model_acceptance.py --profile qwen3-0.6b-base --injection --output results/model-acceptance
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from liquidaion_fly.continuous import ContinuousConfig, ContinuousRun, load_run
from liquidaion_fly.models import get_profile
from liquidaion_fly.observatory import engine_config
from liquidaion_fly.observatory_engine import ObservatoryEngine

SEED = "Однажды в студёную зимнюю пору"
INJECTION = " Из-за деревьев неожиданно появился огромный зелёный мамонт."


def hooks(engine) -> int:
    return engine._hook_count()


def logit_effect(engine, prompt, tokens=24):
    """Same-prefix Clean vs Fly comparison at the engine's current layer pair."""
    result = engine.scientific_compare(prompt, "clean_vs_fly", max_tokens=tokens)
    return {"capture_layer": engine.config.capture_layer, "steer_layer": engine.config.steer_layer,
            "matched_steps": result["matched_steps"], **result["metrics"],
            "mean_delta_relative_rms": sum(r["delta_relative_rms"] for r in result["condition_trace"]) /
            max(len(result["condition_trace"]), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alternative", default=None, help="CAPTURE:STEER alternative layer pair")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--injection", action="store_true", help="Also run the full external-injection check")
    args = parser.parse_args()
    profile = get_profile(args.profile)
    out = args.output / profile.key
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path("configs/observatory.json").read_text())
    settings.update(model_profile=profile.key, revision=profile.revision, capture_layer=profile.capture_layer,
                    steer_layer=profile.steer_layer, max_context_tokens=1024, max_new_tokens=48)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine = ObservatoryEngine(profile.model_id, Path("data/cx512"), device="cuda", config=engine_config(settings))
    load_seconds = time.perf_counter() - started
    info = engine.model_info()
    report = {"profile": profile.as_dict(), "loaded": info, "load_seconds_total": load_seconds,
              "after_load_allocated_bytes": torch.cuda.memory_allocated(), "checks": []}
    llm = engine.llm.model
    report["device_placement"] = sorted({str(p.device) for p in llm.parameters()})
    report["parameter_dtypes"] = sorted({str(p.dtype) for p in llm.parameters()})
    resolved = getattr(llm.config, "_commit_hash", None)

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else "FAIL ") + name + (f" [{detail}]" if detail else ""), flush=True)

    check("weights loaded from the pinned revision", resolved in (None, profile.revision), resolved)
    check("architecture read from config", info["hidden_size"] == llm.config.hidden_size
          and info["num_hidden_layers"] == llm.config.num_hidden_layers,
          f"{info['num_hidden_layers']} layers, hidden {info['hidden_size']}, vocab {info['vocab_size']}")
    check("projection sizes follow hidden size", tuple(engine.coupling.encoder.shape) == (info["hidden_size"], 64)
          and tuple(engine.coupling.decoder.shape) == (64, info["hidden_size"]))
    check("no CPU offload", report["device_placement"] == ["cuda:0"], report["device_placement"])
    prompt = SEED if profile.interaction == "raw" else "Привет! Расскажи коротко, кто ты."
    pair = engine.generate_pair(prompt, max_new_tokens=48)
    report["generate_pair"] = {branch: {k: pair[branch][k] for k in ("text", "stop_reason", "tokens_per_second")}
                               | {"tokens": len(pair[branch]["token_ids"])} for branch in ("clean", "fly")}
    report["prompt_ids_are_raw"] = profile.interaction != "raw" or pair["clean"]["prompt_ids"] == \
        engine.llm.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    check("clean and fly generation", pair["clean"]["token_ids"] and pair["fly"]["token_ids"])
    check("base profile input is the raw seed (no template)", report["prompt_ids_are_raw"])
    check("no hooks left after generation", hooks(engine) == 0)
    effects = [logit_effect(engine, prompt)]
    layers = info["num_hidden_layers"]
    if args.alternative:
        capture, steer = (int(x) for x in args.alternative.split(":"))
    else:
        capture, steer = layers // 2, layers - 2
    fingerprint = engine.coupling.projection_fingerprint
    engine.configure(capture_layer=capture, steer_layer=steer)
    check("same-model layer change keeps projections", engine.coupling.projection_fingerprint == fingerprint)
    effects.append(logit_effect(engine, prompt))
    report["logit_effects"] = effects
    check("steering changes logits at both layer pairs", all(e["mean_kl_baseline_to_condition"] > 0 for e in effects),
          [f"{e['capture_layer']}->{e['steer_layer']}: KL {e['mean_kl_baseline_to_condition']:.2e}" for e in effects])
    check("layer pairs give different effects (the selection is used)",
          effects[0]["mean_kl_baseline_to_condition"] != effects[1]["mean_kl_baseline_to_condition"])
    engine.configure(capture_layer=profile.capture_layer, steer_layer=profile.steer_layer)
    check("no hooks left after experiments", hooks(engine) == 0)
    base = ContinuousConfig(seed_text=SEED, window=128, total_tokens=args.tokens, decoding="sample", temperature=1.0,
                            top_p=0.95, sampling_seed=42, eos_policy="suppress", checkpoints=(64,), probe_tokens=8)
    speeds = {}
    for condition in ("clean", "fly"):
        torch.cuda.reset_peak_memory_stats()
        summary = ContinuousRun(engine, replace(base, condition=condition), out / f"continuous-{condition}").run()
        speeds[condition] = {"tokens_per_second": summary["tokens_per_second"],
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                             "reservoir_advances_per_token": summary["reservoir_advances"] / summary["forward_calls"]}
    report["continuous"] = speeds
    report["continuous_divergence"] = next((i for i, (a, b) in enumerate(zip(
        load_run(out / "continuous-clean")["tokens"]["generated"], load_run(out / "continuous-fly")["tokens"]["generated"]))
        if a != b), None)
    check("continuous generation (clean, fly)", speeds["fly"]["reservoir_advances_per_token"] == 2.0,
          f"clean {speeds['clean']['tokens_per_second']:.1f} tok/s, fly {speeds['fly']['tokens_per_second']:.1f} tok/s")
    check("fits in 8 GB", speeds["fly"]["peak_reserved_bytes"] < 8 * 2**30,
          f"peak reserved {speeds['fly']['peak_reserved_bytes'] / 2**30:.2f} GiB")
    if args.injection:
        run = ContinuousRun(engine, replace(base, condition="fly", total_tokens=320, cue_target="da", cue_strength=0.5,
                                            injections=((100, INJECTION),)), out / "injection")
        summary = run.run()
        rows = load_run(out / "injection")["telemetry"]
        record = summary["injections"][0]
        categories = []
        for row in rows:
            if not categories or categories[-1] != row["category"]:
                categories.append(row["category"])
        report["injection"] = {"record": record, "category_sequence": categories,
                               "external_text": "".join(r["fragment"] for r in rows if r["source"] == "external")}
        check("injection inserted exactly", report["injection"]["external_text"] == INJECTION
              and record["status"] == "inserted")
        check("AUTONOMOUS -> EXTERNAL -> POST-INJECTION -> AUTONOMOUS", categories ==
              ["autonomous", "external", "post_injection", "autonomous"], categories)
        cue = [r for r in rows if r["cue_applied"]]
        before = sum(r["da_mean"] for r in rows[80:100]) / 20
        during = sum(r["da_mean"] for r in cue) / max(len(cue), 1)
        check("DA cue applied to injected tokens and raises DA activity", len(cue) == record["inserted"] and during > before,
              f"{len(cue)} cued updates, DA mean {before:+.3f} -> {during:+.3f}")
    report["passed"] = all(c["passed"] for c in report["checks"])
    (out / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"profile": profile.key, "passed": report["passed"]}))


if __name__ == "__main__":
    main()
