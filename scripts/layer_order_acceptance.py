"""Real-GPU check of capture/steering layer pairs in every order (capture below, on, or above steering).

  PYTHONPATH=src python scripts/layer_order_acceptance.py --profile qwen3-0.6b-base --output results/layer-order
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquidaion_fly.continuous import ContinuousConfig, ContinuousRun, load_run
from liquidaion_fly.models import coupling_order, get_profile
from liquidaion_fly.observatory import engine_config
from liquidaion_fly.observatory_engine import ObservatoryEngine

SEED = "Однажды в студёную зимнюю пору"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen3-0.6b-base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", default="6:14,14:14,14:6,26:2", help="Comma-separated CAPTURE:STEER pairs")
    parser.add_argument("--tokens", type=int, default=120)
    args = parser.parse_args()
    profile = get_profile(args.profile)
    out = args.output / profile.key
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path("configs/observatory.json").read_text())
    settings.update(model_profile=profile.key, revision=profile.revision, capture_layer=profile.capture_layer,
                    steer_layer=profile.steer_layer, max_context_tokens=1024, max_new_tokens=48)
    engine = ObservatoryEngine(profile.model_id, Path("data/cx512"), device="cuda", config=engine_config(settings))
    prompt = SEED if profile.interaction == "raw" else "Привет! Расскажи коротко, кто ты."
    base = ContinuousConfig(seed_text=SEED, window=128, total_tokens=args.tokens, decoding="sample", temperature=1.0,
                            top_p=0.95, sampling_seed=42, eos_policy="suppress", condition="fly", checkpoints=(64,),
                            probe_tokens=8)
    report = {"profile": profile.key, "pairs": [], "checks": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else "FAIL ") + name + (f" [{detail}]" if detail else ""), flush=True)

    for pair in args.pairs.split(","):
        capture, steer = (int(x) for x in pair.split(":"))
        engine.configure(capture_layer=capture, steer_layer=steer)
        order = coupling_order(capture, steer)
        compare = engine.scientific_compare(prompt, "clean_vs_fly", max_tokens=24)
        trace = compare["condition_trace"]
        name = f"c{capture}-s{steer}"
        summary = ContinuousRun(engine, base, out / name).run()
        rows = load_run(out / name)["telemetry"]
        fired = sorted({(r["capture_layer"], r["steer_layer"]) for r in rows})
        report["pairs"].append({"capture_layer": capture, "steer_layer": steer, "order": order,
                                "mean_kl_clean_to_fly": compare["metrics"]["mean_kl_baseline_to_condition"],
                                "argmax_disagreement": compare["metrics"]["argmax_disagreement_fraction"],
                                "mean_delta_relative_rms": sum(r["delta_relative_rms"] for r in trace) / max(len(trace), 1),
                                "tokens_per_second": summary["tokens_per_second"],
                                "reservoir_advances_per_token": summary["reservoir_advances"] / summary["forward_calls"],
                                "text": load_run(out / name)["text"][:240]})
        check(f"{name} ({order}): hooks fired on the selected layers", fired == [(capture, steer)], fired)
        check(f"{name}: two reservoir updates per token", report["pairs"][-1]["reservoir_advances_per_token"] == 2.0)
        check(f"{name}: steering changes the logits", report["pairs"][-1]["mean_kl_clean_to_fly"] > 0,
              f"KL {report['pairs'][-1]['mean_kl_clean_to_fly']:.2e}")
        check(f"{name}: no hooks left", engine._hook_count() == 0)
    report["passed"] = all(c["passed"] for c in report["checks"])
    (out / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"profile": profile.key, "passed": report["passed"]}))


if __name__ == "__main__":
    main()
