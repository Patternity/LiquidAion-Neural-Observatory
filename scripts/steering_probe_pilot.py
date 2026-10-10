"""Diagnostic pilot: the steering_dynamics_probe.py plugin outside the GUI, plus one legacy run.

The plugin's settings can be overridden from the command line (e.g. when the GPU only fits a
smaller model). Example:

  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python scripts/steering_probe_pilot.py \\
      --profile qwen3-0.6b-base --bypasses 2:2,26:26 --legacy 6:14 --output results/steering-dynamics-pilot
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from threading import Event

from liquidaion_fly.models import get_profile
from liquidaion_fly.observatory import engine_config
from liquidaion_fly.observatory_engine import ObservatoryEngine
from liquidaion_fly.plugins import ExperimentAPI, load_plugin

PLUGIN = Path("plugins/examples/steering_dynamics_probe.py")


def pairs(text):
    return [tuple(int(x) for x in pair.split(":")) for pair in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen3-4b-base")
    parser.add_argument("--bypasses", type=pairs, default=None, help="CAPTURE:STEER,... (default: the plugin's)")
    parser.add_argument("--leaks", default=None, help="Comma-separated leak rates (default: the plugin's)")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--legacy", type=pairs, default=None, help="CAPTURE:STEER for one extra legacy run at leak 0.2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = get_profile(args.profile)
    settings = json.loads(Path("configs/observatory.json").read_text())
    settings.update(model_profile=profile.key, revision=profile.revision, capture_layer=profile.capture_layer,
                    steer_layer=profile.steer_layer, max_context_tokens=1024, max_new_tokens=48)
    engine = ObservatoryEngine(profile.model_id, Path("data/cx512"), device="cuda", config=engine_config(settings))
    run = load_plugin(PLUGIN)
    overrides = {"BYPASSES": args.bypasses, "LEAK_RATES": [float(x) for x in args.leaks.split(",")] if args.leaks
                 else None, "MAX_TOKENS": args.tokens}
    run.__globals__.update({key: value for key, value in overrides.items() if value is not None})
    api = ExperimentAPI(engine, PLUGIN, Event(), print, output_root=args.output)
    api._output_dir = args.output / f"{datetime.now():%Y%m%d-%H%M%S}-{profile.key}"
    api._output_dir.mkdir(parents=True)
    run(api)
    if args.legacy:
        tokens = json.loads(next(api.output_dir.glob("leak-*/metadata.json")).read_text())["token_ids"]
        api.new_session()
        api.set_bypasses(args.legacy)
        api.set_coupling(leak_rate=0.2)
        capture, steer = args.legacy[0]
        result = api.steering_probe(run.__globals__["PROMPT"], len(tokens), name=f"legacy-c{capture}-s{steer}-leak-0.200",
                                    forced_tokens=tokens)
        print("legacy run:", result["output"])
    print("output:", api.output_dir)


if __name__ == "__main__":
    main()
