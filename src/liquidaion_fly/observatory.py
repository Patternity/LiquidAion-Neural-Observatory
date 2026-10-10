"""Launch the local anatomical neural observatory or run a real diagnostic."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_configuration(path: Path, cpu: bool = False) -> dict:
    config = json.loads(path.read_text())
    if cpu:
        config["device"] = "cpu"
    return config


def validate_graph(graph_dir: Path) -> dict:
    from .connectome import load_graph, sha256
    graph, ids = load_graph(graph_dir)
    source = json.loads((graph_dir / "provenance.json").read_text())
    if len(ids) != source["nodes"] or graph.nnz != source["edges"]:
        raise ValueError("Connectome shape differs from source provenance")
    for filename, key in [("edges.csv", "edge_sha256"), ("nodes.csv", "node_sha256")]:
        if sha256(graph_dir / filename) != source[key]:
            raise ValueError(f"Connectome integrity check failed: {filename}")
    return source


def check_device(device: str, replay: bool = False) -> dict:
    from .environment import inspect_environment
    info = inspect_environment()
    if device == "cuda" and not replay and not info["torch_cuda_available"]:
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver. Use --cpu only for explicit CPU validation, or --replay for offline viewing.")
    if device == "cuda" and not replay:
        import torch
        # Exercise a real kernel instead of inferring compatibility from card name.
        probe = torch.ones(4, device="cuda")
        if float((probe + probe).sum()) != 8:
            raise RuntimeError("CUDA kernel verification failed")
        free, total = torch.cuda.mem_get_info()
        info["selected_gpu"] = {"name": torch.cuda.get_device_name(), "total_bytes": total, "free_bytes": free}
    return info


def selected_model(config: dict, supplied: str | None) -> str:
    model = supplied or config["model"]
    path = Path(model).expanduser()
    if path.is_dir():
        for name in ["config.json", "tokenizer_config.json", "tokenizer.json"]:
            if not (path / name).is_file():
                raise FileNotFoundError(f"Model directory is missing {name}")
        if not (path / "model.safetensors").is_file() and not (path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError("Model directory has no pretrained safetensors weights")
        return str(path.resolve())
    if supplied:
        raise FileNotFoundError(f"Local model directory not found: {model}")
    return model


def engine_config(config: dict):
    from .observatory_engine import ObservatoryConfig
    accepted = {f.name for f in fields(ObservatoryConfig)}
    values = {k: v for k, v in config.items() if k in accepted}
    return ObservatoryConfig(**values)


def diagnostic(config: dict, model_path: str, graph_dir: Path, anatomy_dir: Path,
               output: Path, max_tokens: int = 8) -> dict:
    from .anatomy import ensure_anatomy
    from .observatory_engine import ObservatoryEngine
    from .recording import load_recording
    anatomy = ensure_anatomy(graph_dir, anatomy_dir)
    source = validate_graph(graph_dir)
    cfg = engine_config(config)
    engine = ObservatoryEngine(model_path, graph_dir, device=config["device"], config=cfg)
    engine.configure(max_new_tokens=max_tokens)
    baseline = engine.generate_pair("Say hello in one short sentence.", max_new_tokens=max_tokens)
    a = engine.demo_clean_vs_fly("Explain the sky's color in one sentence.", max_tokens=max_tokens)
    b = engine.demo_random_topology("Explain the sky's color in one sentence.", max_tokens=max_tokens)
    c = engine.demo_persistence("A friend helped me fix a bug.", "Write one short thank-you sentence.", max_tokens=max_tokens)
    engine.idle_tick(steps=5)
    output.mkdir(parents=True, exist_ok=True)
    engine.save_recording(output / "session.json")
    recording = load_recording(output / "session.json")
    result = {"device": config["device"], "model_instances": 1,
              "anatomy_neurons": len(anatomy.neuron_ids), "anatomy_meshes": list(anatomy.meshes),
              "connectome_hash": source["edge_sha256"], "chat": baseline,
              "experiment_a": a, "experiment_b": b, "experiment_c": c,
              "session_saved": True, "hardware": check_device(config["device"])}
    # Portable recordings include the actual state arrays in the same JSON file.
    def serial(value):
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    (output / "smoke.json").write_text(json.dumps(result, default=serial, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LiquidAion Neural Observatory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/observatory.json")
    parser.add_argument("--model-path")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/cx512")
    parser.add_argument("--anatomy", type=Path, default=ROOT / "data/anatomy")
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU validation, not target GPU acceptance")
    parser.add_argument("--replay", type=Path, help="Open a saved session without loading model weights")
    parser.add_argument("--smoke", action="store_true", help="Run real dual generation and three experiments")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results/observatory")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model-profile", help="Language model profile (see liquidaion_fly.models.PROFILES); "
                                                "default: the original Qwen3-0.6B or the GUI's saved choice")
    parser.add_argument("--debug-log", choices=("summary", "full"),
                        help="Opt-in per-generation inference log. 'summary' stores counts, stop reasons, "
                             "cache/hook/trim events and digests; 'full' also stores the rendered prompt, "
                             "token ids and generated text (the conversation itself)")
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "results/debug")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = load_configuration(args.config, args.cpu)
    if args.model_profile:
        from .models import get_profile
        profile = get_profile(args.model_profile)
        config.update(model_profile=profile.key, revision=profile.revision,
                      capture_layer=profile.capture_layer, steer_layer=profile.steer_layer)
    try:
        hardware = check_device(config["device"], replay=args.replay is not None or args.prepare_only)
        validate_graph(args.graph)
        from .anatomy import ensure_anatomy
        anatomy = ensure_anatomy(args.graph, args.anatomy)
        if args.prepare_only:
            print(json.dumps({"hardware": hardware, "anatomy_neurons": len(anatomy.neuron_ids)}, indent=2))
            return
        model_path = selected_model(config, args.model_path) if not args.replay else config["model"]
        if args.smoke:
            diagnostic(config, model_path, args.graph, args.anatomy, args.output, args.max_tokens)
            print(json.dumps({"status": "real_diagnostic_complete", "device": config["device"], "output": str(args.output)}))
            return
        debug_log = None
        if args.debug_log and not args.replay:
            from datetime import datetime
            path = args.debug_dir / f"inference-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
            debug_log = (path, args.debug_log)
            logging.getLogger(__name__).info("Inference debug log (%s): %s", args.debug_log, path)
        from .observatory_gui import run_gui
        run_gui(engine_config(config), model_path, args.graph, args.anatomy, replay_path=args.replay,
                device=config["device"], debug_log=debug_log)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Observatory startup failed: {exc}\n")


if __name__ == "__main__":
    main()
