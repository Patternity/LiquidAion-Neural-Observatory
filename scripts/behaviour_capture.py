"""Deterministic behaviour capture for before/after refactoring comparisons (exact equality).

  PYTHONPATH=<src> python scripts/behaviour_capture.py cpu OUT.json
  PYTHONPATH=<src> CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/behaviour_capture.py gpu OUT.json

Run it with PYTHONPATH pointing at two checkouts (e.g. a git worktree of the previous commit) and
compare the outputs with scripts/behaviour_compare.py. Used for the plugin migration (docs/PLUGINS.md).
"""
import json
import sys
import tempfile
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import torch

import liquidaion_fly.observatory_engine as engine_module
from liquidaion_fly.continuous import ContinuousConfig, ContinuousRun, load_run
from liquidaion_fly.observatory_engine import ObservatoryConfig, ObservatoryEngine

REPO = Path(__file__).resolve().parents[1]


class Tok:
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"{m['role']}:{m['content']}\n" for m in messages) + "assistant:"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[ord(c) % 64 for c in text]])}

    def decode(self, tokens, **kwargs):
        return "".join(chr(int(t) + 64) for t in tokens)


def arr(x):
    return np.asarray(x, dtype=np.float64).tolist()


def graph_dir(tmp):
    d = Path(tmp) / "graph"
    d.mkdir(parents=True)
    (d / "nodes.csv").write_text("root_id,nt_type\n1,ACH\n2,DA\n3,GABA\n4,DA\n5,GLUT\n")
    (d / "edges.csv").write_text("pre_root_id,post_root_id,syn_count\n1,2,2\n2,3,3\n3,1,4\n3,4,1\n4,5,2\n5,1,3\n2,5,1\n")
    return d


def cpu_engine(tmp, **config):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(321)
    torch.set_num_threads(1)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8)
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).eval()
    engine_module.LocalLLM = lambda *a, **k: SimpleNamespace(model=model, tokenizer=Tok(), device=torch.device("cpu"),
                                                             revision="software-test")
    settings = dict(capture_layer=0, steer_layer=2, features=8, max_context_tokens=96, max_new_tokens=4, threads=1)
    settings.update(config)
    return ObservatoryEngine("software-test-only", graph_dir(tmp), config=ObservatoryConfig(**settings))


def battery(engine, tmp, tokens, label, out):
    r = {}
    pair = engine.generate_pair("Hello", max_new_tokens=tokens)
    pair2 = engine.generate_pair("Again", max_new_tokens=tokens)
    r["pair"] = {b: pair[b]["token_ids"] for b in pair} | {"second_" + b: pair2[b]["token_ids"] for b in pair2}
    r["fly_trace"] = [{k: v for k, v in rec.items() if isinstance(v, (int, float))} for rec in pair2["fly"]["trace"]]
    r["state_after_pairs"] = arr(engine.reservoir.state)
    r["time_after_pairs"] = engine.reservoir.time
    sci = engine.scientific_compare("Probe", "clean_vs_fly", max_tokens=tokens)
    r["sci_clean_fly"] = sci["metrics"]
    sci = engine.scientific_compare("Probe", "fly_vs_random", max_tokens=tokens)
    r["sci_fly_random"] = sci["metrics"]
    r["persistence"] = engine.demo_persistence("A", "B", max_tokens=tokens, idle_steps=3)["metrics"]
    engine.idle_tick(5)
    r["state_after_idle"] = arr(engine.reservoir.state)
    for condition in ("fly", "random", "fly_reset"):
        config = ContinuousConfig(seed_text="Once upon", window=12, total_tokens=3 * tokens, condition=condition,
                                  reset_interval=5 if condition == "fly_reset" else None, checkpoints=(4,),
                                  probe_tokens=2, decoding="sample", temperature=1.0, top_p=0.9, sampling_seed=7,
                                  cue_target="da", cue_strength=0.5, injections=((3, "Hi!"),),
                                  leak_schedule=((6, 0.5),), eos_policy="suppress", strength=0.6, feedback=0.4)
        path = Path(tmp) / f"{label}-{condition}"
        run = ContinuousRun(engine, config, path)
        summary = run.run()
        saved = load_run(path)
        r[f"continuous_{condition}"] = {"tokens": saved["tokens"], "advances": summary["reservoir_advances"],
                                        "final_state_rms": summary["final_state_rms"],
                                        "probes": [{k: v for k, v in p.items() if isinstance(v, (int, float, dict))}
                                                   for p in summary["probes"]],
                                        "telemetry": [{k: v for k, v in row.items() if isinstance(v, (int, float, bool))}
                                                      for row in saved["telemetry"]]}
    engine.new_session()
    out[label] = r


def main():
    mode, target = sys.argv[1], Path(sys.argv[2])
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "cpu":
            for label, config in (("legacy_forward", dict(capture_layer=0, steer_layer=2)),
                                  ("legacy_self", dict(capture_layer=1, steer_layer=1)),
                                  ("legacy_backward", dict(capture_layer=3, steer_layer=1)),
                                  ("sync_multi", dict(bypasses=[[0, 2, True], [3, 1, True], [0, 2, True]]))):
                engine = cpu_engine(Path(tmp) / label, **config)
                battery(engine, tmp, 4, label, out)
        else:
            from liquidaion_fly.models import get_profile
            from liquidaion_fly.observatory import engine_config
            profile = get_profile("qwen3-0.6b-base")
            settings = json.loads((REPO / "configs/observatory.json").read_text())
            settings.update(model_profile=profile.key, revision=profile.revision, capture_layer=6, steer_layer=14,
                            max_context_tokens=1024, max_new_tokens=32)
            engine = ObservatoryEngine(profile.model_id, REPO / "data/cx512", device="cuda", config=engine_config(settings))
            for label, rows in (("gpu_legacy_6_14", [[6, 14, True]]), ("gpu_sync_2_26", [[2, 2, True], [26, 26, True]])):
                engine.configure(bypasses=rows)
                battery(engine, tmp, 16, label, out)
            engine.configure(bypasses=[[6, 14, True]])
            engine.new_session()
            torch.cuda.synchronize()
            started = time.perf_counter()
            pair = engine.generate_pair("Однажды в студёную зимнюю пору", max_new_tokens=128)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            run = ContinuousRun(engine, ContinuousConfig(seed_text="Однажды в студёную зимнюю пору", window=128,
                                                         total_tokens=200, decoding="sample", temperature=1.0,
                                                         top_p=0.95, sampling_seed=42, eos_policy="suppress",
                                                         checkpoints=(), probe_tokens=1), Path(tmp) / "bench")
            summary = run.run()
            out["benchmark"] = {"generate_pair_256_tokens_s": elapsed,
                                "pair_tokens": len(pair["clean"]["token_ids"]) + len(pair["fly"]["token_ids"]),
                                "continuous_fly_tok_s": summary["tokens_per_second"]}
    target.write_text(json.dumps(out, default=float))
    print("written", target, list(out))


if __name__ == "__main__":
    main()
