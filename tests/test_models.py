"""Model profiles, substrate switching and adjustable coupling layers (random tiny weights)."""
import gc
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from conftest import TestTokenizer, assert_no_hooks, write_graph
from liquidaion_fly.continuous import ContinuousConfig, ContinuousRun, load_run
from liquidaion_fly.models import LEGACY_PROFILE, PROFILES, get_profile, valid_layer_pair
from liquidaion_fly.observatory_engine import ObservatoryConfig, ObservatoryEngine

# Tiny stand-ins with different architectures per profile: dimensions must come from the model.
SHAPES = {"qwen3-0.6b": (32, 3), "qwen3-0.6b-base": (48, 4), "qwen3-1.7b-base": (48, 5), "qwen3-4b-base": (64, 6)}


@pytest.fixture
def factory(monkeypatch):
    loads = []
    failing = set()

    def load(source, device="cpu", revision=None, threads=1, profile=None):
        hidden, layers = SHAPES[profile.key]
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=64, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        loads.append((source, revision, profile.key))
        if profile.key in failing:
            raise OSError("simulated download failure")
        return SimpleNamespace(model=model, tokenizer=TestTokenizer(), device=torch.device("cpu"), revision=revision,
                               dtype_name="float32", quantization=profile.quantization, attention="eager",
                               load_seconds=0.0)
    monkeypatch.setattr("liquidaion_fly.observatory_engine.LocalLLM", load)
    load.failing = failing
    load.loads = loads
    return load


def make_engine(tmp_path, **config):
    settings = dict(capture_layer=0, steer_layer=1, features=8, max_context_tokens=96, max_new_tokens=3, threads=1)
    settings.update(config)
    return ObservatoryEngine("Qwen/Qwen3-0.6B", write_graph(tmp_path / "graph"), config=ObservatoryConfig(**settings))


def test_four_pinned_profiles_and_legacy_revision_unchanged():
    assert list(PROFILES) == ["qwen3-0.6b", "qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base"]
    ids = [p.model_id for p in PROFILES.values()]
    assert ids == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
    assert all(len(p.revision) == 40 and int(p.revision, 16) >= 0 for p in PROFILES.values())
    legacy = get_profile(LEGACY_PROFILE)
    assert legacy.revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert (legacy.kind, legacy.interaction, legacy.dtype, legacy.attention, legacy.quantization) == \
        ("instruct", "chat", "float16", "eager", None)
    assert (legacy.capture_layer, legacy.steer_layer) == (6, 14)
    assert all(p.interaction == "raw" for p in PROFILES.values() if p.kind == "base")
    assert get_profile("qwen3-4b-base").quantization == "nf4"
    with pytest.raises(ValueError):
        get_profile("qwen3-8b")


def test_layer_pair_rule():
    assert valid_layer_pair(0, 1, 2) and valid_layer_pair(26, 27, 28) and valid_layer_pair(6, 14, 28)
    assert valid_layer_pair(5, 5, 28) and valid_layer_pair(8, 7, 28) and valid_layer_pair(27, 0, 28)
    assert not valid_layer_pair(27, 28, 28) and not valid_layer_pair(28, 3, 28) and not valid_layer_pair(-1, 3, 28)
    assert not valid_layer_pair(3, -1, 28)


def test_projection_dimensions_follow_the_loaded_model_and_switch_releases_old_model(factory, tmp_path):
    engine = make_engine(tmp_path)
    assert engine.coupling.encoder.shape == (32, 8) and engine.coupling.decoder.shape == (8, 32)
    old = weakref.ref(engine.llm.model)
    engine.generate_pair("Hi")
    info = engine.switch_model("qwen3-4b-base", 1, 4)
    gc.collect()
    assert old() is None, "The previous model must be released"
    assert info["hidden_size"] == 64 and info["num_hidden_layers"] == 6 and info["profile"] == "qwen3-4b-base"
    assert engine.coupling.encoder.shape == (64, 8) and engine.coupling.decoder.shape == (8, 64)
    assert engine.reservoir.state.shape == (3,) and np.all(engine.reservoir.state == 0)
    assert engine.sessions["clean"].messages == [] and engine.sessions["fly"].cache is None
    assert engine.recording.events[0]["type"] == "model_switched"
    assert engine.metadata["model_profile"] == "qwen3-4b-base"
    assert engine.metadata["model_revision"] == get_profile("qwen3-4b-base").revision
    assert factory.loads[-1][1] == get_profile("qwen3-4b-base").revision
    pair = engine.generate_pair("Again")
    assert pair["fly"]["trace"][0]["delta_relative_rms"] > 0
    assert_no_hooks(engine)


def test_failed_load_restores_previous_model_and_never_substitutes(factory, tmp_path, monkeypatch):
    engine = make_engine(tmp_path)
    with pytest.raises(RuntimeError, match="restored"):
        engine.switch_model("qwen3-4b-base", 2, 9)  # invalid for the 6-layer stand-in
    assert engine.profile.key == LEGACY_PROFILE and engine.model_layers() == 3
    engine.generate_pair("Still works")
    factory.failing.add("qwen3-1.7b-base")
    with pytest.raises(RuntimeError, match="simulated download failure"):
        engine.switch_model("qwen3-1.7b-base")
    assert engine.profile.key == LEGACY_PROFILE
    with pytest.raises(ValueError):
        engine.switch_model("not-a-model")
    assert_no_hooks(engine)


def test_same_model_layer_change_keeps_projections_and_state_and_is_used_by_inference(factory, tmp_path, monkeypatch):
    engine = make_engine(tmp_path, model_profile="qwen3-4b-base", capture_layer=1, steer_layer=3)
    engine.generate_pair("Warm")
    state = engine.reservoir.state.copy()
    fingerprint = engine.coupling.projection_fingerprint
    engine.configure(capture_layer=2, steer_layer=5)
    np.testing.assert_array_equal(engine.reservoir.state, state)
    assert engine.coupling.projection_fingerprint == fingerprint
    assert engine.sessions["fly"].cache is None
    assert any(e["type"] == "layers_changed" and e["new"] == [2, 5] for e in engine.recording.events)
    seen = []
    layers = engine.llm.model.model.layers
    original = engine.llm.model.forward

    def forward(*args, **kwargs):
        seen.append([index for index, layer in enumerate(layers) if layer._forward_hooks])
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.llm.model, "forward", forward)
    pair = engine.generate_pair("Next")
    fly_calls = seen[len(pair["clean"]["token_ids"]):]
    assert fly_calls and all(call == [2, 5] for call in fly_calls), "Hooks sit exactly on the selected layers"
    assert all(len(layer._forward_hooks) <= 1 for layer in layers)
    for bad in ({"capture_layer": 6, "steer_layer": 5}, {"capture_layer": 2, "steer_layer": 6},
                {"capture_layer": -1, "steer_layer": 3}):
        with pytest.raises(ValueError):
            engine.configure(**bad)
    assert (engine.config.capture_layer, engine.config.steer_layer) == (2, 5)
    assert_no_hooks(engine)


def test_base_profile_uses_raw_continuation_without_template(factory, tmp_path):
    engine = make_engine(tmp_path, model_profile="qwen3-0.6b-base")
    pair = engine.generate_pair("Once upon")
    tokenizer = engine.llm.tokenizer
    assert pair["clean"]["prompt_ids"] == tokenizer("Once upon")["input_ids"][0].tolist()
    reply = engine.sessions["clean"].messages[-1]["content"]
    second = engine.generate_pair(" and then")
    assert second["clean"]["prompt_ids"] == tokenizer("Once upon" + reply + " and then")["input_ids"][0].tolist()
    chat = make_engine(tmp_path / "chat")
    assert chat.generate_pair("Once upon")["clean"]["prompt_ids"] == \
        tokenizer("user:Once upon\nassistant:")["input_ids"][0].tolist()
    with pytest.raises(ValueError, match="Base model"):
        ContinuousRun(engine, ContinuousConfig(seed_text="Once", window=8, total_tokens=4, checkpoints=(),
                                               prompt_policy="chat_prefix"))


def test_metadata_records_model_layers_and_projections_and_restore_checks_profile(factory, tmp_path):
    engine = make_engine(tmp_path, model_profile="qwen3-1.7b-base", capture_layer=1, steer_layer=4)
    info = engine.metadata["language_model"]
    assert (info["profile"], info["capture_layer"], info["steer_layer"], info["hidden_size"]) == \
        ("qwen3-1.7b-base", 1, 4, 48)
    assert info["projection_fingerprint"] == engine.coupling.projection_fingerprint
    run = ContinuousRun(engine, ContinuousConfig(seed_text="Once", window=8, total_tokens=6, checkpoints=()),
                        tmp_path / "run")
    run.run()
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())["language_model"]
    assert (meta["profile"], meta["capture_layer"], meta["steer_layer"]) == ("qwen3-1.7b-base", 1, 4)
    engine.generate_pair("Hi")
    path = engine.save_recording(tmp_path / "session.json")
    engine.switch_model("qwen3-0.6b-base", 1, 2)
    with pytest.raises(ValueError, match="language model"):
        engine.restore_session(path)


def test_continuous_generation_runs_on_every_profile_with_correct_updates(factory, tmp_path):
    engine = make_engine(tmp_path)
    for key, (hidden, layers) in SHAPES.items():
        if engine.profile.key != key:
            engine.switch_model(key, 0, layers - 1)
        run = ContinuousRun(engine, ContinuousConfig(seed_text="Once", window=8, total_tokens=12, checkpoints=(4,),
                                                     probe_tokens=2, cue_target="da", cue_strength=0.5,
                                                     injections=((3, "Hi"),)))
        summary = run.run()
        assert summary["reservoir_advances"] == 2 * summary["forward_calls"] == 2 * summary["stream_tokens"]
        assert summary["injections"][0]["status"] == "inserted"
        assert all(p["retained_first_logits_equal_main"] for p in summary["probes"])
        assert run.coupling.decoder.shape == (8, hidden)
        assert_no_hooks(engine)


@pytest.mark.parametrize("capture, steer, order", [(2, 2, "same_layer"), (4, 1, "steer_first")])
def test_same_and_reversed_layer_pairs_drive_chat_and_continuous_generation(factory, tmp_path, capture, steer, order):
    engine = make_engine(tmp_path, model_profile="qwen3-1.7b-base")
    engine.configure(capture_layer=capture, steer_layer=steer)
    assert engine.model_info()["coupling_order"] == order
    assert any(e["type"] == "layers_changed" and e["order"] == order for e in engine.recording.events)
    pair = engine.generate_pair("Warm")
    assert pair["fly"]["token_ids"] and pair["clean"]["token_ids"]
    assert_no_hooks(engine)
    run = ContinuousRun(engine, ContinuousConfig(seed_text="Once", window=8, total_tokens=12, checkpoints=(4,),
                                                 probe_tokens=2, injections=((3, "Hi"),)), tmp_path / "run")
    summary = run.run()
    assert summary["reservoir_advances"] == 2 * summary["forward_calls"] == 2 * summary["stream_tokens"]
    rows = load_run(tmp_path / "run")["telemetry"]
    assert rows and {(r["capture_layer"], r["steer_layer"]) for r in rows} == {(capture, steer)}
    assert_no_hooks(engine)
