"""Random-weight tiny-transformer software tests, not scientific evidence."""
from threading import Event

import numpy as np
import pytest
import torch

from conftest import assert_no_hooks
from liquidaion_fly.observatory_engine import ContextOverflowError, ObservatoryEngine


def test_single_model_separate_cache_histories_and_stream(engine):
    tokens = []
    snapshots = []
    pair = engine.generate_pair("Hi", on_token=lambda *event: tokens.append(event),
                                on_telemetry=snapshots.append)
    assert len(engine._test_load_calls) == 1
    assert {event[0] for event in tokens} == {"clean", "fly"}
    assert all(len(event) == 3 for event in tokens)
    assert len(engine.sessions["clean"].messages) == 2
    assert len(engine.sessions["fly"].messages) == 2
    assert engine.sessions["clean"].cache is not engine.sessions["fly"].cache
    assert len(snapshots[-1]["state"]) == 3
    assert pair["fly"]["trace"][0]["input_norm"] > 0
    assert pair["fly"]["trace"][0]["delta_relative_rms"] > 0
    pair2 = engine.generate_pair("Next")
    assert pair2["clean"]["cache_reused"]
    assert pair2["fly"]["cache_reused"]
    assert len(engine.sessions["clean"].messages) == 4
    assert_no_hooks(engine)


def test_steering_off_matches_clean_and_feedback_off(engine):
    engine.configure(steering_enabled=False, feedback_enabled=False)
    pair = engine.generate_pair("Hi")
    assert pair["clean"]["token_ids"] == pair["fly"]["token_ids"]
    assert all(r["delta_relative_rms"] == 0 for r in pair["fly"]["trace"])
    assert all(not r["feedback_enabled"] for r in pair["fly"]["trace"])
    assert np.linalg.norm(engine.reservoir.state) > 0
    comparison = engine.scientific_compare("Hi", max_tokens=3)
    assert comparison["metrics"]["logit_rms_difference"] == 0
    assert_no_hooks(engine)


def test_context_is_bounded_and_controls_clear_contaminated_cache(engine):
    # An oversized latest message is rejected explicitly instead of being cut mid-template.
    with pytest.raises(ContextOverflowError):
        engine.generate_pair("long " * 200, max_new_tokens=3)
    assert engine.sessions["clean"].messages == [] and engine.sessions["fly"].messages == []
    for index in range(4):
        engine.generate_pair(f"turn {index} " + "x" * 20, max_new_tokens=3)
    assert len(engine.sessions["clean"].cached_ids) <= engine.config.max_context_tokens
    engine.configure(strength=0.0)
    assert engine.sessions["fly"].cache is None
    engine.reset_state()
    assert np.all(engine.reservoir.state == 0)
    engine.clear_chats()
    assert engine.sessions["clean"].messages == []
    assert engine.sessions["fly"].messages == []


def test_cancellation_and_exception_remove_hooks(engine, monkeypatch):
    cancel = Event()
    def callback(branch, token, text):
        if branch == "fly":
            cancel.set()
    pair = engine.generate_pair("Hi", cancel_event=cancel, on_token=callback)
    assert pair["fly"]["cancelled"] and len(pair["fly"]["token_ids"]) == 1
    assert_no_hooks(engine)
    original = engine.llm.model.forward
    def fail(*args, **kwargs):
        if engine.coupling.handles:
            raise RuntimeError("Injected software-test failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.llm.model, "forward", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        engine.generate_pair("Hi")
    assert engine.sessions["fly"].cache is None
    assert_no_hooks(engine)


def test_idle_real_decay_scientific_same_prefix_and_preserves_live_state(engine):
    engine.generate_pair("Hi")
    initial = engine.reservoir.state.copy()
    before_time = engine.reservoir.time
    snapshot = engine.idle_tick(3)
    assert snapshot["simulation_time"] > before_time
    assert not np.array_equal(initial, engine.reservoir.state)
    initial = engine.reservoir.state.copy()
    histories = {key: session.messages.copy() for key, session in engine.sessions.items()}
    comparison = engine.scientific_compare("Hi", max_tokens=3)
    assert comparison["matched_prefix"] and comparison["matched_steps"] == 3
    assert comparison["metrics"]["logit_rms_difference"] > 0
    random = engine.demo_random_topology("Hi", max_tokens=3)
    assert random["matched_prefix"]
    persistence = engine.demo_persistence("Remember", "Hi", max_tokens=3)
    assert persistence["retained_state_rms"] > 0
    assert persistence["idle_steps"] == 10
    assert persistence["before_idle_state_rms"] != persistence["retained_state_rms"]
    assert any(event["type"] == "persistence_idle" for event in engine.recording.events)
    np.testing.assert_array_equal(initial, engine.reservoir.state)
    assert histories == {key: session.messages for key, session in engine.sessions.items()}
    assert_no_hooks(engine)


def test_cuda_unavailable_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        ObservatoryEngine("software-test", tmp_path, device="cuda:0")


def test_save_restore_roundtrip_and_new_session(engine, tmp_path):
    engine.generate_pair("Hi")
    path = engine.save_recording(tmp_path / "session.json")
    replay = engine.load_recording(path)
    assert len(replay.snapshots) > 0
    assert replay.metadata["connectome_hash"] == engine.metadata["connectome_hash"]
    initial = engine.reservoir.state.copy()
    engine.new_session()
    assert not engine.sessions["fly"].messages
    engine.restore_session(path)
    np.testing.assert_array_equal(initial, engine.reservoir.state)
    assert engine.sessions["fly"].cache is None
    assert engine.sessions["fly"].messages == replay.messages["fly"]


def test_scientific_eos_preserves_long_prompt_budget(engine):
    # An early EOS must not change the second condition's reserved context budget.
    prompt = "long " * 15
    session = engine.sessions["clean"]
    plan = engine._build_context(session, prompt, reserve=3)
    assert plan.tokens == engine.config.max_context_tokens - 5
    with torch.inference_mode():
        engine.llm.tokenizer.eos_token_id = int(engine.llm.model(plan.ids).logits[0, -1].argmax())
    result = engine.scientific_compare(prompt, max_tokens=3)
    assert result["matched_steps"] == 1
    assert result["prompt_ids"] == plan.ids[0].tolist()
    assert len(result["forced_token_ids"]) == 1
    with pytest.raises(ContextOverflowError):
        engine.scientific_compare("long " * 20, max_tokens=3)
    assert_no_hooks(engine)


def test_scientific_telemetry_emits_recorded_experimental_states_and_final_live_state(engine):
    initial = engine.reservoir.state.copy()
    emitted = []
    def callback(snapshot):
        assert engine.recording.events[-1]["snapshot"] == snapshot
        emitted.append(snapshot)
    with_stream = engine.demo_random_topology("Hi", max_tokens=3, on_telemetry=callback)
    without_stream = engine.demo_random_topology("Hi", max_tokens=3)
    assert with_stream["metrics"] == without_stream["metrics"]
    assert {snapshot["branch"] for snapshot in emitted} == {"fly", "random", "fly_free", "random_free", None}
    assert any(np.linalg.norm(snapshot["state"]) > 0 for snapshot in emitted[:-1])
    assert emitted[-1]["branch"] is None
    np.testing.assert_array_equal(emitted[-1]["state"], initial)
    np.testing.assert_array_equal(engine.reservoir.state, initial)
    emitted.clear()
    engine.demo_persistence("Remember", "Hi", max_tokens=3, on_telemetry=callback)
    assert {snapshot["branch"] for snapshot in emitted} == {
        "fly_warmup", "fly_idle", "fly_retained", "fly_reset", None}
    idle = next(snapshot for snapshot in emitted if snapshot["branch"] == "fly_idle")
    assert idle["status"] == "experiment" and idle["simulation_step"] > 0
    np.testing.assert_array_equal(emitted[-1]["state"], initial)
    cancel = Event()
    emitted.clear()
    def cancelled_callback(snapshot):
        callback(snapshot)
        if snapshot["branch"] == "fly":
            cancel.set()
    result = engine.demo_random_topology("Hi", max_tokens=3, cancel_event=cancel,
                                        on_telemetry=cancelled_callback)
    assert result["cancelled"]
    assert emitted[-1]["status"] == "cancelled" and emitted[-1]["branch"] is None
    np.testing.assert_array_equal(emitted[-1]["state"], initial)
    assert_no_hooks(engine)


def test_demo_free_generations_are_separate_and_preserve_live_sessions(engine):
    engine.generate_pair("Prior conversation")
    initial = engine.reservoir.state.copy()
    initial_time = engine.reservoir.time
    histories = {branch: session.messages.copy() for branch, session in engine.sessions.items()}
    caches = {branch: session.cache for branch, session in engine.sessions.items()}
    for method, labels in ((engine.demo_clean_vs_fly, ("clean", "fly")),
                           (engine.demo_random_topology, ("fly", "random"))):
        result = method("Same prompt", max_tokens=3)
        assert result["matched_prefix"] and result["matched_steps"] == 3
        free = result["free_generation"]
        for role, label in zip(("reference", "condition"), labels):
            assert free[role]["prompt"] == "Same prompt"
            assert free[role]["condition"] == label
            assert len(free[role]["token_ids"]) == 3
            assert free[role]["text"] == engine.llm.tokenizer.decode(free[role]["token_ids"])
            assert free[role]["seconds"] > 0 and free[role]["tokens_per_second"] > 0
        assert free["reference"]["prompt_ids"] == free["condition"]["prompt_ids"]
        assert free["reference"]["initial_state_rms"] == free["condition"]["initial_state_rms"]
        assert not free["cancelled"]
        assert result["scientific_seconds"] > 0 and result["free_generation_seconds"] > 0
        np.testing.assert_array_equal(engine.reservoir.state, initial)
        assert engine.reservoir.time == initial_time
        assert histories == {branch: session.messages for branch, session in engine.sessions.items()}
        assert all(caches[branch] is engine.sessions[branch].cache for branch in caches)
        assert_no_hooks(engine)


def test_demo_free_generation_cancellation_cleans_hooks_and_restores_live_display(engine):
    initial = engine.reservoir.state.copy()
    cancel = Event()
    snapshots = []
    def callback(snapshot):
        snapshots.append(snapshot)
        if snapshot["branch"] == "fly_free" and snapshot.get("token_index") == 0:
            cancel.set()
    result = engine.demo_clean_vs_fly("Hi", max_tokens=3, cancel_event=cancel,
                                      on_telemetry=callback)
    assert result["cancelled"] and result["free_generation"]["cancelled"]
    assert len(result["free_generation"]["reference"]["token_ids"]) == 3
    assert len(result["free_generation"]["condition"]["token_ids"]) == 1
    assert result["matched_steps"] == 3
    assert snapshots[-1]["branch"] is None and snapshots[-1]["status"] == "cancelled"
    np.testing.assert_array_equal(snapshots[-1]["state"], initial)
    np.testing.assert_array_equal(engine.reservoir.state, initial)
    assert_no_hooks(engine)
