"""Live external injection, external neural cue and runtime leak rate (random tiny weights).

These check token provenance, reservoir bookkeeping and reproducibility, not language.
"""
import threading
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
import pytest

from conftest import assert_no_hooks
from liquidaion_fly.continuous import (ContinuousConfig, ContinuousRun, cue_neuron_indices, load_run,
                                       seed_token_ids)
from liquidaion_fly.reservoir import Reservoir

SEED = "Once upon a winter"
ROOT = Path(__file__).resolve().parents[1]


def config(**values):
    settings = dict(seed_text=SEED, window=8, total_tokens=30, condition="fly", checkpoints=(), probe_tokens=4,
                    snapshot_interval=8, decoding="sample")
    settings.update(values)
    return ContinuousConfig(**settings)


def spy_inputs(engine, monkeypatch):
    calls = []
    original = engine.llm.model.forward

    def forward(*args, **kwargs):
        calls.append(kwargs["input_ids"][0].tolist())
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.llm.model, "forward", forward)
    return calls


def expected_categories(run):
    """Recompute every row's category from the recorded stream and window size alone."""
    seed = [(token, None) for token in run.seed_ids]
    history = seed + run.stream
    categories = []
    for position, (_, source) in enumerate(run.stream):
        if source is not None:
            categories.append("external")
            continue
        window = history[:len(seed) + position][-run.config.window:]
        categories.append("post_injection" if any(s is not None for _, s in window) else "autonomous")
    return categories


def test_injected_text_enters_the_token_stream_exactly(engine, monkeypatch):
    calls = spy_inputs(engine, monkeypatch)
    run = ContinuousRun(engine, config(injections=((5, "Hi!"),)))
    run.run()
    injected = seed_token_ids(engine.llm.tokenizer, "Hi!")
    assert [token for token, _ in run.stream[5:5 + len(injected)]] == injected
    assert all(source == 1 for _, source in run.stream[5:5 + len(injected)])
    history = run.seed_ids + [token for token, _ in run.stream]
    for index, call in enumerate(calls):
        assert call == history[:len(run.seed_ids) + index][-8:], f"forward {index} saw the wrong window"
    record = run.injections[1]
    assert record["status"] == "inserted" and record["start_position"] == 5
    assert record["end_position"] == 5 + len(injected) and record["inserted"] == len(injected)
    assert len(run.generated) == 30, "Injected tokens do not count toward the model-token budget"


def test_each_injected_token_is_processed_once_with_no_double_processing(engine, monkeypatch):
    calls = spy_inputs(engine, monkeypatch)
    run = ContinuousRun(engine, config(injections=((5, "Hi!"),), feedback=0.25))
    summary = run.run()
    injected = len(seed_token_ids(engine.llm.tokenizer, "Hi!"))
    assert summary["forward_calls"] == 30 + injected == len(run.stream)
    assert summary["reservoir_advances"] == 2 * summary["forward_calls"]
    # Exactly one forward pass has each stream token as its newest (processed) input token.
    processed = [call[-1] for call in calls[1:]]
    assert processed == [token for token, _ in run.stream[:-1]]


def test_categories_follow_window_membership_and_eviction(engine, tmp_path):
    run = ContinuousRun(engine, config(window=8, injections=((5, "Hi!"),)), tmp_path / "r")
    run.run()
    rows = load_run(tmp_path / "r")["telemetry"]
    assert [row["category"] for row in rows] == expected_categories(run)
    sequence = [rows[0]["category"]]
    for row in rows[1:]:
        if row["category"] != sequence[-1]:
            sequence.append(row["category"])
    assert sequence == ["autonomous", "external", "post_injection", "autonomous"]
    left = run.injections[1]["left_window_position"]
    first_auto_after = next(row["position"] for row in rows if row["position"] > 5 and row["category"] == "autonomous")
    assert left == first_auto_after, "AUTONOMOUS resumes exactly when the last injected token leaves the window"
    assert {event["type"] for event in run.events} >= {"injection_started", "injection_completed", "injection_left_window"}
    states = load_run(tmp_path / "r")["states"]["events"]
    assert set(states) == {"injection1_start", "injection1_end", "injection1_left_window"}


def test_multiple_overlapping_interventions(engine):
    run = ContinuousRun(engine, config(window=16, total_tokens=50, injections=((5, "Hi!"), (11, "Yo"))))
    rows = []
    run.run(on_progress=lambda row, active: rows.append(row))
    categories = expected_categories(run)
    assert [row["category"] for row in rows] == categories
    sequence = [categories[0]]
    for category in categories[1:]:
        if category != sequence[-1]:
            sequence.append(category)
    assert sequence == ["autonomous", "external", "post_injection", "external", "post_injection", "autonomous"]
    assert [run.injections[i]["status"] for i in (1, 2)] == ["inserted", "inserted"]
    assert run.injections[1]["left_window_position"] < run.injections[2]["left_window_position"]


def test_injection_into_a_running_repetition_loop(engine):
    # A one-token window makes greedy decoding a deterministic map over 64 tokens, so it must cycle.
    run = ContinuousRun(engine, config(window=1, decoding="greedy", total_tokens=150, injections=((100, "Hi!"),)))
    run.run()
    tokens = [token for token, _ in run.stream]
    before = tokens[60:100]
    period = next(p for p in range(1, 65) if all(before[i] == before[i - p] for i in range(p, len(before))))
    assert period <= 64, "The stream is in an exact loop before the injection"
    injected = seed_token_ids(engine.llm.tokenizer, "Hi!")
    assert tokens[100:100 + len(injected)] == injected and run.injections[1]["status"] == "inserted"


def test_injection_from_another_thread_while_generating(engine):
    run = ContinuousRun(engine, config(total_tokens=60))
    reached, injected = Event(), Event()

    def progress(row, active):
        if row["step"] == 10 and not reached.is_set():
            reached.set()
            injected.wait(5)
    worker = threading.Thread(target=run.run, kwargs={"on_progress": progress})
    worker.start()
    assert reached.wait(10)
    injection_id = run.inject("Hi!")
    injected.set()
    worker.join(30)
    record = run.injections[injection_id]
    assert record["status"] == "inserted" and record["origin"] == "live"
    assert record["start_position"] >= 11
    assert_no_hooks(engine)


def test_cancellation_records_partial_and_pending_injections(engine, tmp_path):
    cancel = Event()
    run = ContinuousRun(engine, config(total_tokens=100, injections=((5, "Hi there friend"),)), tmp_path / "c")

    def progress(row, active):
        if row["source"] == "external" and active.injections[1]["inserted"] == 2:
            active.inject("Never inserted")
            cancel.set()
    summary = run.run(cancel, progress)
    statuses = {record["id"]: record["status"] for record in summary["injections"]}
    assert statuses == {1: "partial_cancelled", 2: "not_inserted_cancelled"}
    saved = load_run(tmp_path / "c")
    assert len(saved["telemetry"]) == len(saved["tokens"]["stream"]) == summary["stream_tokens"]
    assert saved["text"] == SEED + engine.llm.tokenizer.decode([token for token, _ in saved["tokens"]["stream"]])
    assert any(event["type"] == "injection_interrupted" for event in summary["events"])
    assert_no_hooks(engine)


def test_cue_off_and_zero_strength_preserve_baseline(engine):
    def run_with(**values):
        run = ContinuousRun(engine, config(injections=((5, "Hi!"),), **values))
        run.run()
        return [token for token, _ in run.stream], run.reservoir.state.copy()
    off_tokens, off_state = run_with(cue_target="off")
    zero_tokens, zero_state = run_with(cue_target="da", cue_strength=0.0)
    assert off_tokens == zero_tokens
    np.testing.assert_array_equal(off_state, zero_state)
    # Without any injection the DA cue is never applied, so the run equals the plain baseline.
    plain = ContinuousRun(engine, config())
    plain.run()
    cued = ContinuousRun(engine, config(cue_target="da", cue_strength=2.0))
    cued.run()
    assert plain.generated == cued.generated
    np.testing.assert_array_equal(plain.reservoir.state, cued.reservoir.state)


def test_da_cue_is_real_reservoir_input_applied_only_while_processing_injected_tokens(engine, tmp_path):
    off = ContinuousRun(engine, config(injections=((5, "Hi!"),), decoding="greedy"), tmp_path / "off")
    off.run()
    cue = ContinuousRun(engine, config(injections=((5, "Hi!"),), decoding="greedy", cue_target="da", cue_strength=1.0),
                        tmp_path / "da")
    cue.run()
    injected = len(seed_token_ids(engine.llm.tokenizer, "Hi!"))
    rows = load_run(tmp_path / "da")["telemetry"]
    applied = [row["position"] for row in rows if row["cue_applied"]]
    # Forwards that process injected tokens: positions 6 .. 5 + injected (the last one resumes generation).
    assert applied == list(range(6, 6 + injected))
    off_rows = load_run(tmp_path / "off")["telemetry"]
    for a, b in zip(off_rows[:6], rows[:6]):
        assert a["state_rms"] == b["state_rms"], "Identical before the first cued update"
    assert rows[6]["da_mean"] > off_rows[6]["da_mean"], "The positive cue raises the DA neuron's activity"
    assert cue.cue_indices == [1] and cue.da_indices == [1]


def test_da_neurons_are_the_sixteen_annotated_root_ids_in_graph_order():
    indices, roots = cue_neuron_indices(ROOT / "data/cx512", "da")
    nodes = pd.read_csv(ROOT / "data/cx512/nodes.csv", dtype={"root_id": str})
    assert len(indices) == 16 and roots == nodes.loc[nodes.nt_type == "DA", "root_id"].tolist()
    from liquidaion_fly.connectome import load_graph
    _, graph_ids = load_graph(ROOT / "data/cx512")
    assert [graph_ids[i] for i in indices] == roots
    control, control_roots = cue_neuron_indices(ROOT / "data/cx512", "random", seed=42)
    assert len(control) == 16 and not set(control) & set(indices)
    assert control == cue_neuron_indices(ROOT / "data/cx512", "random", seed=42)[0]


def test_reservoir_leak_change_keeps_state_and_changes_dynamics():
    from scipy import sparse
    graph = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 2], [3, 0, 0]], dtype=float))
    a, b = Reservoir(graph, features=4, seed=1), Reservoir(graph, features=4, seed=1)
    features = np.ones(4)
    a.advance(features, steps=3)
    b.advance(features, steps=3)
    fingerprint = b.fingerprint
    b.set_leak(0.05)
    np.testing.assert_array_equal(a.state, b.state)
    assert b.leak == pytest.approx(0.05) and b.fingerprint != fingerprint
    a.advance(steps=5)
    b.advance(steps=5)
    assert not np.allclose(a.state, b.state)
    assert np.linalg.norm(b.state) > np.linalg.norm(a.state), "Smaller leak: slower decay without input"
    with pytest.raises(ValueError):
        b.set_leak(0)


def test_live_leak_change_is_recorded_and_replays_from_schedule(engine, tmp_path):
    run = ContinuousRun(engine, config(total_tokens=40), tmp_path / "live")

    def progress(row, active):
        if row["step"] == 12 and row["source"] == "model":
            active.request_leak(0.05)
            active.inject("Hi!")
    summary = run.run(on_progress=progress)
    change = next(event for event in summary["events"] if event["type"] == "leak_changed")
    assert change["old"] == pytest.approx(0.2) and change["new"] == pytest.approx(0.05) and change["origin"] == "live"
    record = summary["injections"][0]
    # Replaying the recorded positions reproduces the stream and the reservoir exactly.
    replay = ContinuousRun(engine, config(total_tokens=40, leak_schedule=((change["position"], 0.05),),
                                          injections=((record["start_position"], record["text"]),)))
    replay.run()
    assert replay.stream == [(t, 1 if s is not None else None) for t, s in run.stream]
    np.testing.assert_array_equal(replay.reservoir.state, run.reservoir.state)
    rows = load_run(tmp_path / "live")["telemetry"]
    assert {row["leak"] for row in rows if row["position"] < change["position"]} == {0.2}
    assert all(abs(row["leak"] - 0.05) < 1e-12 for row in rows if row["position"] >= change["position"])


def test_probes_with_injection_do_not_mutate_and_match_main_logits(engine):
    plain = ContinuousRun(engine, config(injections=((5, "Hi!"),), total_tokens=30))
    plain.run()
    probed = ContinuousRun(engine, config(injections=((5, "Hi!"),), total_tokens=30, checkpoints=(4, 8),
                                          cue_target="da", cue_strength=0.0))
    summary = probed.run()
    assert probed.stream == plain.stream
    np.testing.assert_array_equal(probed.reservoir.state, plain.reservoir.state)
    assert all(probe["retained_first_logits_equal_main"] for probe in summary["probes"])


def test_engine_leak_rate_is_configurable_recorded_and_restored(engine, tmp_path):
    engine.generate_pair("Hi")
    state = engine.reservoir.state.copy()
    baseline_fingerprint = engine.reservoir.fingerprint
    engine.configure(leak_rate=0.05)
    np.testing.assert_array_equal(engine.reservoir.state, state)
    assert engine.reservoir.leak == pytest.approx(0.05) and engine.reservoir.fingerprint != baseline_fingerprint
    event = next(e for e in engine.recording.events if e["type"] == "leak_changed")
    assert event["old"] == pytest.approx(0.2) and event["new"] == pytest.approx(0.05)
    path = engine.save_recording(tmp_path / "session.json")
    engine.configure(leak_rate=0.2)
    engine.restore_session(path)
    assert engine.reservoir.leak == pytest.approx(0.05) and engine.config.leak_rate == pytest.approx(0.05)
    # A recording made before leak control (no leak_rate) restores with the 0.2 baseline.
    import json
    payload = json.loads(path.read_text())
    payload["settings"].pop("leak_rate")
    payload["state"]["fingerprint"] = baseline_fingerprint
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload))
    engine.restore_session(legacy)
    assert engine.reservoir.leak == pytest.approx(0.2)
    # A state recorded under other dynamics is rejected rather than silently restored.
    payload["settings"]["leak_rate"] = 0.05
    legacy.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dynamics"):
        engine.restore_session(legacy)


def test_no_role_markers_or_hidden_text_are_added_to_injections(qwen_engine, monkeypatch):
    calls = spy_inputs(qwen_engine, monkeypatch)
    text = "Из-за деревьев неожиданно появился огромный зелёный мамонт."
    run = ContinuousRun(qwen_engine, ContinuousConfig(seed_text="Однажды в студёную зимнюю пору", window=64,
                                                       total_tokens=6, checkpoints=(), injections=((3, text),)))
    run.run()
    tokenizer = qwen_engine.llm.tokenizer
    injected = [token for token, source in run.stream if source is not None]
    assert injected == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(injected) == text
    special = set(tokenizer.all_special_ids)
    assert not any(set(call) & special for call in calls), "No special/role tokens anywhere in the model input"
