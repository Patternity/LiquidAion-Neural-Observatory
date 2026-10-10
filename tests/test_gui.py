"""Headless lifecycle and signal checks; fixtures never enter user anatomy caches."""
from __future__ import annotations

import os
import hashlib
import json
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shiboken6
from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QApplication, QWidget, QTextEdit

from liquidaion_fly.bypass import active_bypasses
from liquidaion_fly.observatory_gui import ObservatoryWindow, EngineWorker
from liquidaion_fly.observatory_engine import ObservatoryConfig
from liquidaion_fly.recording import SessionRecording


class FakeBrain(QWidget):
    neuron_selected = Signal(int)

    def __init__(self, anatomy, graph):
        super().__init__()
        self.updates = []
        self.selected = None
        self.setMinimumHeight(120)

    def update_state(self, values):
        self.updates.append(list(values))

    def select_neuron(self, index):
        self.selected = index

    def set_connections_visible(self, visible):
        self.connections = visible

    def set_anatomy_visible(self, visible):
        self.anatomy_visible = visible

    def reset_camera(self):
        pass

    def focus_neurons(self):
        pass

    def export_image(self, path):
        self.grab().save(str(path))


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()
    for widget in application.topLevelWidgets():
        widget.close()
        shiboken6.delete(widget)
    application.processEvents()


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    directory = Path(__file__).resolve().parents[1] / "data" / "cx512"
    ids = pd.read_csv(directory / "nodes.csv", dtype={"root_id": str}).root_id.tolist()
    anatomy = SimpleNamespace(neuron_ids=ids, positions=np.zeros((len(ids), 3)))
    fake_module = types.ModuleType("liquidaion_fly.anatomy")
    fake_module.Anatomy = SimpleNamespace(load=lambda path: anatomy)
    monkeypatch.setitem(sys.modules, "liquidaion_fly.anatomy", fake_module)
    widget = ObservatoryWindow(ObservatoryConfig(), "/unused/model", directory, "/unused/anatomy",
        start_worker=False, brain_factory=FakeBrain,
        preferences=QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    widget.on_initialized({"mode": "live"})
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    shiboken6.delete(widget)
    app.processEvents()


def test_shared_prompt_emits_one_paired_command_and_streams_separately(window):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    window.prompt.setText("Give two words.")
    window.send_prompt()
    assert calls[-1][0] == "generate"
    assert calls[-1][1]["prompt"] == "Give two words."
    assert not window.send.isEnabled()
    window.on_token("clean", "calm sea")
    window.on_token("fly", "quiet shore")
    assert window.clean_chat.pending == "calm sea"
    assert window.fly_chat.pending == "quiet shore"
    window.on_finished("generate", {})
    assert window.clean_chat.history[-1] == ("assistant", "calm sea")
    assert window.fly_chat.history[-1] == ("assistant", "quiet shore")
    assert window.send.isEnabled()


def test_view_pause_retains_new_measurement_without_animating(window):
    state = [.2] * len(window.node_ids)
    window.on_telemetry({"state": state, "state_rms": .2, "simulation_time": 1})
    assert len(window.brain.updates) == 1
    window.toggle_visualization()
    window.on_telemetry({"state": [.3] * len(state), "state_rms": .3, "simulation_time": 2})
    assert len(window.brain.updates) == 1
    assert window.latest["simulation_time"] == 2
    window.toggle_visualization()
    assert window.brain.updates[-1][0] == .3


def test_neuron_selection_uses_real_ids_and_activity(window):
    window.on_telemetry({"state": [.4] * len(window.node_ids), "state_rms": .4})
    window.select_neuron(10)
    assert window.node_ids[10] in window.neuron_info.text()
    assert "0.400000" in window.neuron_info.text()
    assert window.brain.selected == 10


def test_export_contains_dashboard_geometry_and_measured_snapshot(window, tmp_path):
    window.on_telemetry({"state": [.1] * len(window.node_ids), "state_rms": .1})
    window.export_dashboard(tmp_path)
    assert (tmp_path / "dashboard.png").stat().st_size > 0
    assert (tmp_path / "anatomy-state.png").stat().st_size > 0
    assert '"state_rms": 0.1' in (tmp_path / "telemetry.json").read_text()
    assert "not a biophysical simulation" in (tmp_path / "summary.md").read_text()


def test_offline_worker_loads_recording_without_constructing_model(app, monkeypatch, tmp_path):
    graph_dir = Path(__file__).resolve().parents[1] / "data" / "cx512"
    metadata = {"model": "fixture", "connectome_hash": hashlib.sha256((graph_dir / "edges.csv").read_bytes()).hexdigest(),
        "node_hash": hashlib.sha256((graph_dir / "nodes.csv").read_bytes()).hexdigest()}
    recording = SessionRecording(metadata)
    recording.append("telemetry", snapshot={"state": [.25] * 512, "state_rms": .25})
    path = recording.save(tmp_path / "replay.json", {"clean": [], "fly": []}, {}, {})
    import liquidaion_fly.observatory_engine as engine_module
    monkeypatch.setattr(engine_module, "ObservatoryEngine", lambda *args, **kwargs: pytest.fail("Offline replay loaded a model"))
    worker = EngineWorker(ObservatoryConfig(), "/missing/model", graph_dir, replay_path=path)
    initialized, snapshots = [], []
    worker.initialized.connect(initialized.append)
    worker.telemetry.connect(snapshots.append)
    worker.initialize()
    assert initialized[0]["mode"] == "replay"
    assert snapshots[0]["state_rms"] == .25
    assert worker.engine is None
    worker.shutdown()
    shiboken6.delete(worker)


def test_worker_idle_uses_actual_engine_state(app):
    worker = EngineWorker(ObservatoryConfig(), "unused", "unused")
    calls = []
    worker.engine = SimpleNamespace(idle_tick=lambda: calls.append(True) or {"state_rms": .125})
    snapshots = []
    worker.telemetry.connect(snapshots.append)
    worker.idle()
    assert calls == [True]
    assert snapshots == [{"state_rms": .125}]
    worker.busy = True
    worker.idle()
    assert len(calls) == 1


def test_science_summary_and_three_measured_plots(window, tmp_path):
    window.on_telemetry({"state": [.1] * len(window.node_ids), "state_rms": .1,
        "delta_relative_rms": .02, "steering_vector_norm": 3.5})
    assert window.rms_plot.values[-1] == .1
    assert window.delta_plot.values[-1] == .02
    assert window.steering_plot.values[-1] == 3.5
    result = {"comparison": "clean_vs_fly", "matched_prefix": True, "matched_steps": 2,
        "metrics": {"mean_kl_baseline_to_condition": .001, "logit_rms_difference": .02,
            "argmax_disagreement_fraction": .5},
        "condition_trace": [{"delta_relative_rms": .02, "before_rms": 2., "after_rms": 2.1,
            "features": [0.] * 64}]}
    window.on_finished("scientific", result)
    concise = window.experiment_result.toPlainText()
    assert "Mean KL: 0.001" in concise
    assert "hidden perturbation" in concise
    assert "features" not in concise
    window.science_details.setChecked(True)
    assert '"features"' in window.experiment_result.toPlainText()
    window.science_details.setChecked(False)
    window.export_dashboard(tmp_path)
    assert '"features"' in (tmp_path / "science-result.json").read_text()


def test_replay_rejects_wrong_connectome_before_painting_state(app, tmp_path):
    graph_dir = Path(__file__).resolve().parents[1] / "data" / "cx512"
    recording = SessionRecording({"connectome_hash": "wrong", "node_hash": "wrong"})
    recording.append("telemetry", snapshot={"state": [.25] * 512, "state_rms": .25})
    path = recording.save(tmp_path / "mismatch.json", {"clean": [], "fly": []}, {}, {})
    worker = EngineWorker(ObservatoryConfig(), "/unused/model", graph_dir, replay_path=path)
    errors, snapshots = [], []
    worker.error.connect(errors.append)
    worker.telemetry.connect(snapshots.append)
    worker.initialize()
    assert "neuron identity cannot be verified" in errors[0]
    assert snapshots == []
    assert worker.replay is None
    shiboken6.delete(worker)


def test_replay_excludes_demo_free_branches_and_shows_recorded_result(app, tmp_path):
    graph_dir = Path(__file__).resolve().parents[1] / "data" / "cx512"
    metadata = {"connectome_hash": hashlib.sha256((graph_dir / "edges.csv").read_bytes()).hexdigest(),
        "node_hash": hashlib.sha256((graph_dir / "nodes.csv").read_bytes()).hexdigest()}
    recording = SessionRecording(metadata)
    recording.append("initialized", snapshot={"state": [0.] * 512})
    recording.append("message_pair", prompt="Hello", responses={"clean": "Hello!", "fly": "Hello!"})
    recording.append("scientific_comparison", result={"comparison": "clean_vs_fly", "matched_prefix": True,
        "matched_steps": 2, "metrics": {"logit_rms_difference": .02}})
    recording.append("generated_token", branch="clean_free", scientific=False, text="Demo only")
    recording.append("generated_token", branch="fly_free", scientific=False, text="Demo only")
    recording.append("generated_token", branch="random_free", scientific=False, text="Demo only")
    recording.append("demo_free_generation", comparison="clean_vs_fly", result={"reference": {"text": "Demo only"}})
    recording.append("finished", snapshot={"state": [.1] * 512})
    messages = {branch: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello!"}]
        for branch in ("clean", "fly")}
    path = recording.save(tmp_path / "free.json", messages, {}, {})
    worker = EngineWorker(ObservatoryConfig(), "/unused/model", graph_dir, replay_path=path)
    chats, results = [], []
    worker.replay_chat.connect(chats.append)
    worker.replay_result.connect(results.append)
    worker._load_replay(path)
    worker._seek_replay(1)
    assert chats[-1] == messages
    assert set(chats[-1]) == {"clean", "fly"}
    assert results[-1]["matched_steps"] == 2
    assert results[-1]["free_generation"]["reference"]["text"] == "Demo only"
    shiboken6.delete(worker)


def test_actual_recording_all_frames_replay_without_changing_final_chats(app):
    graph_dir = Path(__file__).resolve().parents[1] / "data" / "cx512"
    path = graph_dir.parents[1] / "results" / "observatory" / "session.json"
    if not path.exists():
        pytest.skip("Actual pretrained acceptance recording is not present")
    worker = EngineWorker(ObservatoryConfig(), "/unused/model", graph_dir, replay_path=path)
    chats, results = [], []
    worker.replay_chat.connect(chats.append)
    worker.replay_result.connect(results.append)
    worker._load_replay(path)
    for index in range(len(worker.replay.snapshots)):
        worker._seek_replay(index)
    assert chats[-1] == worker.replay.messages
    assert any(result for result in results)
    shiboken6.delete(worker)


def test_chat_keeps_first_characters_visible_after_long_token(window, app):
    panel = window.clean_chat
    panel.user("x" * 1000)
    panel.stream("Hello, world!")
    app.processEvents()
    assert panel.text.toPlainText().endswith("Hello, world!")
    assert panel.text.horizontalScrollBar().value() == 0
    assert panel.text.lineWrapMode() == QTextEdit.WidgetWidth


def test_generation_limits_distinguish_new_tokens_from_context_and_persist(window, app):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    assert window.max_tokens.value() == 512
    assert window.context_tokens.value() == 4096
    assert window.max_tokens.maximum() >= 1024
    window.max_tokens.setValue(1024)
    assert calls[-1] == ("configure", {"max_new_tokens": 1024, "max_context_tokens": 4096})
    assert int(window.preferences.value("generation/max_new_tokens")) == 1024
    # Shrinking the context also caps the new-token limit so a prompt still fits.
    window.context_tokens.setValue(1024)
    assert window.max_tokens.value() <= 1024 - 64
    assert int(window.preferences.value("generation/max_context_tokens")) == 1024
    window.prompt.setText("Hi")
    window.send_prompt()
    assert calls[-1][0] == "generate" and calls[-1][1]["tokens"] == window.max_tokens.value()


def test_persisted_limits_apply_at_next_launch(app, tmp_path, monkeypatch):
    from liquidaion_fly.observatory_gui import apply_preferences
    preferences = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    preferences.setValue("generation/max_new_tokens", 1024)
    preferences.setValue("generation/max_context_tokens", 8192)
    config = apply_preferences(ObservatoryConfig(), preferences)
    assert (config.max_new_tokens, config.max_context_tokens) == (1024, 8192)
    preferences.setValue("generation/max_new_tokens", 9000)
    assert apply_preferences(ObservatoryConfig(), preferences) == ObservatoryConfig()


def test_truncated_and_cancelled_answers_are_marked(window):
    window.prompt.setText("Tell a long story.")
    window.send_prompt()
    window.on_token("clean", "Once upon")
    window.on_finished("generate", {
        "clean": {"stop_reason": "max_new_tokens", "max_new_tokens": 512, "token_ids": [1] * 512, "seconds": 1.0},
        "fly": {"stop_reason": "cancelled_before_start", "token_ids": [], "seconds": 0.0}})
    assert window.clean_chat.history[-1] == ("assistant", "Once upon")
    assert "TRUNCATED" in window.clean_chat.text.toPlainText()
    assert "Truncated" in window.clean_chat.stats.text()
    assert window.fly_chat.history[-1] == ("assistant", "")
    assert "NOT GENERATED" in window.fly_chat.text.toPlainText()
    window.on_telemetry({"branch": "fly", "prompt_tokens": 3900, "max_new_tokens": 512, "context_limit": 4096,
                         "history_dropped_messages": 4, "context_near_limit": True})
    assert "CONTEXT NEARLY FULL" in window.context_info.text()
    assert "4 messages" in window.context_info.text()


def test_published_recording_replays_final_chats_with_stop_metadata(app):
    graph_dir = Path(__file__).resolve().parents[1] / "data" / "cx512"
    path = graph_dir.parents[1] / "publication" / "hero" / "session.json"
    if not path.exists():
        pytest.skip("Published hero recording is not present")
    worker = EngineWorker(ObservatoryConfig(), "/unused/model", graph_dir, replay_path=path)
    chats = []
    worker.replay_chat.connect(chats.append)
    worker._load_replay(path)
    worker._seek_replay(len(worker.replay.snapshots) - 1)
    for branch in ("clean", "fly"):
        recorded = worker.replay.messages[branch]
        assert any("stop_reason" in message for message in recorded)
        assert [(m["role"], m["content"]) for m in chats[-1][branch]] == [(m["role"], m["content"]) for m in recorded]
    shiboken6.delete(worker)


def test_continuous_dialog_emits_run_and_shows_progress(window, tmp_path):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    window.open_continuous()
    dialog = window.continuous_dialog
    dialog.output_root = tmp_path
    dialog.window_size.setValue(64)
    dialog.total.setValue(256)
    dialog.condition.setCurrentIndex(dialog.condition.findData("fly_reset"))
    dialog.reset_interval.setValue(128)
    dialog.request_start()
    command, payload = calls[-1]
    assert command == "continuous"
    assert payload["window"] == 64 and payload["total_tokens"] == 256 and payload["reset_interval"] == 128
    assert payload["condition"] == "fly_reset" and payload["output"].startswith(str(tmp_path))
    assert (payload["decoding"], payload["temperature"], payload["top_p"], payload["sampling_seed"]) == ("sample", 0.8, 0.9, 42)
    assert not window.continuous_button.isEnabled() and dialog.stop.isEnabled()
    dialog.on_progress({"step": 9, "total": 256, "tokens_per_second": 30.0, "window_len": 64, "state_rms": 0.12,
                        "text": " далее", "rms": [(i, 0.1) for i in range(10)], "resets": []})
    assert dialog.text.toPlainText().endswith(" далее")
    assert len(dialog.plot.values) == 10 and "token 10 / 256" in dialog.progress.text()
    window.on_finished("continuous", {"stop_reason": "cancelled", "generated_tokens": 10, "tokens_per_second": 30.0,
                                      "peak_gpu_allocated_bytes": None})
    assert window.continuous_button.isEnabled() and not dialog.running
    assert "cancelled" in dialog.progress.text()


def test_worker_runs_continuous_condition_and_streams(app, monkeypatch, tmp_path):
    from conftest import build_engine
    engine = build_engine(monkeypatch, tmp_path)
    worker = EngineWorker(ObservatoryConfig(), "unused", "unused")
    worker.engine = engine
    updates, snapshots, finished = [], [], []
    worker.continuous_progress.connect(updates.append)
    worker.telemetry.connect(snapshots.append)
    worker.finished.connect(lambda command, result: finished.append((command, result)))
    worker.execute("continuous", {"seed_text": "Once", "window": 8, "total_tokens": 20, "condition": "fly",
                                  "reset_interval": None, "strength": 0.25, "feedback": 0.25, "eos_policy": "stop",
                                  "prompt_policy": "raw", "output": str(tmp_path / "run")})
    command, summary = finished[-1]
    assert command == "continuous" and summary["generated_tokens"] == 20
    streamed = "".join(text for update in updates for text, _ in update["segments"])
    assert streamed == engine.llm.tokenizer.decode(
        json.loads((tmp_path / "run" / "token_ids.json").read_text())["generated"])
    assert {category for update in updates for _, category in update["segments"]} == {"autonomous"}
    assert any(snapshot.get("branch") == "continuous" and len(snapshot["state"]) == 3 for snapshot in snapshots)
    assert (tmp_path / "run" / "telemetry.jsonl").exists()
    shiboken6.delete(worker)


def test_classic_leak_control_configures_the_engine(window):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    assert window.leak.value() == 0.2
    window.leak.spin.setValue(0.05)
    assert calls[-1] == ("configure", {"leak_rate": 0.05})
    window.leak.slider.setValue(window.leak.to_slider(0.5))
    assert calls[-1] == ("configure", {"leak_rate": 0.5})
    window.on_finished("restore_session", {"settings": {"leak_rate": 0.01, "max_context_tokens": 4096,
                                                        "max_new_tokens": 512}})
    assert window.leak.value() == 0.01


def test_continuous_dialog_injection_cue_and_leak_controls(window, tmp_path):
    calls, injected, leaks = [], [], []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    window.open_continuous()
    dialog = window.continuous_dialog
    dialog.output_root = tmp_path
    dialog.inject_requested.disconnect()
    dialog.leak_requested.disconnect()
    dialog.inject_requested.connect(injected.append)
    dialog.leak_requested.connect(leaks.append)
    dialog.cue.setCurrentIndex(dialog.cue.findData("da"))
    assert dialog.cue_info.text().startswith("16 neurons selected")
    assert not dialog.inject_button.isEnabled()
    dialog.request_start()
    payload = calls[-1][1]
    assert payload["cue_target"] == "da" and payload["cue_strength"] == 0.5 and payload["leak_rate"] == 0.2
    assert dialog.inject_button.isEnabled()
    dialog.inject_text.setText(" Из-за деревьев появился мамонт.")
    dialog.request_inject()
    assert injected == [" Из-за деревьев появился мамонт."], "Injected text is sent exactly as typed"
    dialog.on_injection_queued(1, injected[0])
    assert "#1 pending" in dialog.inject_state.text()
    dialog.leak.spin.setValue(0.02)
    assert leaks == [0.02]
    dialog.on_progress({"step": 5, "position": 9, "total": 256, "tokens_per_second": 30.0, "window_len": 64,
                        "state_rms": 0.1, "da_abs_mean": 0.05, "category": "post_injection", "leak": 0.02,
                        "segments": [(" автономно", "autonomous"), (" мамонт", "external"), (" дальше", "post_injection")],
                        "rms": [(i, 0.1, 0.05, "external" if i == 3 else "autonomous") for i in range(10)],
                        "injections": [{"id": 1, "status": "inserted", "inserted": 7, "left_window_position": None}]})
    html = dialog.text.toHtml()
    assert "#f0a35e" in html and "#6fa8ff" in html and "#c9d3e3" in html
    assert dialog.text.toPlainText().endswith(" автономно мамонт дальше")
    assert "#1 inserted · in window" in dialog.inject_state.text() and "POST-INJECTION" in dialog.progress.text()
    assert dialog.plot.external == [3]


def test_dialog_text_matches_exported_run_text(window, monkeypatch, tmp_path):
    from conftest import build_engine
    from liquidaion_fly.continuous import load_run
    engine = build_engine(monkeypatch, tmp_path / "engine")
    worker = EngineWorker(ObservatoryConfig(), "unused", "unused")
    worker.engine = engine
    window.open_continuous()
    dialog = window.continuous_dialog
    worker.continuous_progress.connect(dialog.on_progress)
    payload = {"seed_text": "Once", "window": 8, "total_tokens": 40, "condition": "fly", "reset_interval": None,
               "strength": 0.25, "feedback": 0.25, "eos_policy": "stop", "prompt_policy": "raw", "decoding": "sample",
               "temperature": 0.8, "top_p": 0.9, "sampling_seed": 42, "leak_rate": 0.2, "cue_target": "da",
               "cue_strength": 0.5, "injections": ((10, "Hi!"),), "output": str(tmp_path / "run")}
    dialog.text.clear()
    dialog._append("Once", "autonomous")
    worker.execute("continuous", payload)
    saved = load_run(tmp_path / "run")
    assert dialog.text.toPlainText() == saved["text"], (repr(dialog.text.toPlainText()[-60:]), repr(saved["text"][-60:]))
    shiboken6.delete(worker)


def model_info(profile="qwen3-0.6b", layers=28, hidden=1024, capture=6, steer=14, kind="instruct"):
    return {"profile": profile, "display": profile, "kind": kind, "interaction": "chat" if kind == "instruct" else "raw",
            "model_id": "x", "revision": "c" * 40, "dtype": "float16", "quantization": None, "attention": "eager",
            "device": "cuda", "num_hidden_layers": layers, "hidden_size": hidden, "capture_layer": capture,
            "steer_layer": steer, "features": 64, "reservoir_neurons": 512, "projection_fingerprint": "ab" * 32}


def test_model_coupling_ranges_follow_layer_count_and_allow_any_order(window):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    panel = window.model_panel
    window.show_model(model_info())
    assert len(panel.rows) == 1, "A single bypass row is the default"
    assert (panel.capture.minimum(), panel.capture.maximum()) == (0, 27)
    assert (panel.steer.minimum(), panel.steer.maximum()) == (0, 27)
    assert "0…27" in panel.layer_range.text() and "hidden size:   1024" in panel.dimensions.text()
    assert "Legacy single bypass" in panel.order.text() and "capture → steering" in panel.order.text()
    panel.capture.setValue(20)
    assert panel.steer.value() == 14, "Steering is never moved by the capture control"
    assert calls[-1] == ("configure", {"bypasses": [[20, 14, True]]})
    assert panel.state.text().startswith("Pending") and "steering first" in panel.order.text()
    window.on_finished("configure", {"capture_layer": 20, "steer_layer": 14, "bypasses": [[20, 14, True]]})
    assert panel.state.text() == "Applied: 20 → 14"
    assert int(window.preferences.value("model/qwen3-0.6b/capture_layer")) == 20
    panel.steer.setValue(20)
    assert calls[-1] == ("configure", {"bypasses": [[20, 20, True]]}) and "same layer" in panel.order.text()
    window.show_model(model_info("qwen3-4b-base", 36, 2560, 8, 18, "base"))
    assert panel.capture.maximum() == 35 and panel.steer.maximum() == 35 and "0…35" in panel.layer_range.text()
    assert "Raw continuation" in panel.interaction.text() and "Raw text" in window.prompt.placeholderText()
    window._set_busy(True)
    assert not panel.capture.isEnabled() and not panel.load.isEnabled(), "No layer or model changes mid-generation"
    window._set_busy(False)


def test_bypass_rows_add_edit_remove_update_the_engine(window):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    panel = window.model_panel
    window.show_model(model_info())
    assert not panel.rows[0].remove.isEnabled(), "The last row cannot be removed"
    panel.add.click()
    assert len(panel.rows) == 2 and calls[-1] == ("configure", {"bypasses": [[6, 14, True], [6, 14, False]]})
    second = panel.rows[1]
    second.capture.setValue(20)
    second.steer.setValue(5)
    second.enabled.setChecked(True)
    assert calls[-1] == ("configure", {"bypasses": [[6, 14, True], [20, 5, True]]})
    assert "Active: 2 bypasses" in panel.order.text() and "Synchronous Multi-Bypass" in panel.order.text()
    assert second.direction.text() == "backward"
    window.on_finished("configure", {"capture_layer": 6, "steer_layer": 14, "bypasses": [[6, 14, True], [20, 5, True]]})
    assert panel.state.text() == "Applied: 6 → 14, 20 → 5"
    assert json.loads(window.preferences.value("model/qwen3-0.6b/bypasses")) == [[6, 14, True], [20, 5, True]]
    panel.add.click()
    panel.rows[2].enabled.setChecked(True)
    assert "1 duplicate row applied once" in panel.order.text(), "Identical rows are flagged, not double-applied"
    for row in list(panel.rows):
        row.enabled.setChecked(False)
    assert [row.enabled.isChecked() for row in panel.rows].count(True) == 1, "The last enabled row stays enabled"
    assert active_bypasses(panel.value()) == active_bypasses(calls[-1][1]["bypasses"]), "Panel shows what runs"
    assert "must stay enabled" in panel.state.text()
    sent = len(calls)
    panel.rows[0].enabled.setChecked(True)
    for row in panel.rows[1:]:
        row.enabled.setChecked(False)
    panel.rows[2].remove.click()
    assert len(panel.rows) == 2 and calls[-1] == ("configure", {"bypasses": [[6, 14, True], [20, 5, False]]})
    assert len(calls) > sent
    window._set_busy(True)
    assert not panel.add.isEnabled() and not panel.rows[0].capture.isEnabled() and not panel.rows[1].remove.isEnabled()
    window._set_busy(False)
    assert panel.add.isEnabled() and panel.rows[1].remove.isEnabled()


def test_last_enabled_bypass_cannot_be_unchecked(window):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    panel = window.model_panel
    window.show_model(model_info("qwen3-4b-base", 36, 2560, 8, 18, "base"))
    panel.rows[0].enabled.setChecked(False)
    assert panel.rows[0].enabled.isChecked() and not calls, "Nothing changes: the engine keeps running 8 -> 18"
    assert "must stay enabled" in panel.state.text() and "Active: 1 bypass" in panel.order.text()


def test_model_switch_requires_confirmation_and_persists_choice(window):
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    window.show_model(model_info())
    window.confirm_model_switch = lambda display: False
    window.model_panel.combo.setCurrentIndex(window.model_panel.combo.findData("qwen3-1.7b-base"))
    window.model_panel.load.click()
    assert not any(command == "switch_model" for command, _ in calls)
    assert window.model_panel.combo.currentData() == "qwen3-0.6b", "Declined switch restores the selector"
    window.confirm_model_switch = lambda display: True
    window.model_panel.combo.setCurrentIndex(window.model_panel.combo.findData("qwen3-1.7b-base"))
    window.model_panel.load.click()
    assert calls[-1] == ("switch_model", {"profile": "qwen3-1.7b-base", "bypasses": [[6, 14, True]]})
    assert window.preferences.value("model/profile") == "qwen3-1.7b-base"
    assert not window.send.isEnabled()
    window.open_continuous()
    window.on_finished("switch_model", model_info("qwen3-1.7b-base", 28, 2048, 6, 14, "base"))
    assert window.send.isEnabled() and window.model_panel.combo.currentData() == "qwen3-1.7b-base"
    dialog = window.continuous_dialog
    assert "qwen3-1.7b-base" in dialog.model_label.text() and "capture 6 → steering 14" in dialog.model_label.text()
    assert not dialog.prompt_policy.model().item(dialog.prompt_policy.findData("chat_prefix")).isEnabled()
    payload = dialog.payload()
    assert "qwen3-1.7b-base-c6-s14" in payload["output"], "Runs of different substrates never collide"


def test_saved_model_preference_applies_at_launch(app, tmp_path):
    from liquidaion_fly.observatory_gui import apply_preferences
    preferences = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    assert apply_preferences(ObservatoryConfig(), preferences).model_profile == "qwen3-0.6b"
    preferences.setValue("model/profile", "qwen3-4b-base")
    config = apply_preferences(ObservatoryConfig(), preferences)
    assert (config.model_profile, config.capture_layer, config.steer_layer) == ("qwen3-4b-base", 8, 18)
    preferences.setValue("model/qwen3-4b-base/capture_layer", 10)
    preferences.setValue("model/qwen3-4b-base/steer_layer", 30)
    config = apply_preferences(ObservatoryConfig(), preferences)
    assert (config.capture_layer, config.steer_layer) == (10, 30)
    preferences.setValue("model/qwen3-4b-base/steer_layer", 4)
    assert (apply_preferences(ObservatoryConfig(), preferences).steer_layer) == 4, "Reversed pairs are kept"
    preferences.setValue("model/profile", "unknown-model")
    assert apply_preferences(ObservatoryConfig(), preferences).model_profile == "qwen3-0.6b"


def test_plugin_runs_on_the_worker_and_failures_restore_the_controls(window, app, monkeypatch, tmp_path):
    from conftest import build_engine
    engine = build_engine(monkeypatch, tmp_path)
    worker = EngineWorker(ObservatoryConfig(), "unused", "unused")
    worker.engine = engine
    lines, finished = [], []
    worker.plugin_log.connect(lines.append)
    worker.finished.connect(lambda command, result: finished.append((command, result)))
    path = tmp_path / "tiny.py"
    path.write_text("def run(api):\n    api.log(api.generate('Hi', max_tokens=1).text)\n")
    worker.execute("plugin", {"path": str(path), "output_root": str(tmp_path / "out")})
    assert finished[-1][0] == "plugin" and finished[-1][1]["status"] == "finished" and not worker.busy
    assert any(line.startswith("Running tiny.py") for line in lines) and len(lines) >= 3
    calls = []
    window.command.connect(lambda command, payload: calls.append((command, payload)))
    window.show_model(model_info())
    window.run_plugin(str(path), output_root=tmp_path / "out")
    assert calls[-1] == ("plugin", {"path": str(path), "output_root": str(tmp_path / "out")})
    assert not window.send.isEnabled() and not window.plugin_run.isEnabled() and window.plugin_stop.isEnabled()
    assert not window.model_panel.add.isEnabled(), "No coupling changes while a script owns the engine"
    window.on_plugin_log("line from the script")
    window.plugin_stop.click()
    window.on_finished("plugin", {"status": "error", "error": "tiny.py failed: RuntimeError: x", "model": model_info(),
                                  "sessions_touched": True})
    assert window.send.isEnabled() and window.plugin_run.isEnabled() and not window.plugin_stop.isEnabled()
    assert "ERROR" in window.plugin_status.text() and "line from the script" in window.plugin_log.toPlainText()
    assert "tiny.py failed" in window.experiment_result.toPlainText()
    shiboken6.delete(worker)


def test_session_file_name_carries_a_timestamp(window, monkeypatch):
    from datetime import datetime
    from liquidaion_fly.observatory_gui import session_filename
    assert session_filename(datetime(2026, 10, 10, 14, 25, 30)) == "liquidaion-session-20261010-142530.json"
    offered = []
    monkeypatch.setattr("liquidaion_fly.observatory_gui.QFileDialog.getSaveFileName",
                        lambda parent, title, name, filters: (offered.append(name), ("", ""))[1])
    window.save_session()
    assert offered and offered[0].startswith("liquidaion-session-2") and offered[0].endswith(".json")
