"""Headless lifecycle and signal checks; fixtures never enter user anatomy caches."""
from __future__ import annotations

import os
import hashlib
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
