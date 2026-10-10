"""Qt UI plugin system: docks, events across threads, lifecycle, layout, failures, end-to-end plugin."""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, QThread, Signal
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from conftest import build_engine
from liquidaion_fly.observatory_engine import ObservatoryConfig
from liquidaion_fly.observatory_gui import EngineWorker, ObservatoryWindow
from liquidaion_fly.plugin_api import Parameter, PluginError, build_registry
from liquidaion_fly.ui_plugins import ParameterForm

EXAMPLES = ["example_normalized_interface", "example_telemetry_panel", "example_research_interface"]
GRAPH = Path(__file__).resolve().parents[1] / "data" / "cx512"


class FakeBrain(QWidget):
    neuron_selected = Signal(int)

    def __init__(self, anatomy, graph):
        super().__init__()

    def __getattr__(self, name):
        if name in {"update_state", "select_neuron", "set_connections_visible", "set_anatomy_visible",
                    "reset_camera", "focus_neurons", "set_activity_scale", "export_image"}:
            return lambda *args: None
        raise AttributeError(name)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def make_window(app, monkeypatch, tmp_path):
    ids = pd.read_csv(GRAPH / "nodes.csv", dtype={"root_id": str}).root_id.tolist()
    anatomy = SimpleNamespace(neuron_ids=ids, positions=np.zeros((len(ids), 3)))
    fake_module = types.ModuleType("liquidaion_fly.anatomy")
    fake_module.Anatomy = SimpleNamespace(load=lambda path: anatomy)
    monkeypatch.setitem(sys.modules, "liquidaion_fly.anatomy", fake_module)
    windows = []

    def make(registry=None, prefs="preferences.ini"):
        window = ObservatoryWindow(ObservatoryConfig(), "/unused/model", GRAPH, "/unused/anatomy",
                                   start_worker=False, brain_factory=FakeBrain, registry=registry,
                                   preferences=QSettings(str(tmp_path / prefs), QSettings.IniFormat))
        window.on_initialized({"mode": "live"})
        window.show()
        app.processEvents()
        windows.append(window)
        return window
    yield make
    for window in windows:
        if shiboken6.isValid(window):
            window.close()
            shiboken6.delete(window)


def flush(app):
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_builtin_panels_are_registered_docks_with_stable_ids(make_window):
    window = make_window()
    builtin = [spec.component_id for spec in window.registry.list("ui_panel")]
    assert set(window.DEFAULT_TOP + window.DEFAULT_BOTTOM) <= set(builtin)
    for component_id, dock in window.workspace.docks.items():
        assert dock.objectName() == component_id and dock.content is not None and not dock.isHidden()
    assert window.metrics is not None and window.model_panel is not None and window.clean_chat is not None
    view = [a.objectName() for a in window.menuBar().actions()[0].menu().actions()]
    assert "toggle:builtin.ui.measurements" in view


def test_plugin_panel_gets_worker_events_on_the_gui_thread(make_window, app):
    window = make_window(build_registry(enabled=["example_telemetry_panel"]))
    dock = window.workspace.docks["example_telemetry_panel.live"]
    panel = dock.content
    threads = []
    dock.context.subscribe("telemetry", lambda name, payload: threads.append(QThread.currentThread()), owner=panel)

    class Emitter(QObject):
        telemetry = Signal(dict)
    emitter, thread = Emitter(), QThread()
    emitter.moveToThread(thread)
    emitter.telemetry.connect(window.on_telemetry)  # queued: emitter lives in another thread
    thread.start()
    for value in (0.1, 0.2, 0.3):
        emitter.telemetry.emit({"state": [0.0] * 512, "state_rms": value, "delta_relative_rms": value})
    deadline = 0
    while panel.received < 3 and deadline < 200:
        app.processEvents()
        deadline += 1
    thread.quit()
    thread.wait()
    assert panel.received == 3 and "0.30000" in panel.values.text()
    assert threads and all(t == app.thread() for t in threads), "Panels are updated on the GUI thread only"


def test_closing_a_plugin_panel_releases_it_and_reopening_rebuilds_it(make_window, app):
    window = make_window(build_registry(enabled=["example_telemetry_panel"]))
    dock = window.workspace.docks["example_telemetry_panel.live"]
    old = dock.content
    window.events.publish("telemetry", {"state_rms": 0.5})
    assert old.received == 1
    dock.close()
    flush(app)
    assert dock.content is None and not shiboken6.isValid(old), "The widget and its subscriptions are gone"
    window.events.publish("telemetry", {"state_rms": 0.5})  # no stale receiver may fail
    dock.reopen()
    assert dock.content is not None and dock.content is not old and dock.content.received == 0
    window.events.publish("telemetry", {"state_rms": 0.5})
    assert dock.content.received == 1
    builtin = window.workspace.docks["builtin.ui.measurements"]
    builtin.close()
    flush(app)
    assert shiboken6.isValid(window.metrics), "Built-in panels are hidden, not destroyed"
    builtin.reopen()


def test_layout_is_restored_by_panel_id(make_window, app):
    window = make_window(build_registry(enabled=["example_telemetry_panel"]), prefs="layout.ini")
    window.workspace.docks["builtin.ui.neuron"].close()
    window.workspace.docks["example_telemetry_panel.live"].close()
    window.save_layout()
    restored = make_window(build_registry(enabled=["example_telemetry_panel"]), prefs="layout.ini")
    assert restored.workspace.docks["builtin.ui.neuron"].isHidden()
    assert restored.workspace.docks["example_telemetry_panel.live"].isHidden()
    restored.reset_layout()
    assert not restored.workspace.docks["builtin.ui.neuron"].isHidden()


def test_failing_panel_factory_is_isolated(make_window):
    registry = build_registry()

    def broken(context):
        raise RuntimeError("panel construction failed")
    registry.register_plugin("badpanel", lambda p: p.panel("badpanel.view", "Broken", broken))
    window = make_window(registry)
    dock = window.workspace.docks["badpanel.view"]
    assert isinstance(dock.content, QLabel) and "panel construction failed" in dock.content.text()
    assert "panel construction failed" in window.plugin_log.toPlainText()
    window.events.publish("telemetry", {"state_rms": 0.1})


def test_parameter_form_generates_and_validates_controls(app):
    form = ParameterForm((Parameter("on", "bool", True), Parameter("n", "int", 3, minimum=0, maximum=9),
                          Parameter("x", "float", 0.5, minimum=0.0, maximum=1.0), Parameter("s", "str", "a"),
                          Parameter("e", "enum", "b", choices=("a", "b")), Parameter("layer", "layer", 2),
                          Parameter("r", "range", (0.1, 0.2), minimum=0.0, maximum=1.0)), layers=28)
    assert form.values() == {"on": True, "n": 3, "x": 0.5, "s": "a", "e": "b", "layer": 2, "r": (0.1, 0.2)}
    assert form.controls["layer"]["widget"].maximum() == 27, "Layer controls follow the model's layer count"
    form.controls["r"]["widget"].findChildren(QWidget)[0].setValue(0.9)
    with pytest.raises(PluginError, match="range minimum exceeds maximum"):
        form.values()
    assert "range" in form.error.text()
    shiboken6.delete(form)


def test_components_panel_selects_plugin_interface_and_defers_connectome(make_window):
    window = make_window(build_registry(enabled=["example_research_interface"]))
    panel = window.workspace.docks["builtin.ui.components"].content
    commands = []
    window.command.connect(lambda name, payload: commands.append((name, payload)))
    row = panel.rows["neural_interface"]
    row["combo"].setCurrentIndex(row["combo"].findData("example_research_interface.gated"))
    panel.apply("neural_interface")
    assert commands[-1] == ("configure", {"components": {"neural_interface": "example_research_interface.gated"},
                                          "component_params": {"example_research_interface.gated":
                                                               {"gate_gain": 10.0, "floor": 1e-6}}})
    connectome = panel.rows["connectome"]
    connectome["combo"].setCurrentIndex(connectome["combo"].findData("builtin.connectome.random_control"))
    count = len(commands)
    panel.apply("connectome")
    assert len(commands) == count and "next launch" in panel.message.text()
    saved = json.loads(window.preferences.value("components/selection"))
    assert saved["connectome"] == "builtin.connectome.random_control"


def test_end_to_end_custom_interface_with_its_own_panel(make_window, app, monkeypatch, tmp_path):
    registry = build_registry(enabled=["example_research_interface"])
    engine = build_engine(monkeypatch, tmp_path, registry=registry)
    window = make_window(registry)
    worker = EngineWorker(ObservatoryConfig(), "unused", "unused", registry=registry)
    worker.engine = engine
    window.command.connect(worker.execute)
    worker.telemetry.connect(window.on_telemetry)
    worker.finished.connect(window.on_finished)
    worker.plugin_log.connect(window.on_plugin_log)
    window.on_initialized({"mode": "live", "model": engine.model_info()})
    panel = window.workspace.docks["example_research_interface.panel"].content
    panel.form.controls["gate_gain"]["widget"].setValue(25.0)
    panel.use_gated()
    flush(app)
    assert type(engine.coupling.interface).__name__ == "GatedSteeringInterface"
    assert engine.params["neural_interface"]["gate_gain"] == 25.0
    assert "example_research_interface.gated" in panel.state.text()
    worker.execute("generate", {"prompt": "Hi", "tokens": 3})
    flush(app)
    assert panel.history and "gate" in panel.gate.text(), "The panel shows the interface's own telemetry"
    output = tmp_path / "experiments"
    window.run_plugin_experiment("example_research_interface.demo", output_root=output)
    flush(app)
    assert "FINISHED" in window.plugin_status.text(), window.plugin_log.toPlainText()
    saved = json.loads(next(output.rglob("gated_vs_original.json")).read_text())
    assert set(saved["result"]) == {"builtin.interface.original", "example_research_interface.gated"}
    assert engine.config.components["neural_interface"] == "example_research_interface.gated", \
        "The experiment restores the selection it found"
    shiboken6.delete(worker)
