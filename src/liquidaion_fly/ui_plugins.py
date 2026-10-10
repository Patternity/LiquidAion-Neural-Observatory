"""Qt side of the plugin system: event hub, panel context, parameter forms and the dock workspace.

Panels are ComponentSpecs of category "ui_panel" (plugin_api.PluginRegistrar.panel). Their
factory(context) runs on the GUI thread and returns a QWidget. Panels receive events through
context.subscribe(); delivery is always on the GUI thread, and subscriptions end automatically
when the panel widget is destroyed. Panels change the engine only through context commands,
which the engine worker applies between operations.

Events (payload is a small dict; never a GPU tensor):
  model_loaded           model information (profile, layers, hidden size, components, ...)
  session_started        {"operation": "new_session"}
  experiment_started     {"operation": name}
  experiment_completed   {"operation": name, "result": summary}
  experiment_cancelled   {"operation": name}
  experiment_failed      {"operation": name, "error": message}
  token                  {"branch": "clean" | "fly", "text": decoded text so far}
  telemetry              throttled engine snapshot (<= 20 per second): reservoir state (list),
                         state_rms, delta_relative_rms, steering_vector_norm, input_norm,
                         interface_metrics, simulation_time, status, ...
  configuration_changed  the engine settings after a successful configure
  continuous_progress    Continuous Generation progress rows
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox, QWidget)

from .plugin_api import ComponentSpec, Parameter, PluginError

EVENTS = ("model_loaded", "session_started", "experiment_started", "experiment_completed", "experiment_cancelled",
          "experiment_failed", "token", "telemetry", "configuration_changed", "continuous_progress")
LAYOUT_VERSION = 2


class EventHub(QObject):
    """GUI-thread event fan-out. Publish only from the GUI thread (window slots do)."""
    event = Signal(str, object)

    def publish(self, name: str, payload: Any = None) -> None:
        if name not in EVENTS:
            raise ValueError(f"Unknown event {name!r}; events: {EVENTS}")
        self.event.emit(name, payload)


class Subscription(QObject):
    """Delivers selected events to a callback; owned by (and destroyed with) the panel widget."""

    def __init__(self, hub: EventHub, names: Iterable[str], callback: Callable[[str, Any], None],
                 owner: QObject, on_error: Callable[[str], None]):
        super().__init__(owner)
        unknown = set(names) - set(EVENTS)
        if unknown:
            raise ValueError(f"Unknown events {sorted(unknown)}; events: {EVENTS}")
        self.names, self.callback, self.on_error, self.hub = frozenset(names), callback, on_error, hub
        hub.event.connect(self._deliver)

    @Slot(str, object)
    def _deliver(self, name: str, payload: Any) -> None:
        if name in self.names:
            try:
                self.callback(name, payload)
            except Exception as exc:  # a faulty panel must not break the observatory
                self.on_error(f"Panel callback for {name!r} failed: {type(exc).__name__}: {exc}")

    def cancel(self) -> None:
        try:
            self.hub.event.disconnect(self._deliver)
        except (RuntimeError, TypeError):
            pass
        self.deleteLater()


class PanelContext:
    """What a panel factory receives. Commands are queued to the engine worker (thread-safe)."""

    def __init__(self, panel_id: str, shell: Any):
        self.panel_id = panel_id
        self.shell = shell                      # the main window; built-in panels use it, plugins should not
        self.registry = shell.registry
        self.hub: EventHub = shell.events
        self.settings = QSettings(shell.preferences.fileName(), shell.preferences.format()) \
            if shell.preferences.fileName() else QSettings("LiquidAion", "Observatory")
        self.settings.beginGroup(f"panels/{panel_id}")
        self.subscriptions: list[Subscription] = []

    def subscribe(self, events: str | Iterable[str], callback: Callable[[str, Any], None],
                  owner: QObject) -> Subscription:
        """callback(name, payload) runs on the GUI thread; the subscription ends when owner is destroyed."""
        names = [events] if isinstance(events, str) else list(events)
        subscription = Subscription(self.hub, names, callback, owner, self.log)
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe_all(self) -> None:
        for subscription in self.subscriptions:
            subscription.cancel()
        self.subscriptions.clear()

    def model_info(self) -> dict | None:
        return self.shell.model_info

    def is_busy(self) -> bool:
        return not self.shell.send.isEnabled()

    def command(self, name: str, payload: Any = None) -> None:
        """Queue an engine command; it runs on the worker after the current operation."""
        self.shell.command.emit(name, payload)

    def configure(self, **settings: Any) -> None:
        self.command("configure", settings)

    def select_component(self, category: str, component_id: str, params: dict | None = None) -> None:
        """Select a component (validated by the engine; nothing is substituted on error)."""
        self.registry.get(component_id, category)
        payload: dict[str, Any] = {"components": {category: component_id}}
        if params is not None:
            payload["component_params"] = {component_id: params}
        self.command("configure", payload)

    def set_component_params(self, component_id: str, params: dict) -> None:
        self.registry.get(component_id).resolve_params(params)
        self.command("configure", {"component_params": {component_id: params}})

    def run_experiment(self, component_id: str) -> None:
        """Run a registered experiment on the engine worker (the shell locks the controls meanwhile)."""
        self.registry.get(component_id, "experiment")
        self.shell.run_plugin_experiment(component_id)

    def log(self, message: str) -> None:
        self.shell.on_plugin_log(f"[{self.panel_id}] {message}")


class ParameterForm(QWidget):
    """Controls generated from Parameter declarations; values() returns validated values."""
    changed = Signal()

    def __init__(self, parameters: Iterable[Parameter], values: dict | None = None, layers: int | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.parameters = list(parameters)
        self.controls: dict[str, Any] = {}
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        values = values or {}
        for parameter in self.parameters:
            control = self._control(parameter, values.get(parameter.name, parameter.default), layers)
            self.controls[parameter.name] = control
            label = QLabel(parameter.name)
            label.setToolTip(parameter.description)
            layout.addRow(label, control["widget"])
        self.error = QLabel("")
        self.error.setStyleSheet("color:#f08a8a;font-size:10px")
        self.error.setWordWrap(True)
        layout.addRow(self.error)
        if not self.parameters:
            layout.addRow(QLabel("No parameters"))

    def _control(self, p: Parameter, value: Any, layers: int | None) -> dict[str, Any]:
        if p.type == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(self.changed)
            return {"widget": box, "get": box.isChecked}
        if p.type in {"int", "layer"}:
            spin = QSpinBox()
            low = int(p.minimum) if p.minimum is not None else (0 if p.type == "layer" else -10**9)
            high = int(p.maximum) if p.maximum is not None else ((layers - 1) if p.type == "layer" and layers else 10**9)
            spin.setRange(low, high)
            spin.setValue(int(value if value is not None else low))
            spin.valueChanged.connect(self.changed)
            return {"widget": spin, "get": spin.value}
        if p.type == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(p.minimum if p.minimum is not None else -1e12, p.maximum if p.maximum is not None else 1e12)
            spin.setSingleStep(p.step or 0.01)
            spin.setValue(float(value if value is not None else 0.0))
            spin.valueChanged.connect(self.changed)
            return {"widget": spin, "get": spin.value}
        if p.type == "enum":
            combo = QComboBox()
            for choice in p.choices:
                combo.addItem(str(choice), choice)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.currentIndexChanged.connect(self.changed)
            return {"widget": combo, "get": combo.currentData}
        if p.type == "range":
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            spins = []
            for part in (value or (p.minimum or 0.0, p.maximum or 1.0)):
                spin = QDoubleSpinBox()
                spin.setDecimals(6)
                spin.setRange(p.minimum if p.minimum is not None else -1e12, p.maximum if p.maximum is not None else 1e12)
                spin.setValue(float(part))
                spin.valueChanged.connect(self.changed)
                box.addWidget(spin)
                spins.append(spin)
            return {"widget": row, "get": lambda: (spins[0].value(), spins[1].value())}
        line = QLineEdit(str(value or ""))
        line.textChanged.connect(self.changed)
        if p.type == "path":
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            browse = QPushButton("…")
            browse.setFixedWidth(28)
            browse.clicked.connect(lambda: line.setText(QFileDialog.getExistingDirectory(self, p.name) or line.text()))
            box.addWidget(line, 1)
            box.addWidget(browse)
            return {"widget": row, "get": line.text}
        return {"widget": line, "get": line.text}

    def values(self) -> dict[str, Any]:
        """Validated values; raises PluginError (and shows it) when a value is invalid."""
        result = {}
        try:
            for parameter in self.parameters:
                result[parameter.name] = parameter.validate(self.controls[parameter.name]["get"]())
        except PluginError as exc:
            self.error.setText(str(exc))
            raise
        self.error.setText("")
        return result


class PanelDock(QDockWidget):
    """A dock for one panel spec. Non-persistent panels are destroyed on close and rebuilt on reopen."""

    def __init__(self, spec: ComponentSpec, shell: Any, persistent: bool):
        super().__init__(spec.name, shell.workspace)
        self.spec, self.shell, self.persistent = spec, shell, persistent
        self.setObjectName(spec.component_id)  # stable id for saved layouts
        self.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.context: PanelContext | None = None
        self.content: QWidget | None = None
        self.error: str | None = None

    def build(self) -> None:
        if self.content is not None:
            return
        self.context = PanelContext(self.spec.component_id, self.shell)
        try:
            widget = self.spec.factory(self.context)
            if not isinstance(widget, QWidget):
                raise PluginError(f"{self.spec.component_id}: the panel factory must return a QWidget")
            self.error = None
        except Exception as exc:
            self.context.unsubscribe_all()
            self.error = f"{type(exc).__name__}: {exc}"
            widget = QLabel(f"Panel {self.spec.component_id} failed to load:\n{self.error}")
            widget.setWordWrap(True)
            self.shell.on_plugin_log(f"Panel {self.spec.component_id} failed: {self.error}")
        self.content = widget
        self.setWidget(widget)

    def dispose(self) -> None:
        if self.context is not None:
            self.context.unsubscribe_all()
            self.context = None
        if self.content is not None:
            self.content.deleteLater()
            self.content = None
            self.setWidget(QWidget())

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if not self.persistent:
            self.dispose()
        super().closeEvent(event)

    def reopen(self) -> None:
        self.build()
        self.show()
        self.raise_()


class Workspace(QMainWindow):
    """The dockable panel area inside the shell window."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setDockOptions(QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AnimatedDocks)
        self.docks: dict[str, PanelDock] = {}

    def add(self, dock: PanelDock) -> None:
        if dock.spec.component_id in self.docks:
            raise PluginError(f"Panel {dock.spec.component_id} is already in the workspace")
        self.docks[dock.spec.component_id] = dock
        area = {"left": Qt.LeftDockWidgetArea, "right": Qt.RightDockWidgetArea, "top": Qt.TopDockWidgetArea,
                "bottom": Qt.BottomDockWidgetArea}[dock.spec.metadata.get("area", "right")]
        self.addDockWidget(area, dock)

    def arrange(self, top: list[str], bottom: list[str], tabs: dict[str, list[str]]) -> None:
        """The default layout: a top row and a bottom row of docks, plus tab groups."""
        for row, area in ((top, Qt.TopDockWidgetArea), (bottom, Qt.BottomDockWidgetArea)):
            docks = [self.docks[i] for i in row if i in self.docks]
            for dock in docks:
                self.removeDockWidget(dock)
                self.addDockWidget(area, dock)
                dock.show()
            for left, right in zip(docks, docks[1:]):
                self.splitDockWidget(left, right, Qt.Horizontal)
        for anchor, members in tabs.items():
            if anchor in self.docks:
                for member in members:
                    if member in self.docks:
                        self.tabifyDockWidget(self.docks[anchor], self.docks[member])
                self.docks[anchor].raise_()
