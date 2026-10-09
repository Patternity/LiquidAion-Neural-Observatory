"""A local neural observatory; all model work belongs to one dedicated thread."""
from __future__ import annotations

import html
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QObject, QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSpinBox, QSplitter, QTextBrowser, QTextEdit,
    QVBoxLayout, QWidget)

from .brain_view import BrainView

# GUI bounds for generation settings; the engine validates the actual context budget.
NEW_TOKENS_MAX = 4096
CONTEXT_MIN, CONTEXT_MAX, CONTEXT_STEP = 512, 32768, 512
PROMPT_ROOM = 64
STOP_NOTES = {"max_new_tokens": "TRUNCATED · reached max_new_tokens ({limit}); not a completed answer",
              "cancelled": "CANCELLED · partial answer",
              "cancelled_before_start": "NOT GENERATED · cancelled before this branch started"}


def apply_preferences(config, preferences):
    """Overlay the persisted generation limits on the launch configuration when valid."""
    if config is None or not hasattr(config, "max_new_tokens"):
        return config
    try:
        new_tokens = int(preferences.value("generation/max_new_tokens", config.max_new_tokens))
        context = int(preferences.value("generation/max_context_tokens", config.max_context_tokens))
        candidate = replace(config, max_new_tokens=new_tokens, max_context_tokens=context)
        candidate.validate()
        return candidate
    except (TypeError, ValueError):
        return config


STYLE = """
QWidget { background:#0b111e; color:#dbe6f8; font-family:Inter,Arial,sans-serif; font-size:12px; }
QMainWindow { background:#0b111e; }
QLabel#Title { font-size:23px; font-weight:700; color:#f1f6ff; }
QLabel#Subtitle { color:#8c9fbd; }
QGroupBox { border:1px solid #26334a; border-radius:9px; margin-top:12px; padding:12px 8px 7px; }
QGroupBox::title { subcontrol-origin:margin; left:11px; color:#a8c0e5; }
QTextBrowser,QTextEdit,QLineEdit { background:#101a2b; border:1px solid #26364e; border-radius:7px; padding:7px; }
QPushButton { background:#1a2a41; border:1px solid #344c6c; border-radius:6px; padding:7px 11px; }
QPushButton:hover { background:#243b59; }
QPushButton:disabled { color:#62748c; background:#131c2a; }
QPushButton#Send { background:#285faa; border:1px solid #4b89db; font-weight:bold; }
QComboBox,QSpinBox,QDoubleSpinBox { background:#132137; border:1px solid #30435f; padding:4px; border-radius:4px; }
QCheckBox { spacing:5px; }
QSlider::groove:horizontal { height:4px; background:#243752; }
QSlider::handle:horizontal { background:#71aef5; width:12px; margin:-4px 0; border-radius:5px; }
QSplitter::handle { background:#172337; }
"""


class TracePlot(QWidget):
    """A tiny measured-value plot, never a generated animation."""
    def __init__(self, color="#77b8ff", parent=None):
        super().__init__(parent)
        self.values = deque(maxlen=180)
        self.color = color
        self.setMinimumHeight(44)

    def append(self, value):
        self.values.append(float(value))
        self.update()

    def clear(self):
        self.values.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1726"))
        painter.setPen(QPen(QColor("#233047"), 1))
        painter.drawLine(3, self.height() - 5, self.width() - 3, self.height() - 5)
        if len(self.values) < 2:
            return
        values = list(self.values)
        low, high = min(0.0, min(values)), max(1e-8, max(values))
        painter.setPen(QPen(QColor(self.color), 1.6))
        coords = [(3 + i * (self.width() - 6) / (len(values) - 1),
            4 + (high - v) / (high - low) * (self.height() - 9)) for i, v in enumerate(values)]
        for a, b in zip(coords, coords[1:]):
            painter.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))


class EngineWorker(QObject):
    initialized = Signal(dict)
    token = Signal(str, str)
    telemetry = Signal(dict)
    replay_chat = Signal(dict)
    replay_result = Signal(object)
    finished = Signal(str, object)
    error = Signal(str)
    stopped = Signal()

    def __init__(self, config, model_path, graph_dir, device="cpu", replay_path=None, debug_log=None):
        super().__init__()
        self.debug_log = debug_log
        if isinstance(config, dict):
            from .observatory_engine import ObservatoryConfig
            accepted = {field.name for field in fields(ObservatoryConfig)}
            device = config.get("device", device)
            config = ObservatoryConfig(**{key: value for key, value in config.items() if key in accepted})
        self.config, self.model_path, self.graph_dir = config, model_path, graph_dir
        self.device, self.replay_path = device, replay_path
        self.engine = None
        self.replay = None
        self.cancel_event = threading.Event()
        self.timer = None
        self.last_emit = 0.0
        self.busy = False

    @Slot()
    def initialize(self):
        try:
            if self.replay_path:
                self._load_replay(self.replay_path)
            else:
                from .observatory_engine import ObservatoryEngine
                self.engine = ObservatoryEngine(self.model_path, self.graph_dir, device=self.device, config=self.config)
                if self.debug_log:
                    self.engine.enable_debug_log(*self.debug_log)
                self.initialized.emit({"mode": "live", "node_ids": self.engine.node_ids})
                self.telemetry.emit(self.engine.telemetry.snapshot())
            self.timer = QTimer(self)
            self.timer.setInterval(250)
            self.timer.timeout.connect(self.idle)
            self.timer.start()
        except Exception as exc:
            self.error.emit(f"Initialization failed: {type(exc).__name__}: {exc}")

    def _load_replay(self, path):
        from .recording import load_recording
        candidate = load_recording(Path(path))
        for key, filename in (("connectome_hash", "edges.csv"), ("node_hash", "nodes.csv")):
            expected = hashlib.sha256((Path(self.graph_dir) / filename).read_bytes()).hexdigest()
            if candidate.metadata.get(key) != expected:
                raise ValueError(f"Recording {key} differs from the displayed connectome; neuron identity cannot be verified")
        node_count = len(pd.read_csv(Path(self.graph_dir) / "nodes.csv"))
        if any(len(snapshot.get("state", [])) != node_count for snapshot in candidate.snapshots):
            raise ValueError("Recording state dimension differs from the displayed connectome")
        self.replay = candidate
        self.initialized.emit({"mode": "replay", "snapshots": len(self.replay.snapshots),
            "metadata": self.replay.metadata, "messages": self.replay.messages,
            "timestamps": [snapshot.get("timestamp_utc") for snapshot in self.replay.snapshots]})
        self.replay_event_indices = [index for index, event in enumerate(self.replay.events) if "snapshot" in event]
        if self.replay.snapshots:
            self._seek_replay(0)

    def _seek_replay(self, index):
        self.telemetry.emit(self.replay.snapshot(index))
        end = self.replay_event_indices[index + 1] if index + 1 < len(self.replay_event_indices) else len(self.replay.events)
        prefix = self.replay.events[:end]
        messages = {"clean": [], "fly": []}
        pending = {}
        future_pair = {}
        latest_result = None
        for position, event in enumerate(prefix):
            event_type = event.get("type")
            if event_type in {"chats_cleared", "new_session"}:
                messages = {"clean": [], "fly": []}
                pending = {}
            elif event_type == "generated_token" and not event.get("scientific", False) and event.get("branch") in {"clean", "fly"}:
                pending[event["branch"]] = event.get("text", "")
                if not future_pair:
                    future_pair = next((item for item in self.replay.events[position:] if item.get("type") == "message_pair"), {})
            elif event_type in {"scientific_comparison", "persistence_comparison"}:
                latest_result = dict(event.get("result", {}))
            elif event_type == "demo_free_generation":
                if latest_result is None or latest_result.get("comparison") != event.get("comparison"):
                    latest_result = {"comparison": event.get("comparison")}
                latest_result["free_generation"] = event.get("result", {})
            elif event_type == "message_pair":
                for branch, response in event.get("responses", {}).items():
                    messages[branch].extend([{"role": "user", "content": event["prompt"]},
                        {"role": "assistant", "content": response}])
                pending = {}
                future_pair = {}
        for branch, text in pending.items():
            if future_pair:
                messages[branch].append({"role": "user", "content": future_pair["prompt"]})
            messages[branch].append({"role": "assistant", "content": text})
        self.replay_chat.emit(messages)
        self.replay_result.emit(latest_result)

    @Slot()
    def idle(self):
        if self.engine is not None and not self.busy and self.replay is None:
            try:
                self.telemetry.emit(self.engine.idle_tick())
            except Exception as exc:
                self.error.emit(f"Idle dynamics failed: {exc}")
                self.timer.stop()

    def publish(self, snapshot):
        now = time.monotonic()
        if now - self.last_emit >= 0.05:
            self.telemetry.emit(snapshot)
            self.last_emit = now

    @Slot(str, object)
    def execute(self, command, payload):
        self.busy = True
        self.cancel_event.clear()
        try:
            result = None
            if command == "open_replay":
                self._load_replay(payload)
            elif command == "replay_snapshot":
                if self.replay is not None:
                    self._seek_replay(int(payload))
            elif self.engine is None:
                raise RuntimeError("The model engine is not available in offline replay mode")
            elif command == "generate":
                result = self.engine.generate_pair(payload["prompt"], max_new_tokens=payload["tokens"],
                    cancel_event=self.cancel_event, on_token=lambda branch, token_id, text: self.token.emit(branch, text),
                    on_telemetry=self.publish)
            elif command == "configure":
                result = self.engine.configure(**payload)
            elif command in {"reset_state", "clear_chats", "new_session"}:
                self.replay = None
                result = getattr(self.engine, command)()
            elif command == "save_recording":
                result = self.engine.save_recording(Path(payload))
            elif command == "restore_session":
                result = self.engine.restore_session(Path(payload))
                self.replay = None
            elif command == "scientific":
                result = self.engine.scientific_compare(payload["prompt"], comparison=payload["comparison"], max_tokens=payload["tokens"], cancel_event=self.cancel_event, on_telemetry=self.publish)
            elif command.startswith("demo_"):
                if command == "demo_persistence":
                    result = self.engine.demo_persistence(prompt_a=payload, cancel_event=self.cancel_event, on_telemetry=self.publish)
                else:
                    result = getattr(self.engine, command)(prompt=payload, cancel_event=self.cancel_event, on_telemetry=self.publish)
            else:
                raise ValueError(f"Unknown worker command: {command}")
            if self.engine is not None and self.replay is None:
                self.telemetry.emit(self.engine.telemetry.snapshot())
            self.finished.emit(command, result)
        except Exception as exc:
            self.error.emit(f"{command}: {type(exc).__name__}: {exc}")
            self.finished.emit(command, {"error": str(exc)})
        finally:
            self.busy = False

    @Slot()
    def shutdown(self):
        if self.timer is not None:
            self.timer.stop()
        self.stopped.emit()


class ChatPanel(QGroupBox):
    def __init__(self, title, accent, parent=None):
        super().__init__(title, parent)
        self.history = []
        self.notes = {}
        self.pending = ""
        self.accent = accent
        layout = QVBoxLayout(self)
        self.stats = QLabel("Waiting · 0 tokens")
        self.stats.setStyleSheet(f"color:{accent}")
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(False)
        self.text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text.document().setDocumentMargin(10)
        layout.addWidget(self.stats)
        layout.addWidget(self.text, 1)

    def user(self, text):
        self.history.append(("user", text))
        self.pending = ""
        self.render()

    def stream(self, text):
        self.pending = text
        self.render()

    def complete(self, note=None):
        if self.pending or note:
            self.history.append(("assistant", self.pending))
            if note:
                self.notes[len(self.history) - 1] = note
        self.pending = ""
        self.render()

    def clear(self):
        self.history.clear()
        self.notes.clear()
        self.pending = ""
        self.stats.setText("Waiting · 0 tokens")
        self.render()

    def render(self):
        entries = self.history + ([("assistant", self.pending)] if self.pending else [])
        blocks = []
        for index, (role, text) in enumerate(entries):
            label = "YOU" if role == "user" else "MODEL"
            color = "#91a2be" if role == "user" else self.accent
            safe = html.escape(text).replace("\n", "<br>")
            blocks.append(f'<p style="color:{color};font-size:10px"><b>{label}</b></p><p style="color:#e0e9f8;font-size:13px">{safe}</p>')
            if index in self.notes:
                blocks.append(f'<p style="color:#f0b45a;font-size:10px"><b>{html.escape(self.notes[index])}</b></p>')
        self.text.setHtml("".join(blocks))
        self.text.horizontalScrollBar().setValue(0)
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())


class ObservatoryWindow(QMainWindow):
    command = Signal(str, object)
    shutdown_worker = Signal()

    def __init__(self, config, model_path, graph_dir, anatomy_dir, device="cpu", replay_path=None,
                 start_worker=True, brain_factory=BrainView, preferences=None, debug_log=None):
        super().__init__()
        self.preferences = preferences if preferences is not None else QSettings("LiquidAion", "Observatory")
        config = apply_preferences(config, self.preferences)
        from .anatomy import Anatomy
        from .connectome import load_graph
        self.graph_dir = Path(graph_dir)
        self.anatomy = Anatomy.load(Path(anatomy_dir))
        self.node_table = pd.read_csv(self.graph_dir / "nodes.csv", dtype={"root_id": str})
        graph_loaded = load_graph(self.graph_dir)
        self.graph = graph_loaded[0] if isinstance(graph_loaded, tuple) else graph_loaded
        self.node_ids = [str(x) for x in self.anatomy.neuron_ids]
        if self.node_table["root_id"].tolist() != self.node_ids:
            raise ValueError("Anatomical neuron IDs differ from the connectome order")
        self.latest = {}
        self.science_result = None
        self.active_operation = None
        self.selected = None
        self.visualization_paused = False
        self.is_replay = bool(replay_path)
        self.replay_count = 0
        self.replay_timestamps = []
        self.live_engine_available = False
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self.advance_replay)
        self.setWindowTitle("LiquidAion · Neural Observatory")
        self.resize(1700, 1100)
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(STYLE)
        self._build_ui(config, brain_factory)
        self.thread = None
        self.worker = None
        if start_worker:
            self.thread = QThread(self)
            self.worker = EngineWorker(config, model_path, graph_dir, device, replay_path, debug_log)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.initialize)
            self.command.connect(self.worker.execute)
            self.shutdown_worker.connect(self.worker.shutdown)
            self.worker.initialized.connect(self.on_initialized)
            self.worker.token.connect(self.on_token)
            self.worker.telemetry.connect(self.on_telemetry)
            self.worker.replay_chat.connect(self.on_replay_chat)
            self.worker.replay_result.connect(self.on_replay_result)
            self.worker.finished.connect(self.on_finished)
            self.worker.error.connect(self.on_error)
            self.worker.stopped.connect(self.thread.quit, Qt.DirectConnection)
            self.thread.finished.connect(self.worker.deleteLater)
            self.thread.start()
        self._set_busy(True)
        self.status.setText("OFFLINE REPLAY · loading recording" if replay_path else "INITIALIZING · loading one local model")

    def _build_ui(self, config, brain_factory):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(14, 10, 14, 10)
        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("LiquidAion  /  Neural Observatory")
        title.setObjectName("Title")
        subtitle = QLabel("One frozen language model · two histories · real connectome · measured state")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top.addLayout(title_box, 1)
        self.status = QLabel("INITIALIZING")
        self.status.setStyleSheet("color:#7bb7fc;font-weight:700")
        top.addWidget(self.status)
        outer.addLayout(top)
        main = QSplitter(Qt.Vertical)
        three_panes = QSplitter(Qt.Horizontal)
        self.clean_chat = ChatPanel("CLEAN  /  unmodified baseline", "#89baff")
        self.fly_chat = ChatPanel("FLY  /  connectome feedback", "#8ce5be")
        self.clean_chat.setMinimumWidth(260)
        self.fly_chat.setMinimumWidth(260)
        brain_group = QGroupBox("ANATOMY  /  registered FlyWire reference")
        brain_layout = QVBoxLayout(brain_group)
        self.brain = brain_factory(self.anatomy, self.graph)
        self.brain.neuron_selected.connect(self.select_neuron)
        brain_layout.addWidget(self.brain, 1)
        brain_tools = QHBoxLayout()
        self.connections = QCheckBox("Actual graph connections")
        self.connections.toggled.connect(self.brain.set_connections_visible)
        self.anatomy_check = QCheckBox("Anatomical surface")
        self.anatomy_check.setChecked(True)
        self.anatomy_check.toggled.connect(self.brain.set_anatomy_visible)
        self.pause_button = QPushButton("Pause view")
        self.pause_button.clicked.connect(self.toggle_visualization)
        fit = QPushButton("Full brain")
        fit.clicked.connect(self.brain.reset_camera)
        focus = QPushButton("Focus graph")
        focus.clicked.connect(self.brain.focus_neurons)
        for widget in (self.connections, self.anatomy_check, self.pause_button, fit, focus):
            brain_tools.addWidget(widget)
        brain_layout.addLayout(brain_tools)
        legend = QLabel("Signed rate: blue negative / red positive. Brightness = |state| / 0.300; zero = invisible. Gold = selected neuron.")
        self.activity_legend = legend
        legend.setStyleSheet("color:#9aaccc;font-size:10px")
        brain_layout.addWidget(legend)
        legend.setWordWrap(True)
        brain_group.setMinimumWidth(620)
        three_panes.addWidget(self.clean_chat)
        three_panes.addWidget(brain_group)
        three_panes.addWidget(self.fly_chat)
        three_panes.setStretchFactor(0, 1)
        three_panes.setStretchFactor(1, 2)
        three_panes.setStretchFactor(2, 1)
        three_panes.setSizes([390, 900, 390])
        main.addWidget(three_panes)
        side = QWidget()
        side_layout = QHBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        controls = QGroupBox("COUPLING")
        form = QFormLayout(controls)
        self.steering = QCheckBox("Inject into hidden activations")
        self.steering.setChecked(getattr(config, "steering_enabled", True))
        self.feedback = QCheckBox("LLM → reservoir feedback")
        self.feedback.setChecked(getattr(config, "feedback_enabled", True))
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0, 1)
        self.strength.setSingleStep(.05)
        self.strength.setValue(getattr(config, "strength", .25))
        self.feedback_strength = QDoubleSpinBox()
        self.feedback_strength.setRange(0, 1)
        self.feedback_strength.setSingleStep(.05)
        self.feedback_strength.setValue(getattr(config, "feedback", .25))
        self.activity_scale = QDoubleSpinBox()
        self.activity_scale.setDecimals(3)
        self.activity_scale.setRange(.001, 1)
        self.activity_scale.setValue(.3)
        self.activity_scale.setSingleStep(.05)
        self.activity_scale.valueChanged.connect(self.set_activity_scale)
        form.addRow("Display activity scale", self.activity_scale)
        form.addRow(self.steering)
        form.addRow("Steering strength", self.strength)
        form.addRow(self.feedback)
        form.addRow("Feedback strength", self.feedback_strength)
        for widget in (self.steering, self.feedback):
            widget.toggled.connect(self.configure)
        for widget in (self.strength, self.feedback_strength):
            widget.valueChanged.connect(self.configure)
        side_layout.addWidget(controls)
        metrics = QGroupBox("LIVE MEASUREMENTS")
        metric_layout = QVBoxLayout(metrics)
        self.metrics = QLabel("State RMS 0.0000\nRelative intervention 0.0000\nSimulation time 0.0\nInput norm 0.0000")
        self.metrics.setStyleSheet("font-family:monospace;color:#bfcee6")
        self.metrics.setMinimumHeight(130)
        self.metrics.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        metric_layout.addWidget(self.metrics)
        metric_layout.addWidget(QLabel("Reservoir RMS history"))
        self.rms_plot = TracePlot("#86dabc")
        metric_layout.addWidget(self.rms_plot)
        metric_layout.addWidget(QLabel("Relative hidden-state intervention"))
        self.delta_plot = TracePlot("#e5bb79")
        metric_layout.addWidget(self.delta_plot)
        metric_layout.addWidget(QLabel("Measured steering magnitude"))
        self.steering_plot = TracePlot("#bc9be9")
        metric_layout.addWidget(self.steering_plot)
        self.gpu_metrics = QLabel("Hardware measurement pending")
        self.gpu_metrics.setStyleSheet("font-family:monospace;color:#91a9cd;font-size:11px")
        self.gpu_metrics.setMinimumHeight(48)
        metric_layout.insertWidget(1, self.gpu_metrics)
        for plot in (self.rms_plot, self.delta_plot, self.steering_plot):
            plot.setMinimumHeight(26)
            plot.setMaximumHeight(32)
        side_layout.addWidget(metrics, 2)
        selection = QGroupBox("SELECTED NEURON")
        selection_layout = QVBoxLayout(selection)
        self.neuron_combo = QComboBox()
        self.neuron_combo.addItems(self.node_ids)
        self.neuron_combo.currentIndexChanged.connect(self.select_neuron)
        selection_layout.addWidget(self.neuron_combo)
        self.neuron_info = QLabel("Select a neuron in the 3D view or by root ID")
        self.neuron_info.setWordWrap(True)
        self.neuron_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        selection_layout.addWidget(self.neuron_info)
        self.neuron_plot = TracePlot("#f5d285")
        selection_layout.addWidget(self.neuron_plot)
        side_layout.addWidget(selection)
        experiment = QGroupBox("EXPERIMENTS")
        exp_layout = QVBoxLayout(experiment)
        self.comparison = QComboBox()
        self.comparison.addItem("Scientific: clean vs Fly (same tokens)", "clean_vs_fly")
        self.comparison.addItem("Scientific: Fly vs random topology", "fly_vs_random")
        exp_layout.addWidget(self.comparison)
        self.scientific = QPushButton("Run controlled comparison")
        self.scientific.clicked.connect(self.run_scientific)
        exp_layout.addWidget(self.scientific)
        demo_row = QHBoxLayout()
        self.demo_buttons = []
        for title, command in (("A · Coupling", "demo_clean_vs_fly"), ("B · Topology", "demo_random_topology"), ("C · Memory", "demo_persistence")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, cmd=command: self.run_demo(cmd))
            demo_row.addWidget(button)
            self.demo_buttons.append(button)
        exp_layout.addLayout(demo_row)
        self.science_details = QCheckBox("Review full result and traces")
        self.science_details.toggled.connect(self.render_science_result)
        exp_layout.addWidget(self.science_details)
        self.experiment_result = QTextBrowser()
        self.experiment_result.setMinimumHeight(130)
        self.experiment_result.setMaximumHeight(250)
        self.experiment_result.setPlainText("Controlled comparisons use identical token context. Text differences alone do not establish topology benefit.")
        exp_layout.addWidget(self.experiment_result)
        side_layout.addWidget(experiment, 2)
        main.addWidget(side)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        main.setSizes([580, 360])
        outer.addWidget(main, 1)
        input_row = QHBoxLayout()
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Shared prompt for CLEAN and FLY. Each branch keeps its own history.")
        self.prompt.returnPressed.connect(self.send_prompt)
        self.context_tokens = QSpinBox()
        self.context_tokens.setRange(CONTEXT_MIN, CONTEXT_MAX)
        self.context_tokens.setSingleStep(CONTEXT_STEP)
        self.context_tokens.setValue(getattr(config, "max_context_tokens", 4096))
        self.context_tokens.setPrefix("Context: ")
        self.context_tokens.setToolTip("Total context window: history + latest message + new tokens. "
                                       "Oldest complete exchanges are dropped from the model input when it is full. "
                                       "Eager attention on an 8 GB GPU fits prompts of about 5k tokens.")
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, NEW_TOKENS_MAX)
        self.max_tokens.setValue(getattr(config, "max_new_tokens", 512))
        self.max_tokens.setPrefix("Max new: ")
        self.max_tokens.setToolTip("max_new_tokens: generation limit per answer, identical for Clean and Fly. "
                                   "Answers that hit it are marked TRUNCATED.")
        self._limit_new_tokens()
        self.max_tokens.valueChanged.connect(self.generation_settings_changed)
        self.context_tokens.valueChanged.connect(self.generation_settings_changed)
        self.context_info = QLabel("Context use: –")
        self.context_info.setStyleSheet("color:#8c9fbd;font-size:10px")
        self.send = QPushButton("Send to both")
        self.send.setObjectName("Send")
        self.send.clicked.connect(self.send_prompt)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.cancel_generation)
        input_row.addWidget(self.prompt, 1)
        input_row.addWidget(self.max_tokens)
        input_row.addWidget(self.context_tokens)
        input_row.addWidget(self.send)
        input_row.addWidget(self.cancel)
        outer.addLayout(input_row)
        outer.addWidget(self.context_info)
        session_row = QHBoxLayout()
        for title, callback in (("Reset state", lambda: self.command.emit("reset_state", None)),
                ("Clear chats", self.clear_chats), ("New session", self.new_session),
                ("Save session", self.save_session), ("Open replay", self.open_replay),
                ("Restore session", self.restore_session), ("Export dashboard", self.export_dashboard)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            session_row.addWidget(button)
        self.replay_play = QPushButton("Play replay")
        self.replay_play.clicked.connect(self.toggle_replay)
        self.replay_speed = QComboBox()
        for speed in (.25, .5, 1, 2, 4):
            self.replay_speed.addItem(f"{speed:g}×", speed)
        self.replay_speed.setCurrentIndex(2)
        self.replay_speed.currentIndexChanged.connect(self.update_replay_speed)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.valueChanged.connect(self.seek_replay)
        self.timeline.setRange(0, 0)
        self.timeline_label = QLabel("0 / 0")
        session_row.addWidget(self.replay_play)
        session_row.addWidget(self.replay_speed)
        session_row.addWidget(self.timeline, 1)
        session_row.addWidget(self.timeline_label)
        outer.addLayout(session_row)
        self.footnote = QLabel("Reservoir: sparse leaky rate dynamics, not a biophysical simulation. Rendering follows real telemetry; view pause does not pause the engine.")
        self.footnote.setStyleSheet("color:#7388aa;font-size:10px")
        outer.addWidget(self.footnote)
        self.setCentralWidget(central)

    def _set_busy(self, busy):
        enabled = not busy and not self.is_replay
        for widget in (self.send, self.scientific, self.steering, self.feedback, self.strength, self.feedback_strength,
                       self.max_tokens, self.context_tokens, *self.demo_buttons):
            widget.setEnabled(enabled)
        self.cancel.setEnabled(busy and not self.is_replay)

    @Slot(dict)
    def on_initialized(self, information):
        self.is_replay = information["mode"] == "replay"
        if not self.is_replay:
            self.live_engine_available = True
            self.command.emit("configure", self.generation_settings())
        self._set_busy(False)
        self.status.setText("OFFLINE REPLAY · model not required" if self.is_replay else "LIVE · model ready · idle dynamics active")
        if self.is_replay:
            self.replay_count = information["snapshots"]
            self.replay_timestamps = information.get("timestamps", [])
            self.timeline.setRange(0, max(0, self.replay_count - 1))
            self.timeline_label.setText(f"1 / {self.replay_count}")
            self.clean_chat.clear()
            self.fly_chat.clear()
            self._render_replay_messages(information.get("messages", []))

    @Slot(object)
    def on_replay_result(self, result):
        self.science_result = result
        if result is None:
            self.experiment_result.setPlainText("No completed experiment at this recorded time.")
        else:
            self.render_science_result()

    @Slot(dict)
    def on_replay_chat(self, messages):
        self.clean_chat.history.clear()
        self.fly_chat.history.clear()
        self.clean_chat.pending = self.fly_chat.pending = ""
        self._render_replay_messages(messages)

    def _render_replay_messages(self, messages):
        if isinstance(messages, dict):
            for branch in ("clean", "fly"):
                panel = self.clean_chat if branch == "clean" else self.fly_chat
                for message in messages.get(branch, []):
                    panel.history.append((message.get("role", "assistant"), message.get("content", "")))
                panel.render()
        elif isinstance(messages, list):
            for message in messages:
                branch = message.get("branch", "fly")
                panel = self.clean_chat if branch == "clean" else self.fly_chat
                if "role" in message:
                    panel.history.append((message["role"], message.get("content", message.get("text", ""))))
            self.clean_chat.render()
            self.fly_chat.render()

    def set_activity_scale(self, value):
        self.brain.set_activity_scale(value)
        self.activity_legend.setText(f"Signed rate: blue negative / red positive. Brightness = |state| / {value:.3f}; zero = invisible. Gold = selected neuron.")

    def _limit_new_tokens(self):
        self.max_tokens.setMaximum(max(1, min(NEW_TOKENS_MAX, self.context_tokens.value() - PROMPT_ROOM)))

    def generation_settings(self):
        return {"max_new_tokens": self.max_tokens.value(), "max_context_tokens": self.context_tokens.value()}

    def generation_settings_changed(self, *args):
        self._limit_new_tokens()
        settings = self.generation_settings()
        for key, value in settings.items():
            self.preferences.setValue(f"generation/{key}", value)
        self.preferences.sync()
        if self.live_engine_available and not self.is_replay:
            self.command.emit("configure", settings)

    def configure(self):
        self.command.emit("configure", {"steering_enabled": self.steering.isChecked(),
            "feedback_enabled": self.feedback.isChecked(), "strength": self.strength.value(), "feedback": self.feedback_strength.value()})

    def send_prompt(self):
        prompt = self.prompt.text().strip()
        if not prompt or not self.send.isEnabled():
            return
        self.clean_chat.user(prompt)
        self.fly_chat.user(prompt)
        self.prompt.clear()
        self._set_busy(True)
        self.active_operation = "generate"
        self.status.setText("GENERATING · CLEAN → FLY · one shared model")
        self.command.emit("generate", {"prompt": prompt, "tokens": self.max_tokens.value()})

    @Slot(str, str)
    def on_token(self, branch, text):
        (self.clean_chat if branch == "clean" else self.fly_chat).stream(text)

    @Slot(dict)
    def on_telemetry(self, snapshot):
        self.latest = snapshot
        state = snapshot.get("state", [])
        rms = float(snapshot.get("state_rms", 0))
        delta = float(snapshot.get("delta_relative_rms", 0))
        self.metrics.setText(f"State RMS        {rms:.6f}\nRelative delta   {delta:.6f}\nSimulation time  {float(snapshot.get('simulation_time', 0)):.2f}\nInput norm       {float(snapshot.get('input_norm', 0)):.6f}\nMapped input     {float(snapshot.get('mapped_input_norm', 0)):.6f}\nSteering norm    {float(snapshot.get('steering_vector_norm', 0)):.6f}\nFeedback events  {int(snapshot.get('feedback_event_count', 0))}")
        if str(snapshot.get("device", "cpu")).startswith("cuda"):
            labels = (("allocated", "gpu_allocated_bytes"), ("reserved", "gpu_reserved_bytes"),
                ("free", "gpu_free_bytes"), ("total", "gpu_total_bytes"))
            measured = [f"{label} {float(snapshot[key]) / 2**30:.2f} GiB" if key in snapshot else f"{label} unavailable"
                for label, key in labels]
            self.gpu_metrics.setText("CUDA · " + " / ".join(measured[:2]) + "\n" + " / ".join(measured[2:]))
        else:
            cuda_available = snapshot.get("hardware", {}).get("torch_cuda_available")
            suffix = "CUDA unavailable" if cuda_available is False else "CUDA not active"
            self.gpu_metrics.setText(f"CPU · {suffix}\nGPU allocated / reserved / free / total: unavailable")
        if not self.visualization_paused:
            if len(state) == len(self.node_ids):
                self.brain.update_state(state)
            self.rms_plot.append(rms)
            self.delta_plot.append(delta)
            self.steering_plot.append(snapshot.get("steering_vector_norm", 0))
            if self.selected is not None and len(state) == len(self.node_ids):
                self.neuron_plot.append(state[self.selected])
                self._update_neuron_info()
        branch = snapshot.get("branch")
        if branch in {"clean", "fly"} and snapshot.get("prompt_tokens") is not None:
            self._show_context_use(branch, snapshot)
        if self.active_operation == "science" and branch is not None and not self.is_replay:
            self.status.setText(f"EXPERIMENT · {branch} · token {snapshot.get('token_index', '–')}")
        if branch in {"clean", "fly"}:
            panel = self.clean_chat if branch == "clean" else self.fly_chat
            panel.stats.setText(f"{snapshot.get('status', 'idle')} · {snapshot.get('generated_tokens', 0)} tokens · {float(snapshot.get('tokens_per_second', 0)):.2f} tok/s")

    def _show_context_use(self, branch, snapshot):
        prompt, limit = int(snapshot["prompt_tokens"]), int(snapshot.get("max_new_tokens", 0))
        context = int(snapshot.get("context_limit", 0))
        text = f"Context use ({branch}): prompt {prompt} + max new {limit} / context {context} tokens"
        dropped = int(snapshot.get("history_dropped_messages", 0))
        if dropped:
            text += f" · oldest {dropped} messages left out of the model input"
        warn = bool(snapshot.get("context_near_limit"))
        if warn:
            text += " · CONTEXT NEARLY FULL"
        self.context_info.setText(text)
        self.context_info.setStyleSheet(f"color:{'#f0b45a' if warn else '#8c9fbd'};font-size:10px")

    def select_neuron(self, index):
        if not 0 <= index < len(self.node_ids):
            return
        self.selected = index
        self.neuron_plot.clear()
        self.neuron_combo.blockSignals(True)
        self.neuron_combo.setCurrentIndex(index)
        self.neuron_combo.blockSignals(False)
        self.brain.select_neuron(index)
        self._update_neuron_info()

    def _update_neuron_info(self):
        if self.selected is None:
            return
        row = self.node_table.iloc[self.selected]
        state = self.latest.get("state", [])
        value = f"{state[self.selected]:.6f}" if len(state) == len(self.node_ids) else "unavailable"
        fields = [f"Root ID: {self.node_ids[self.selected]}", f"State: {value}"]
        for key in ("class", "sub_class", "side", "nt_type", "hemilineage"):
            if key in row and pd.notna(row[key]) and str(row[key]):
                fields.append(f"{key}: {row[key]}")
        fields.append("Position: official representative point (nm)")
        fields.append(", ".join(f"{x:.0f}" for x in self.anatomy.positions[self.selected]))
        self.neuron_info.setText("\n".join(fields))

    def toggle_visualization(self):
        self.visualization_paused = not self.visualization_paused
        self.pause_button.setText("Resume view" if self.visualization_paused else "Pause view")
        if not self.visualization_paused and self.latest:
            self.on_telemetry(self.latest)

    @Slot(str, object)
    def on_finished(self, command, result):
        if command in {"generate", "scientific", "demo_clean_vs_fly", "demo_random_topology", "demo_persistence"}:
            self.active_operation = None
        if isinstance(result, dict) and result.get("error"):
            self.on_error(f"{command}: {result['error']}")
            return
        if command == "generate":
            for branch, panel in (("clean", self.clean_chat), ("fly", self.fly_chat)):
                report = result.get(branch, {}) if isinstance(result, dict) else {}
                reason = report.get("stop_reason") if isinstance(report, dict) else None
                note = STOP_NOTES.get(reason)
                panel.complete(note.format(limit=report.get("max_new_tokens", "?")) if note else None)
                if isinstance(report, dict) and "seconds" in report:
                    label = "Truncated" if reason == "max_new_tokens" else "Complete"
                    panel.stats.setText(f"{label} · {len(report.get('token_ids', []))} tokens · stop: {reason or 'unknown'} · {report['seconds']:.2f}s · {report.get('tokens_per_second', 0):.2f} tok/s")
        if command == "scientific" or command.startswith("demo_"):
            self.science_result = result
            self.render_science_result()
        if command in {"generate", "scientific", "demo_clean_vs_fly", "demo_random_topology", "demo_persistence"}:
            self._set_busy(False)
            self.status.setText("LIVE · idle dynamics active" if not self.is_replay else "OFFLINE REPLAY")
        if command == "save_recording":
            self.status.setText("SESSION SAVED")
        if command == "restore_session" and isinstance(result, dict) and isinstance(result.get("settings"), dict):
            # Show the restored limits without re-sending them to the engine.
            for widget, key in ((self.context_tokens, "max_context_tokens"), (self.max_tokens, "max_new_tokens")):
                widget.blockSignals(True)
                widget.setValue(int(result["settings"].get(key, widget.value())))
                widget.blockSignals(False)
            self._limit_new_tokens()

    def render_science_result(self):
        result = self.science_result
        if not isinstance(result, dict):
            return
        if self.science_details.isChecked():
            self.experiment_result.setPlainText(json.dumps(result, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
            return
        metrics = result.get("metrics", {})
        lines = [f"Comparison: {result.get('comparison', 'unknown')}",
            f"Same prefix: {result.get('matched_prefix', False)} · matched steps: {result.get('matched_steps', 0)}"]
        for label, key in (("Mean KL", "mean_kl_baseline_to_condition"),
                ("Logit RMS", "logit_rms_difference"), ("Argmax disagreement", "argmax_disagreement_fraction")):
            if key in metrics:
                lines.append(f"{label}: {float(metrics[key]):.6g}")
        for branch in ("reference", "condition"):
            trace = result.get(f"{branch}_trace", [])
            relative = [float(row["delta_relative_rms"]) for row in trace if "delta_relative_rms" in row]
            if relative:
                lines.append(f"{branch.capitalize()} hidden perturbation: mean {np.mean(relative):.6g}, max {max(relative):.6g}")
            before = [float(row["before_rms"]) for row in trace if "before_rms" in row]
            after = [float(row["after_rms"]) for row in trace if "after_rms" in row]
            if before and after:
                lines.append(f"{branch.capitalize()} hidden RMS: {np.mean(before):.6g} → {np.mean(after):.6g}")
        free = result.get("free_generation", {})
        for branch in ("reference", "condition"):
            response = free.get(branch, {})
            if response:
                lines.append(f"Free {response.get('condition', branch)}: {response.get('text', '')}")
                lines.append(f"{len(response.get('token_ids', []))} tokens · {response.get('seconds', 0):.2f}s · {response.get('tokens_per_second', 0):.2f} tok/s")
        if "before_idle_state_rms" in result:
            lines.append(f"Actual idle RMS: {result['before_idle_state_rms']:.6g} → {result.get('retained_state_rms', 0):.6g}")
        if result.get("cancelled"):
            lines.append("Cancelled at token boundary.")
        lines.append("Measured coupling effect; topology benefit is not established.")
        self.experiment_result.setPlainText("\n".join(lines))

    @Slot(str)
    def on_error(self, message):
        self.status.setText("ERROR · see details")
        self.experiment_result.setPlainText(message)
        self._set_busy(False)
        self.footnote.setText(message)

    def cancel_generation(self):
        if self.worker is not None:
            self.worker.cancel_event.set()
        self.status.setText("CANCEL REQUESTED · finishing current token")

    def clear_chats(self):
        self.clean_chat.clear()
        self.fly_chat.clear()
        self.command.emit("clear_chats", None)

    def new_session(self):
        self.clear_chats()
        self.rms_plot.clear()
        self.delta_plot.clear()
        self.steering_plot.clear()
        self.command.emit("new_session", None)

    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save complete session", "liquidaion-session.json", "Session recording (*.json)")
        if path:
            self.command.emit("save_recording", path)

    def open_replay(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open offline recording", "", "Session recording (*.json)")
        if path:
            self.replay_timer.stop()
            self.is_replay = True
            self.command.emit("open_replay", path)

    def restore_session(self):
        if not self.live_engine_available:
            self.on_error("Restoring a live session requires launching with local model weights. This process is an offline replay.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Restore model session", "", "Session recording (*.json)")
        if path:
            self.is_replay = False
            self.command.emit("restore_session", path)
            self._set_busy(False)

    def run_scientific(self):
        prompt = self.prompt.text().strip() or "Describe a quiet morning in one sentence."
        self._set_busy(True)
        self.active_operation = "science"
        self.status.setText("SCIENTIFIC COMPARISON · identical token context")
        self.command.emit("scientific", {"prompt": prompt, "comparison": self.comparison.currentData(), "tokens": min(16, self.max_tokens.value())})

    def run_demo(self, command):
        self._set_busy(True)
        self.active_operation = "science"
        self.status.setText("RUNNING DEMONSTRATION")
        self.command.emit(command, self.prompt.text().strip() or "Describe a quiet morning in one sentence.")

    def seek_replay(self, index):
        self.timeline_label.setText(f"{index + 1} / {self.replay_count}")
        if self.is_replay:
            self.command.emit("replay_snapshot", index)

    def toggle_replay(self):
        if not self.is_replay or not self.replay_count:
            return
        if self.replay_timer.isActive():
            self.replay_timer.stop()
            self.replay_play.setText("Play replay")
        else:
            if self.timeline.value() >= self.timeline.maximum():
                self.timeline.setValue(0)
            self.update_replay_speed()
            self.replay_timer.start()
            self.replay_play.setText("Pause replay")

    def update_replay_speed(self):
        interval = 100.0
        index = self.timeline.value()
        if index + 1 < len(self.replay_timestamps):
            try:
                before = datetime.fromisoformat(self.replay_timestamps[index])
                after = datetime.fromisoformat(self.replay_timestamps[index + 1])
                interval = max(1.0, (after - before).total_seconds() * 1000)
            except (TypeError, ValueError):
                pass
        self.replay_timer.setInterval(max(10, int(interval / float(self.replay_speed.currentData()))))

    def advance_replay(self):
        if self.timeline.value() >= self.timeline.maximum():
            self.replay_timer.stop()
            self.replay_play.setText("Play replay")
        else:
            self.timeline.setValue(self.timeline.value() + 1)
            self.update_replay_speed()

    def export_dashboard(self, directory=None):
        if not isinstance(directory, (str, Path)):
            directory = QFileDialog.getExistingDirectory(self, "Export measured dashboard")
        if not directory:
            return
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        encoded = QByteArray()
        buffer = QBuffer(encoded)
        buffer.open(QIODevice.WriteOnly)
        if not self.grab().save(buffer, "PNG"):
            raise RuntimeError("Dashboard PNG encoding failed")
        buffer.close()
        pixels = bytes(encoded)
        if not pixels.endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82"):
            raise RuntimeError("Dashboard PNG encoding was incomplete")
        temporary = destination / "dashboard.png.tmp"
        temporary.write_bytes(pixels)
        temporary.replace(destination / "dashboard.png")
        self.brain.export_image(destination / "anatomy-state.png")
        (destination / "telemetry.json").write_text(json.dumps(self.latest, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
        if self.science_result is not None:
            (destination / "science-result.json").write_text(json.dumps(self.science_result, indent=2,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
        summary = ["# LiquidAion measured dashboard", "", f"Mode: {'offline replay' if self.is_replay else 'live'}", "",
            "Anatomy: registered FlyWire/FAFB14.1 coordinates in nanometers.", "",
            "State is a sparse leaky rate reservoir, not a biophysical simulation.", "", "## Clean conversation", ""]
        summary.extend(f"{role}: {text}\n" for role, text in self.clean_chat.history)
        summary.extend(["## Fly conversation", ""])
        summary.extend(f"{role}: {text}\n" for role, text in self.fly_chat.history)
        summary.extend(["## Controlled comparison", "", self.experiment_result.toPlainText()])
        (destination / "summary.md").write_text("\n".join(summary), encoding="utf-8")
        self.status.setText(f"EXPORTED · {destination.name}")

    def closeEvent(self, event):
        self.replay_timer.stop()
        if self.thread is not None and self.thread.isRunning():
            self.worker.cancel_event.set()
            self.shutdown_worker.emit()
            if not self.thread.wait(1500):
                event.ignore()
                self.status.setText("CLOSING · waiting for current model operation")
                QTimer.singleShot(500, self.close)
                return
        self.brain.close()
        super().closeEvent(event)


def run_gui(config, model_path, graph_dir, anatomy_dir, replay_path=None, device="cpu", debug_log=None):
    app = QApplication.instance() or QApplication([])
    window = ObservatoryWindow(config, model_path, graph_dir, anatomy_dir, device=device, replay_path=replay_path,
                               debug_log=debug_log)
    window.show()
    return app.exec()
