"""Control window for continuous sliding-window generation ("Infinite Nekrasov").

Besides run parameters it offers live operator injection, an optional external neural cue,
and a runtime leak-rate control. Every control acts on the running system through the
engine worker; the text colors only display recorded token provenance and are never part
of the model input.
"""
from __future__ import annotations

import math
import shutil
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSlider, QSpinBox, QTextEdit, QVBoxLayout, QWidget)

from .continuous import (BASELINE_LEAK, CONDITIONS, DEFAULT_INSTRUCTION, DEFAULT_SEED_TEXT, PROMPT_POLICIES,
                         cue_neuron_indices)

ROOT = Path(__file__).resolve().parents[2]
WINDOW_PRESETS = (64, 128, 256, 512)
LENGTH_PRESETS = (256, 1000, 5000, 10000)
PLOT_CAPACITY = 40000
LEAK_MIN, LEAK_MAX = 0.005, 0.5
CATEGORY_COLORS = {"autonomous": "#c9d3e3", "external": "#f0a35e", "post_injection": "#6fa8ff"}
CATEGORY_LABELS = {"autonomous": "AUTONOMOUS", "external": "EXTERNAL INJECTION", "post_injection": "POST-INJECTION"}


class LeakControl(QWidget):
    """Logarithmic slider plus a precise spin box for the reservoir leak dt/tau."""
    committed = Signal(float)

    def __init__(self, value=BASELINE_LEAK, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(LEAK_MIN, LEAK_MAX)
        self.spin.setSingleStep(0.005)
        self.spin.setPrefix("Leak rate: ")
        self.spin.setToolTip("x ← (1−leak)·x + leak·tanh(W·x + input). Smaller: slower state change and potentially "
                             "longer persistence. Larger: faster response to the current input. Recurrence means the "
                             "network's actual decay is not set by this coefficient alone. Baseline 0.200.")
        self.slider.setToolTip(self.spin.toolTip())
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.set_value(value)
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderReleased.connect(lambda: self.committed.emit(self.value()))
        self.spin.valueChanged.connect(self._spin_changed)

    @staticmethod
    def to_slider(leak):
        return round(1000 * math.log(leak / LEAK_MIN) / math.log(LEAK_MAX / LEAK_MIN))

    @staticmethod
    def from_slider(position):
        return LEAK_MIN * (LEAK_MAX / LEAK_MIN) ** (position / 1000)

    def value(self):
        return round(self.spin.value(), 3)

    def set_value(self, leak):
        for widget, value in ((self.spin, leak), (self.slider, self.to_slider(leak))):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _slider_moved(self, position):
        self.spin.blockSignals(True)
        self.spin.setValue(self.from_slider(position))
        self.spin.blockSignals(False)
        if not self.slider.isSliderDown():
            self.committed.emit(self.value())

    def _spin_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(self.to_slider(value))
        self.slider.blockSignals(False)
        self.committed.emit(self.value())


class LongTracePlot(QWidget):
    """Reservoir RMS and DA-group mean |state| per stream token, with external spans and resets."""

    def __init__(self, color="#86dabc", parent=None):
        super().__init__(parent)
        self.positions: list[int] = []
        self.values: list[float] = []
        self.da: list[float] = []
        self.external: list[int] = []
        self.resets: list[int] = []
        self.total = 1
        self.color = color
        self.setMinimumHeight(110)

    def clear(self, total=1):
        for series in (self.positions, self.values, self.da, self.external, self.resets):
            series.clear()
        self.total = max(1, total)
        self.update()

    def extend(self, points, resets=()):
        for point in points:
            if len(self.values) >= PLOT_CAPACITY:
                break
            position, rms = point[0], point[1]
            self.positions.append(position)
            self.values.append(float(rms))
            self.da.append(float(point[2]) if len(point) > 2 and point[2] is not None else 0.0)
            if len(point) > 3 and point[3] == "external":
                self.external.append(position)
        self.resets.extend(resets)
        if self.positions:
            self.total = max(self.total, self.positions[-1] + 1)
        self.update()

    def _x(self, position, width):
        return 4 + position / self.total * width

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d1726"))
        width, height = self.width() - 8, self.height() - 22
        painter.setPen(QPen(QColor(CATEGORY_COLORS["external"]), 2))
        for position in self.external:
            x = self._x(position, width)
            painter.drawLine(int(x), self.height() - 6, int(x), self.height() - 2)
        painter.setPen(QPen(QColor("#5d6f8e"), 1))
        for position in self.resets:
            x = self._x(position, width)
            painter.drawLine(int(x), 18, int(x), self.height() - 8)
        if len(self.values) >= 2:
            high = max(max(self.values), max(self.da), 1e-8)
            stride = max(1, len(self.values) // max(width, 1))
            for series, color in ((self.da, "#f0a35e"), (self.values, self.color)):
                painter.setPen(QPen(QColor(color), 1.2))
                previous = None
                for index in range(0, len(series), stride):
                    x = self._x(self.positions[index], width)
                    y = 18 + (1 - series[index] / high) * (height - 4)
                    if previous is not None:
                        painter.drawLine(int(previous[0]), int(previous[1]), int(x), int(y))
                    previous = (x, y)
            painter.setPen(QPen(QColor("#9aaccc"), 1))
            painter.drawText(8, 13, f"green: reservoir RMS · orange: DA neurons mean |state| · max {high:.3f} · "
                                    "ticks below: injected tokens")


class ContinuousDialog(QDialog):
    """Parameters, live progress, provenance-colored text, interventions and the RMS trace of one run."""
    start_requested = Signal(dict)
    stop_requested = Signal()
    inject_requested = Signal(str)
    leak_requested = Signal(float)

    def __init__(self, parent=None, output_root: Path | None = None, graph_dir: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Continuous generation · sliding window")
        self.resize(820, 960)
        self.output_root = Path(output_root or ROOT / "results" / "continuous" / "gui")
        self.graph_dir = Path(graph_dir or ROOT / "data" / "cx512")
        self.output_dir: Path | None = None
        self.model_info: dict | None = None
        self.running = False
        self.injection_status: dict[int, str] = {}
        layout = QVBoxLayout(self)
        self.model_label = QLabel("Language model: –")
        self.model_label.setStyleSheet("color:#f5d285")
        self.model_label.setToolTip("Set in the main window (LANGUAGE MODEL, MODEL COUPLING); shared by both views")
        layout.addWidget(self.model_label)
        form = QFormLayout()
        self.seed = QLineEdit(DEFAULT_SEED_TEXT)
        self.window_size = QSpinBox()
        self.window_size.setRange(1, 4096)
        self.window_size.setValue(128)
        self.window_size.setToolTip("Model input = the most recent N tokens. Presets: " + ", ".join(map(str, WINDOW_PRESETS)))
        self.total = QSpinBox()
        self.total.setRange(1, 100000)
        self.total.setValue(1000)
        self.total.setToolTip("Model-generated tokens to produce; injected tokens are extra. Presets: "
                              + ", ".join(map(str, LENGTH_PRESETS)))
        self.condition = QComboBox()
        for name in CONDITIONS:
            self.condition.addItem({"clean": "Clean (no reservoir)", "fly": "Fly · persistent state",
                                    "fly_reset": "Fly · periodic reset", "random": "Random topology"}[name], name)
        self.condition.setCurrentIndex(1)
        self.reset_interval = QSpinBox()
        self.reset_interval.setRange(1, 100000)
        self.reset_interval.setValue(512)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0, 1)
        self.strength.setSingleStep(0.05)
        self.strength.setValue(0.25)
        self.feedback = QDoubleSpinBox()
        self.feedback.setRange(0, 1)
        self.feedback.setSingleStep(0.05)
        self.feedback.setValue(0.25)
        self.decoding = QComboBox()
        self.decoding.addItem("Sample (temperature, top-p)", "sample")
        self.decoding.addItem("Greedy (deterministic baseline)", "greedy")
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.05, 3.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(0.8)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.01, 1.0)
        self.top_p.setSingleStep(0.01)
        self.top_p.setValue(0.9)
        self.sampling_seed = QSpinBox()
        self.sampling_seed.setRange(0, 2**31 - 1)
        self.sampling_seed.setValue(42)
        self.sampling_seed.setToolTip("Same seed in every condition: identical random numbers, so Clean and Fly "
                                      "diverge only where the reservoir changed the distribution")
        self.eos = QComboBox()
        self.eos.addItem("Stop at end-of-text", "stop")
        self.eos.addItem("Suppress EOS (recorded)", "suppress")
        self.prompt_policy = QComboBox()
        for name in PROMPT_POLICIES:
            self.prompt_policy.addItem({"raw": "Raw continuation", "chat_prefix": f"Pinned prefix: «{DEFAULT_INSTRUCTION}»"}[name], name)
        self.leak = LeakControl()
        self.leak.committed.connect(self._leak_committed)
        self.cue = QComboBox()
        self.cue.addItem("Off", "off")
        self.cue.addItem("DA neurons", "da")
        self.cue.addItem("Random size-matched group (control)", "random")
        self.cue_strength = QDoubleSpinBox()
        self.cue_strength.setRange(0.0, 3.0)
        self.cue_strength.setSingleStep(0.05)
        self.cue_strength.setValue(0.5)
        self.cue_strength.setToolTip("Extra input added inside tanh(W·x + input + cue) of the DA neurons while an "
                                     "injected token is processed. 0 is identical to Off.")
        self.cue_info = QLabel()
        self.cue.currentIndexChanged.connect(self._update_cue_info)
        presets = QHBoxLayout()
        for value in LENGTH_PRESETS:
            button = QPushButton(f"{value:,}".replace(",", " "))
            button.clicked.connect(lambda checked=False, v=value: self.total.setValue(v))
            presets.addWidget(button)
        form.addRow("Seed text", self.seed)
        form.addRow("Context window (tokens)", self.window_size)
        form.addRow("Total model tokens", self.total)
        form.addRow("", presets)
        form.addRow("Condition", self.condition)
        form.addRow("Reset every (tokens)", self.reset_interval)
        form.addRow("Steering strength", self.strength)
        form.addRow("Feedback strength", self.feedback)
        form.addRow("Decoding", self.decoding)
        form.addRow("Temperature", self.temperature)
        form.addRow("Top-p", self.top_p)
        form.addRow("Sampling seed", self.sampling_seed)
        form.addRow("End-of-text", self.eos)
        form.addRow("Prompt policy", self.prompt_policy)
        form.addRow("Reservoir dynamics", self.leak)
        form.addRow("External neural cue", self.cue)
        form.addRow("Cue strength", self.cue_strength)
        form.addRow("", self.cue_info)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.start = QPushButton("Start")
        self.start.clicked.connect(self.request_start)
        self.stop = QPushButton("Stop")
        self.stop.clicked.connect(self.stop_requested)
        self.export = QPushButton("Export results…")
        self.export.clicked.connect(self.export_results)
        for button in (self.start, self.stop, self.export):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        inject_row = QHBoxLayout()
        self.inject_text = QLineEdit()
        self.inject_text.setPlaceholderText("Text to insert into the running token stream (tokenized as is, no role markers)")
        self.inject_text.returnPressed.connect(self.request_inject)
        self.inject_button = QPushButton("Inject")
        self.inject_button.clicked.connect(self.request_inject)
        inject_row.addWidget(self.inject_text, 1)
        inject_row.addWidget(self.inject_button)
        layout.addLayout(inject_row)
        self.inject_state = QLabel("No injections")
        self.inject_state.setStyleSheet("color:#9aaccc")
        layout.addWidget(self.inject_state)
        self.progress = QLabel("Idle")
        self.progress.setStyleSheet("font-family:monospace;color:#bfcee6")
        layout.addWidget(self.progress)
        self.plot = LongTracePlot()
        layout.addWidget(self.plot)
        legend = QLabel(" · ".join(f'<span style="color:{CATEGORY_COLORS[key]}">■ {CATEGORY_LABELS[key]}</span>'
                                   for key in ("autonomous", "external", "post_injection")))
        legend.setToolTip("POST-INJECTION: model tokens produced while at least one injected token is inside the "
                          "model's window. Labels are display only and never enter the model input.")
        layout.addWidget(legend)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text, 1)
        self.condition.currentIndexChanged.connect(self._update_enabled)
        self.decoding.currentIndexChanged.connect(self._update_enabled)
        self._update_cue_info()
        self._update_enabled()
        self.set_running(False)

    def set_model_info(self, info: dict) -> None:
        """Mirror the main window's active model and layer pair (the main controls are authoritative)."""
        self.model_info = info
        quantization = f" · {info['quantization'].upper()}" if info.get("quantization") else ""
        self.model_label.setText(f"Language model: {info['display']}{quantization} · capture {info['capture_layer']} "
                                 f"→ steering {info['steer_layer']} · hidden {info['hidden_size']}")
        raw_only = info.get("interaction") != "chat"
        index = self.prompt_policy.findData("chat_prefix")
        item = self.prompt_policy.model().item(index)
        if item is not None:
            item.setEnabled(not raw_only)
        if raw_only and self.prompt_policy.currentData() == "chat_prefix":
            self.prompt_policy.setCurrentIndex(self.prompt_policy.findData("raw"))

    def _update_cue_info(self):
        target = self.cue.currentData()
        if target == "off":
            self.cue_info.setText("Cue off: injected tokens drive the reservoir only through Qwen's activations")
        else:
            indices, roots = cue_neuron_indices(self.graph_dir, target)
            kind = "nt_type = DA (FlyWire prediction)" if target == "da" else "random non-DA control, seed 42"
            self.cue_info.setText(f"{len(indices)} neurons selected · {kind}")
        self._update_enabled()

    def _update_enabled(self):
        self.reset_interval.setEnabled(self.condition.currentData() == "fly_reset" and not self.running)
        coupled = self.condition.currentData() != "clean"
        for widget in (self.strength, self.feedback):
            widget.setEnabled(coupled and not self.running)
        sampled = self.decoding.currentData() == "sample"
        for widget in (self.temperature, self.top_p, self.sampling_seed):
            widget.setEnabled(sampled and not self.running)
        self.cue.setEnabled(coupled and not self.running)
        self.cue_strength.setEnabled(coupled and not self.running and self.cue.currentData() != "off")
        self.leak.setEnabled(coupled)

    def set_running(self, running: bool):
        self.running = running
        for widget in (self.seed, self.window_size, self.total, self.condition, self.eos, self.prompt_policy,
                       self.decoding, self.start):
            widget.setEnabled(not running)
        self.stop.setEnabled(running)
        self.inject_button.setEnabled(running)
        self.inject_text.setEnabled(running)
        self.export.setEnabled(not running and self.output_dir is not None)
        self._update_enabled()

    def payload(self) -> dict:
        condition = self.condition.currentData()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # One subdirectory per model profile and layer pair, so runs of different substrates never collide.
        info = self.model_info or {}
        group = f"{info.get('profile', 'model')}-c{info.get('capture_layer', 'x')}-s{info.get('steer_layer', 'x')}"
        self.output_dir = self.output_root / group / f"{stamp}-{condition}"
        return {"seed_text": self.seed.text().strip() or DEFAULT_SEED_TEXT, "window": self.window_size.value(),
                "total_tokens": self.total.value(), "condition": condition,
                "reset_interval": self.reset_interval.value() if condition == "fly_reset" else None,
                "strength": self.strength.value(), "feedback": self.feedback.value(),
                "eos_policy": self.eos.currentData(), "prompt_policy": self.prompt_policy.currentData(),
                "decoding": self.decoding.currentData(), "temperature": self.temperature.value(),
                "top_p": self.top_p.value(), "sampling_seed": self.sampling_seed.value(),
                "leak_rate": self.leak.value(), "cue_target": self.cue.currentData(),
                "cue_strength": self.cue_strength.value(), "output": str(self.output_dir)}

    def request_start(self):
        payload = self.payload()
        self.text.clear()
        self._append(payload["seed_text"], "autonomous")
        self.plot.clear(payload["total_tokens"])
        self.injection_status.clear()
        self.inject_state.setText("No injections")
        self.progress.setText(f"Starting · output {payload['output']}")
        self.started_at = time.monotonic()
        self.set_running(True)
        self.start_requested.emit(payload)

    def request_inject(self):
        text = self.inject_text.text()
        if not self.running or not text:
            return
        # Exactly what the operator typed is sent; nothing is trimmed or prefixed.
        self.inject_text.clear()
        self.inject_requested.emit(text)

    def on_injection_queued(self, injection_id, text):
        if injection_id is None:
            self.inject_state.setText("Injection rejected: no active run")
            return
        self.injection_status[injection_id] = "pending"
        self._show_injections()

    def _leak_committed(self, value):
        if self.running:
            self.leak_requested.emit(value)

    def _append(self, text, category):
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.End)
        style = QTextCharFormat()
        style.setForeground(QColor(CATEGORY_COLORS.get(category, CATEGORY_COLORS["autonomous"])))
        cursor.insertText(text, style)
        self.text.setTextCursor(cursor)

    def _show_injections(self):
        if self.injection_status:
            self.inject_state.setText("Injections: " + " · ".join(f"#{k} {v}" for k, v in sorted(self.injection_status.items())))

    def on_progress(self, update: dict):
        for text, category in update.get("segments", []):
            if text:
                self._append(text, category)
        if update.get("text") and not update.get("segments"):
            self._append(update["text"], "autonomous")
        self.plot.extend(update.get("rms", []), update.get("resets", []))
        for record in update.get("injections", []):
            status = record["status"]
            if status == "inserted" and record.get("left_window_position") is None:
                status = "inserted · in window"
            elif record.get("left_window_position") is not None:
                status = "left the window"
            self.injection_status[record["id"]] = status
        self._show_injections()
        if update.get("leak") is not None and not self.leak.slider.isSliderDown():
            self.leak.set_value(update["leak"])
        rms = f" · RMS {update['state_rms']:.4f}" if update.get("state_rms") is not None else ""
        da = f" · DA {update['da_abs_mean']:.4f}" if update.get("da_abs_mean") is not None else ""
        category = CATEGORY_LABELS.get(update.get("category"), "")
        self.progress.setText(f"token {update['step'] + 1} / {update['total']} · {update['tokens_per_second']:.1f} tok/s"
                              f" · window {update['window_len']}{rms}{da} · {category}")

    def on_finished(self, summary: dict):
        self.set_running(False)
        if not isinstance(summary, dict) or summary.get("error"):
            self.progress.setText(f"Failed: {summary.get('error') if isinstance(summary, dict) else summary}")
            return
        for record in summary.get("injections", []):
            self.injection_status[record["id"]] = record["status"]
        self._show_injections()
        peak = summary.get("peak_gpu_allocated_bytes")
        memory = f" · peak GPU {peak / 2**30:.2f} GiB" if peak else ""
        self.progress.setText(f"{summary['stop_reason']} · {summary['generated_tokens']} model + "
                              f"{summary.get('external_tokens', 0)} injected tokens · "
                              f"{summary['tokens_per_second']:.1f} tok/s{memory} · saved to {self.output_dir}")

    def export_results(self, destination=None):
        if self.output_dir is None or not self.output_dir.exists():
            return
        if not isinstance(destination, (str, Path)):
            destination = QFileDialog.getExistingDirectory(self, "Export continuous run")
        if destination:
            target = Path(destination) / self.output_dir.name
            shutil.copytree(self.output_dir, target, dirs_exist_ok=True)
            self.progress.setText(f"Exported to {target}")
