"""Compact control window for continuous sliding-window generation ("Infinite Nekrasov")."""
from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .continuous import CONDITIONS, DEFAULT_INSTRUCTION, DEFAULT_SEED_TEXT, PROMPT_POLICIES

ROOT = Path(__file__).resolve().parents[2]
WINDOW_PRESETS = (64, 128, 256, 512)
LENGTH_PRESETS = (256, 1000, 5000, 10000)
PLOT_CAPACITY = 20000


class LongTracePlot(QWidget):
    """Reservoir RMS over the whole run, drawn decimated to the widget width, with reset markers."""

    def __init__(self, color="#86dabc", parent=None):
        super().__init__(parent)
        self.steps: list[int] = []
        self.values: list[float] = []
        self.resets: list[int] = []
        self.total = 1
        self.color = color
        self.setMinimumHeight(90)

    def clear(self, total=1):
        self.steps.clear()
        self.values.clear()
        self.resets.clear()
        self.total = max(1, total)
        self.update()

    def extend(self, points, resets=()):
        for step, value in points:
            if len(self.values) < PLOT_CAPACITY:
                self.steps.append(step)
                self.values.append(float(value))
        self.resets.extend(resets)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d1726"))
        width, height = self.width() - 8, self.height() - 10
        painter.setPen(QPen(QColor("#5d6f8e"), 1))
        for step in self.resets:
            x = 4 + step / self.total * width
            painter.drawLine(int(x), 4, int(x), self.height() - 4)
        if len(self.values) < 2:
            return
        high = max(max(self.values), 1e-8)
        # One vertex per horizontal pixel keeps 10k-token runs cheap to redraw.
        stride = max(1, len(self.values) // max(width, 1))
        painter.setPen(QPen(QColor(self.color), 1.2))
        previous = None
        for index in range(0, len(self.values), stride):
            x = 4 + self.steps[index] / self.total * width
            y = 5 + (1 - self.values[index] / high) * height
            if previous is not None:
                painter.drawLine(int(previous[0]), int(previous[1]), int(x), int(y))
            previous = (x, y)
        painter.setPen(QPen(QColor("#9aaccc"), 1))
        painter.drawText(8, 16, f"state RMS · max {high:.3f}")


class ContinuousDialog(QDialog):
    """Parameters, live progress, streamed text and the RMS trace of one continuous run."""
    start_requested = Signal(dict)
    stop_requested = Signal()

    def __init__(self, parent=None, output_root: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Continuous generation · sliding window")
        self.resize(760, 760)
        self.output_root = Path(output_root or ROOT / "results" / "continuous" / "gui")
        self.output_dir: Path | None = None
        self.running = False
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.seed = QLineEdit(DEFAULT_SEED_TEXT)
        self.window_size = QSpinBox()
        self.window_size.setRange(1, 4096)
        self.window_size.setValue(128)
        self.window_size.setToolTip("Model input = the most recent N tokens. Presets: " + ", ".join(map(str, WINDOW_PRESETS)))
        self.total = QSpinBox()
        self.total.setRange(1, 100000)
        self.total.setValue(1000)
        self.total.setToolTip("Total tokens to generate. Presets: " + ", ".join(map(str, LENGTH_PRESETS)))
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
        self.eos = QComboBox()
        self.eos.addItem("Stop at end-of-text", "stop")
        self.eos.addItem("Suppress EOS (recorded)", "suppress")
        self.prompt_policy = QComboBox()
        for name in PROMPT_POLICIES:
            self.prompt_policy.addItem({"raw": "Raw continuation", "chat_prefix": f"Pinned prefix: «{DEFAULT_INSTRUCTION}»"}[name], name)
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
        presets = QHBoxLayout()
        for value in LENGTH_PRESETS:
            button = QPushButton(f"{value:,}".replace(",", " "))
            button.clicked.connect(lambda checked=False, v=value: self.total.setValue(v))
            presets.addWidget(button)
        form.addRow("Seed text", self.seed)
        form.addRow("Context window (tokens)", self.window_size)
        form.addRow("Total tokens", self.total)
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
        self.progress = QLabel("Idle")
        self.progress.setStyleSheet("font-family:monospace;color:#bfcee6")
        layout.addWidget(self.progress)
        self.plot = LongTracePlot()
        layout.addWidget(self.plot)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text, 1)
        self.condition.currentIndexChanged.connect(self._update_enabled)
        self.decoding.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()
        self.set_running(False)

    def _update_enabled(self):
        self.reset_interval.setEnabled(self.condition.currentData() == "fly_reset" and not self.running)
        coupled = self.condition.currentData() != "clean"
        for widget in (self.strength, self.feedback):
            widget.setEnabled(coupled and not self.running)
        sampled = self.decoding.currentData() == "sample"
        for widget in (self.temperature, self.top_p, self.sampling_seed):
            widget.setEnabled(sampled and not self.running)

    def set_running(self, running: bool):
        self.running = running
        for widget in (self.seed, self.window_size, self.total, self.condition, self.eos, self.prompt_policy,
                       self.decoding, self.start):
            widget.setEnabled(not running)
        self.stop.setEnabled(running)
        self.export.setEnabled(not running and self.output_dir is not None)
        self._update_enabled()

    def payload(self) -> dict:
        condition = self.condition.currentData()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = self.output_root / f"{stamp}-{condition}"
        return {"seed_text": self.seed.text().strip() or DEFAULT_SEED_TEXT, "window": self.window_size.value(),
                "total_tokens": self.total.value(), "condition": condition,
                "reset_interval": self.reset_interval.value() if condition == "fly_reset" else None,
                "strength": self.strength.value(), "feedback": self.feedback.value(),
                "eos_policy": self.eos.currentData(), "prompt_policy": self.prompt_policy.currentData(),
                "decoding": self.decoding.currentData(), "temperature": self.temperature.value(),
                "top_p": self.top_p.value(), "sampling_seed": self.sampling_seed.value(),
                "output": str(self.output_dir)}

    def request_start(self):
        payload = self.payload()
        self.text.setPlainText(payload["seed_text"])
        self.plot.clear(payload["total_tokens"])
        self.progress.setText(f"Starting · output {payload['output']}")
        self.started_at = time.monotonic()
        self.set_running(True)
        self.start_requested.emit(payload)

    def on_progress(self, update: dict):
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(update.get("text", ""))
        self.text.setTextCursor(cursor)
        self.plot.extend(update.get("rms", []), update.get("resets", []))
        rms = f" · RMS {update['state_rms']:.4f}" if update.get("state_rms") is not None else ""
        self.progress.setText(f"token {update['step'] + 1} / {update['total']} · {update['tokens_per_second']:.1f} tok/s"
                              f" · window {update['window_len']}{rms}")

    def on_finished(self, summary: dict):
        self.set_running(False)
        if not isinstance(summary, dict) or summary.get("error"):
            self.progress.setText(f"Failed: {summary.get('error') if isinstance(summary, dict) else summary}")
            return
        peak = summary.get("peak_gpu_allocated_bytes")
        memory = f" · peak GPU {peak / 2**30:.2f} GiB" if peak else ""
        self.progress.setText(f"{summary['stop_reason']} · {summary['generated_tokens']} tokens · "
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
