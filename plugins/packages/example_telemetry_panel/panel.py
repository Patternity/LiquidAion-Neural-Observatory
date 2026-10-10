"""The panel widget. Every callback below runs on the GUI thread (the context guarantees it)."""
from collections import deque

from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class Sparkline(QWidget):
    def __init__(self):
        super().__init__()
        self.values = deque(maxlen=200)
        self.setMinimumHeight(60)

    def add(self, value):
        self.values.append(float(value))
        self.update()

    def paintEvent(self, event):  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d1726"))
        if len(self.values) < 2:
            return
        top = max(self.values) or 1.0
        painter.setPen(QPen(QColor("#e5bb79"), 1.5))
        step = self.width() / (len(self.values) - 1)
        points = [(i * step, self.height() - 4 - v / top * (self.height() - 8)) for i, v in enumerate(self.values)]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


class TelemetryPanel(QWidget):
    def __init__(self, context):
        super().__init__()
        self.received = 0
        self.status = QLabel("Waiting for telemetry…")
        self.values = QLabel("")
        self.values.setStyleSheet("font-family:monospace")
        self.plot = Sparkline()
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.values)
        layout.addWidget(QLabel("Relative hidden-state intervention"))
        layout.addWidget(self.plot)
        # owner=self: the subscription ends automatically when this widget is destroyed
        # (closing the dock destroys plugin panels; reopening builds a new one).
        context.subscribe(("telemetry", "experiment_started", "experiment_completed", "experiment_cancelled",
                           "experiment_failed", "model_loaded"), self.on_event, owner=self)

    def on_event(self, name, payload):
        if name == "telemetry":
            self.received += 1
            delta = float(payload.get("delta_relative_rms") or 0.0)
            self.plot.add(delta)
            self.values.setText(f"state RMS       {float(payload.get('state_rms') or 0):.5f}\n"
                                f"relative delta  {delta:.5f}\n"
                                f"steering norm   {float(payload.get('steering_vector_norm') or 0):.4f}\n"
                                f"events          {self.received}")
        elif name == "model_loaded":
            self.status.setText(f"Model: {payload.get('display')}")
        else:
            self.status.setText(f"{name.replace('_', ' ')}: {payload.get('operation')}")
