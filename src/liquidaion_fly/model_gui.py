"""LANGUAGE MODEL selector and MODEL COUPLING layer controls.

The panel only displays what the engine reports and requests changes; the engine validates
and applies them between operations. Layer indices are zero-based.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from .models import LEGACY_PROFILE, PROFILES, coupling_order


def describe(info: dict) -> str:
    quantization = f"{info['quantization'].upper()} (bitsandbytes, compute {info['dtype']})" if info.get("quantization") \
        else str(info.get("dtype"))
    return (f"{info['display']} · {info['kind']} · {quantization} · {info.get('attention')} attention\n"
            f"{info['num_hidden_layers']} layers · hidden {info['hidden_size']} · rev {info['revision'][:10]} · "
            f"{info.get('device')}")


ORDER_TEXT = {
    "capture_first": "Order: capture → steering in the same pass",
    "same_layer": "Order: same layer — capture reads the output before steering",
    "steer_first": "Order: steering first (reservoir state from the previous token), then capture",
}


def interaction_label(info: dict) -> str:
    return "Interaction: Chat / Instruct" if info.get("interaction") == "chat" else \
        "Interaction: Raw continuation / Base (no chat template)"


class ModelPanel(QWidget):
    load_requested = Signal(str)
    layers_requested = Signal(int, int)

    def __init__(self, profile_key: str = LEGACY_PROFILE, parent=None):
        super().__init__(parent)
        self.info: dict | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        model_group = QGroupBox("LANGUAGE MODEL")
        model_layout = QVBoxLayout(model_group)
        row = QHBoxLayout()
        self.combo = QComboBox()
        for key, profile in PROFILES.items():
            self.combo.addItem(profile.display, key)
        self.combo.setCurrentIndex(max(0, self.combo.findData(profile_key)))
        self.load = QPushButton("Load")
        self.load.setToolTip("Unloads the current model, loads the selected one (downloads it once if needed), "
                             "resets the reservoir state and both chats. The connectome is unchanged.")
        self.load.clicked.connect(lambda: self.load_requested.emit(self.combo.currentData()))
        row.addWidget(self.combo, 1)
        row.addWidget(self.load)
        model_layout.addLayout(row)
        self.model_info = QLabel("No model loaded")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("color:#bfcee6;font-size:10px")
        self.interaction = QLabel("")
        self.interaction.setStyleSheet("color:#f5d285;font-size:10px")
        model_layout.addWidget(self.model_info)
        model_layout.addWidget(self.interaction)
        layout.addWidget(model_group)
        coupling = QGroupBox("MODEL COUPLING")
        form = QFormLayout(coupling)
        self.capture = QSpinBox()
        self.steer = QSpinBox()
        self.capture_max = QLabel("/ –")
        self.steer_max = QLabel("/ –")
        for spin, label, title in ((self.capture, self.capture_max, "Capture layer"),
                                   (self.steer, self.steer_max, "Steering layer")):
            box = QHBoxLayout()
            box.addWidget(spin)
            box.addWidget(label)
            form.addRow(title, box)
        self.capture.setToolTip("Decoder layer whose last-token hidden state is projected into the reservoir")
        self.steer.setToolTip("Decoder layer whose last-token hidden state receives the reservoir steering")
        self.dimensions = QLabel("")
        self.dimensions.setStyleSheet("font-family:monospace;color:#91a9cd;font-size:10px")
        self.order = QLabel("")
        self.order.setWordWrap(True)
        self.order.setStyleSheet("color:#f5d285;font-size:10px")
        self.state = QLabel("")
        self.state.setStyleSheet("color:#8c9fbd;font-size:10px")
        form.addRow(self.dimensions)
        form.addRow(self.order)
        form.addRow(self.state)
        layout.addWidget(coupling)
        self.capture.valueChanged.connect(self._capture_changed)
        self.steer.valueChanged.connect(self._steer_changed)

    def set_model_info(self, info: dict) -> None:
        """Show the loaded model; set layer ranges from its real layer count; mark the pair applied."""
        self.info = info
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(max(0, self.combo.findData(info["profile"])))
        self.combo.blockSignals(False)
        layers = int(info["num_hidden_layers"])
        for spin in (self.capture, self.steer):
            spin.blockSignals(True)
        # Any pair is valid, including the same layer and steering below capture.
        for spin, key in ((self.capture, "capture_layer"), (self.steer, "steer_layer")):
            spin.setRange(0, layers - 1)
            spin.setValue(int(info[key]))
        for spin in (self.capture, self.steer):
            spin.blockSignals(False)
        self.capture_max.setText(f"/ {layers - 1}")
        self.steer_max.setText(f"/ {layers - 1}")
        self.model_info.setText(describe(info))
        self.interaction.setText(interaction_label(info))
        fingerprint = info.get("projection_fingerprint") or ""
        self.dimensions.setText(f"Model hidden size: {info['hidden_size']:>6}\nReservoir features: {info['features']:>5}\n"
                                f"Reservoir neurons: {info['reservoir_neurons']:>6}\nProjections: {fingerprint[:12]}")
        self.state.setText(f"Applied: {info['capture_layer']} → {info['steer_layer']}")
        self._show_order()

    def _show_order(self):
        self.order.setText(ORDER_TEXT[coupling_order(self.capture.value(), self.steer.value())])

    def _capture_changed(self, value):
        self._request()

    def _steer_changed(self, value):
        self._request()

    def _request(self):
        if self.info is None:
            return
        self._show_order()
        pair = (self.capture.value(), self.steer.value())
        if pair == (self.info["capture_layer"], self.info["steer_layer"]):
            self.state.setText(f"Applied: {pair[0]} → {pair[1]}")
            return
        self.state.setText(f"Pending: {pair[0]} → {pair[1]} (applies before the next operation)")
        self.layers_requested.emit(*pair)

    def mark_applied(self, capture: int, steer: int) -> None:
        if self.info is not None:
            self.info = {**self.info, "capture_layer": capture, "steer_layer": steer}
        self.state.setText(f"Applied: {capture} → {steer}")

    def set_enabled(self, enabled: bool) -> None:
        for widget in (self.combo, self.load, self.capture, self.steer):
            widget.setEnabled(enabled)
