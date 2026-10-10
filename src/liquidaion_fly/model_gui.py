"""LANGUAGE MODEL selector and MODEL COUPLING (Neural Bypass list) controls.

The panel only displays what the engine reports and requests changes; the engine validates
and applies them between operations. Layer indices are zero-based.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from .bypass import LEGACY_SINGLE, active_bypasses, coupling_mode, normalize_bypasses
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
DIRECTION = {"capture_first": "forward", "same_layer": "self", "steer_first": "backward"}
SYNCHRONOUS_TEXT = ("Synchronous Multi-Bypass: one shared reservoir, one update per token; "
                    "steering uses the state from before the token (next-token coupling)")


def mode_text(rows: list) -> str:
    """Active count and the coupling mode the engine will execute for these rows."""
    active = active_bypasses(rows)
    enabled = sum(1 for row in normalize_bypasses(rows) if row[2])
    if not active:
        return "No bypass enabled — at least one is required"
    count = f"Active: {len(active)} bypass{'es' if len(active) > 1 else ''}"
    if enabled > len(active):
        count += f" ({enabled - len(active)} duplicate row{'s' if enabled - len(active) > 1 else ''} applied once)"
    if coupling_mode(active) == LEGACY_SINGLE:
        return f"{count}\nLegacy single bypass · {ORDER_TEXT[coupling_order(*active[0])]}"
    return f"{count}\n{SYNCHRONOUS_TEXT}"


def interaction_label(info: dict) -> str:
    return "Interaction: Chat / Instruct" if info.get("interaction") == "chat" else \
        "Interaction: Raw continuation / Base (no chat template)"


class BypassRow(QWidget):
    """[enabled] capture -> steering [remove]; the direction label is informational only."""
    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, capture: int, steer: int, enabled: bool, layers: int, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox()
        self.enabled.setChecked(enabled)
        self.enabled.setToolTip("Include this bypass in the coupling")
        self.capture = QSpinBox()
        self.steer = QSpinBox()
        for spin in (self.capture, self.steer):
            spin.setRange(0, max(0, layers - 1))
        self.capture.setValue(capture)
        self.steer.setValue(steer)
        self.capture.setToolTip("Capture layer: its last-token hidden state is projected into the shared reservoir")
        self.steer.setToolTip("Steering layer: its last-token hidden state receives the reservoir steering")
        self.direction = QLabel("")
        self.direction.setStyleSheet("color:#8c9fbd;font-size:10px")
        self.direction.setMinimumWidth(52)
        self.remove = QPushButton("✕")
        self.remove.setToolTip("Remove this bypass")
        self.remove.setFixedWidth(28)
        row.addWidget(self.enabled)
        row.addWidget(self.capture)
        row.addWidget(QLabel("→"))
        row.addWidget(self.steer)
        row.addWidget(self.direction)
        row.addStretch(1)
        row.addWidget(self.remove)
        self.enabled.toggled.connect(self._changed)
        self.capture.valueChanged.connect(self._changed)
        self.steer.valueChanged.connect(self._changed)
        self.remove.clicked.connect(lambda: self.remove_requested.emit(self))
        self._show_direction()

    def value(self) -> list:
        return [self.capture.value(), self.steer.value(), self.enabled.isChecked()]

    def _show_direction(self):
        self.direction.setText(DIRECTION[coupling_order(self.capture.value(), self.steer.value())])

    def _changed(self, *args):
        self._show_direction()
        self.changed.emit()

    def set_controls_enabled(self, enabled: bool, removable: bool) -> None:
        for widget in (self.enabled, self.capture, self.steer):
            widget.setEnabled(enabled)
        self.remove.setEnabled(enabled and removable)


class ModelPanel(QWidget):
    load_requested = Signal(str)
    bypasses_requested = Signal(list)

    def __init__(self, profile_key: str = LEGACY_PROFILE, parent=None):
        super().__init__(parent)
        self.info: dict | None = None
        self.layers = 0
        self.rows: list[BypassRow] = []
        self._controls_enabled = True
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
        coupling = QGroupBox("MODEL COUPLING · NEURAL BYPASS")
        coupling_layout = QVBoxLayout(coupling)
        self.layer_range = QLabel("Capture → steering · layers –")
        self.layer_range.setStyleSheet("color:#8c9fbd;font-size:10px")
        coupling_layout.addWidget(self.layer_range)
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(2)
        coupling_layout.addLayout(self.rows_layout)
        self.add = QPushButton("+ Add Bypass")
        self.add.setToolTip("Add a disabled row with the last row's layers; set the layers, then enable it")
        self.add.clicked.connect(self.add_row)
        coupling_layout.addWidget(self.add)
        self.order = QLabel("")
        self.order.setWordWrap(True)
        self.order.setStyleSheet("color:#f5d285;font-size:10px")
        self.dimensions = QLabel("")
        self.dimensions.setStyleSheet("font-family:monospace;color:#91a9cd;font-size:10px")
        self.state = QLabel("")
        self.state.setWordWrap(True)
        self.state.setStyleSheet("color:#8c9fbd;font-size:10px")
        coupling_layout.addWidget(self.order)
        coupling_layout.addWidget(self.dimensions)
        coupling_layout.addWidget(self.state)
        layout.addWidget(coupling)

    # The first row's controls (single-bypass convenience for scripts and tests).
    @property
    def capture(self) -> QSpinBox:
        return self.rows[0].capture

    @property
    def steer(self) -> QSpinBox:
        return self.rows[0].steer

    def value(self) -> list:
        return [row.value() for row in self.rows]

    def _set_rows(self, rows: list) -> None:
        for row in self.rows:
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.rows = []
        for capture, steer, enabled in normalize_bypasses(rows):
            self._append_row(capture, steer, enabled)
        self._refresh()

    def _append_row(self, capture: int, steer: int, enabled: bool) -> BypassRow:
        row = BypassRow(capture, steer, enabled, self.layers)
        row.changed.connect(self._request)
        row.remove_requested.connect(self.remove_row)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        return row

    def add_row(self) -> None:
        capture, steer, _ = self.rows[-1].value() if self.rows else (0, 0, False)
        self._append_row(capture, steer, False)
        self._request()

    def remove_row(self, row: BypassRow) -> None:
        if len(self.rows) <= 1 or row not in self.rows:
            return
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()
        self._request()

    def _refresh(self) -> None:
        self.order.setText(mode_text(self.value()))
        for row in self.rows:
            row.set_controls_enabled(self._controls_enabled, len(self.rows) > 1)
        self.add.setEnabled(self._controls_enabled)

    def set_model_info(self, info: dict) -> None:
        """Show the loaded model; set layer ranges from its real layer count; mark the rows applied."""
        self.info = info
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(max(0, self.combo.findData(info["profile"])))
        self.combo.blockSignals(False)
        self.layers = int(info["num_hidden_layers"])
        self._set_rows(info.get("bypasses") or [[info["capture_layer"], info["steer_layer"], True]])
        self.layer_range.setText(f"Capture → steering · layers 0…{self.layers - 1} · any direction")
        self.model_info.setText(describe(info))
        self.interaction.setText(interaction_label(info))
        fingerprint = info.get("projection_fingerprint") or ""
        self.dimensions.setText(f"Model hidden size: {info['hidden_size']:>6}\nReservoir features: {info['features']:>5}\n"
                                f"Reservoir neurons: {info['reservoir_neurons']:>6}\nProjections: {fingerprint[:12]}")
        self.state.setText(f"Applied: {self._label(self.value())}")

    @staticmethod
    def _label(rows: list) -> str:
        active = active_bypasses(rows)
        return ", ".join(f"{capture} → {steer}" for capture, steer in active) or "none"

    def _applied_rows(self) -> list:
        if self.info is None:
            return []
        return self.info.get("bypasses") or [[self.info["capture_layer"], self.info["steer_layer"], True]]

    def _request(self) -> None:
        self._refresh()
        if self.info is None:
            return
        rows = self.value()
        if not active_bypasses(rows):
            self.state.setText("Not applied: enable at least one bypass")
            return
        if rows == normalize_bypasses(self._applied_rows()):
            self.state.setText(f"Applied: {self._label(rows)}")
            return
        self.state.setText(f"Pending: {self._label(rows)} (applies before the next operation)")
        self.bypasses_requested.emit(rows)

    def mark_applied(self, rows: list) -> None:
        if self.info is not None:
            active = active_bypasses(rows)
            self.info = {**self.info, "bypasses": normalize_bypasses(rows),
                         "capture_layer": active[0][0], "steer_layer": active[0][1]}
        if self.value() == normalize_bypasses(rows):
            self.state.setText(f"Applied: {self._label(rows)}")

    def set_enabled(self, enabled: bool) -> None:
        self._controls_enabled = enabled
        for widget in (self.combo, self.load):
            widget.setEnabled(enabled)
        self._refresh()
