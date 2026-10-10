"""The plugin's own panel: configure the gated interface, select it, run its experiment, watch the gate."""
from collections import deque

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from liquidaion_fly.plugin_api import PluginError
from liquidaion_fly.ui_plugins import ParameterForm

INTERFACE = "example_research_interface.gated"
EXPERIMENT = "example_research_interface.demo"


class GatedInterfacePanel(QWidget):
    def __init__(self, context):
        super().__init__()
        self.context = context
        spec = context.registry.get(INTERFACE, "neural_interface")
        self.form = ParameterForm(spec.parameters)        # generated from the declared parameters
        self.state = QLabel("Original interface active")
        self.gate = QLabel("gate: –")
        self.gate.setStyleSheet("font-family:monospace")
        self.history = deque(maxlen=50)
        use = QPushButton("Use gated steering")
        use.clicked.connect(self.use_gated)
        original = QPushButton("Back to original")
        original.clicked.connect(lambda: context.select_component("neural_interface", "builtin.interface.original"))
        run = QPushButton("Run demonstration")
        run.clicked.connect(lambda: context.run_experiment(EXPERIMENT))
        buttons = QHBoxLayout()
        for button in (use, original, run):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("EXPERIMENTAL · gated steering (example plugin)"))
        layout.addWidget(self.form)
        layout.addLayout(buttons)
        layout.addWidget(self.state)
        layout.addWidget(self.gate)
        layout.addStretch(1)
        context.subscribe(("telemetry", "configuration_changed"), self.on_event, owner=self)

    def use_gated(self):
        try:
            params = self.form.values()     # validated against the declared Parameter limits
        except PluginError:
            return
        self.context.select_component("neural_interface", INTERFACE, params)

    def on_event(self, name, payload):
        if name == "configuration_changed":
            active = (payload.get("components") or {}).get("neural_interface")
            self.state.setText(f"Active interface: {active}")
        elif payload.get("interface_metrics"):
            metrics = payload["interface_metrics"]
            self.history.append(metrics.get("gate", 0.0))
            self.gate.setText(f"gate {metrics.get('gate', 0):.4f} · state RMS {metrics.get('state_rms', 0):.4f} · "
                              f"mean of last {len(self.history)}: {sum(self.history) / len(self.history):.4f}")
