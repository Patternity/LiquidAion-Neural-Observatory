# Writing a LiquidAion plugin

This guide is for researchers who know Python and want to change one assumption of
LiquidAion, without editing the application itself. Examples of such assumptions are the
steering formula, the connectome, the reservoir dynamics or a panel. For the full contracts,
see [PLUGINS.md](PLUGINS.md).

> Plugins are ordinary Python code. They run inside the Observatory with your user's full
> permissions. Enable only plugins you wrote or have read.

## 1. Where a plugin lives

```text
~/.liquidaion/plugins/my_steering/        (or plugins/packages/ in the source checkout,
    plugin.json                            or any directory listed in LIQUIDAION_PLUGIN_PATH)
    plugin.py
    my_interface.py                        any helper modules you like
    panel.py                               optional, Qt widgets
```

`plugin.json` is read without running code:

```json
{"id": "my_steering", "name": "My steering", "version": "0.1.0", "api_version": "1.0",
 "description": "What it changes and why."}
```

The id uses lowercase letters, digits and `_`. Every component the plugin registers must start
with `my_steering.`. Optional fields:

- `"requires": ["other_plugin"]`;
- `"entry": "plugin.py"`.

## 2. Registering components

`plugin.py` defines `register(plugin)`. One plugin may register several related things.
Copy `plugins/packages/example_research_interface/`, rename it, and change the formula:

```python
from liquidaion_fly.plugin_api import Parameter
from .my_interface import MyInterface


def make_interface(context):
    # context: hidden_size, reservoir_neurons, features, seed, params, model, num_layers, device
    return MyInterface(context.hidden_size, context.reservoir_neurons, context.features, context.seed,
                       scale=context.params["scale"])


def create_panel(context):          # Qt is imported only when the panel is shown
    from .panel import MyPanel
    return MyPanel(context)


def run(api):                       # an experiment: the same API as "Run Python Script…"
    api.new_session()
    api.select_components(neural_interface="my_steering.interface")
    api.save_result("demo", api.generate("Once upon a time", max_tokens=32))


def register(plugin):
    plugin.neural_interface("my_steering.interface", "My steering", make_interface,
                            parameters=(Parameter("scale", "float", 1.0, "Delta multiplier", minimum=0.0),))
    plugin.panel("my_steering.panel", "My steering", create_panel, area="right")
    plugin.experiment("my_steering.demo", "My steering demo", run)
```

A neural interface usually subclasses the original one and overrides `steer()`:

```python
import torch
from liquidaion_fly.interfaces import OriginalNeuralInterface


class MyInterface(OriginalNeuralInterface):
    def __init__(self, hidden_size, reservoir_neurons, features, seed, scale):
        super().__init__(hidden_size, reservoir_neurons, features, seed)   # same matrices as the original
        self.scale = scale

    def steer(self, h, state, strength):
        modified, internals = super().steer(h, state, strength * self.scale)
        internals["metrics"] = {"scale": self.scale}        # shown in telemetry as interface_metrics
        return modified, internals
```

Write your own formula in `steer`. These rules apply:

- `h` is the layer output `(1, seq, hidden)`. Change only what you mean to change, usually the
  last position.
- Return a tensor of the same shape, dtype and device, plus a dict.
- The kernel rejects anything else with a message that names the method.
- Do not register hooks, call the model or advance the reservoir. The kernel decides when your
  methods run.

**Connectome.** Return `liquidaion_fly.connectomes.Connectome(component_id, weights, node_ids, ...)`
from `plugin.connectome(...)`.

- `weights` is a sparse `W[post, pre]` matrix.
- Leave unknown metadata as `None`: neurotransmitters, positions, synapses.

**Reservoir.** Subclass `liquidaion_fly.reservoir.Reservoir`, or provide the attributes listed in
PLUGINS.md.

- Declare `capabilities`.
- Without `"extra_input"`, the DA cue is refused, not emulated.

## 3. Enabling it

Use any one of these:

- **Plugins → Manage plugins…**, tick it, then restart;
- `PYTHONPATH=src python -m liquidaion_fly.observatory --plugin my_steering`;
- `LIQUIDAION_PLUGINS=my_steering ./scripts/launch_observatory.sh`.

Problems appear in the **Plugins · scripts** panel, for example a missing dependency or an
exception in `register()`. The rest of the Observatory keeps working.

## 4. Selecting and configuring it

Nothing from a plugin is selected automatically. The Original interface stays active until you
choose otherwise.

- **Components panel.** Choose the interface, set its parameters (generated from your
  `Parameter` list and validated) and press **Apply**.
  - The change takes effect between operations.
  - Changing the interface keeps the reservoir state. Changing the reservoir starts from zero.
  - Changing the connectome applies at the next launch.
- **Your own panel.** Call `context.select_component("neural_interface", "my_steering.interface", params)`.
- **From an experiment.** Call `api.select_components(neural_interface="my_steering.interface")`
  and `api.set_component_params("my_steering.interface", scale=2.0)`.

## 5. Running its experiment

Choose it in **Plugins · scripts → Run experiment**, or start it from your panel with
`context.run_experiment("my_steering.demo")`. While it runs:

- the controls are locked;
- **Stop script** requests cooperative cancellation; `api.check_cancelled()` raises at the next
  check;
- a failure shows its traceback in the plugin console and releases the engine.

## 6. Results

Results are saved in `results/plugins/<experiment id>/<time>/`. Each `save_result()` JSON
records:

- the model;
- every selected component, with version and parameters;
- the coupling semantics, the reservoir and the session.

`plugin_run.json` records the run status.

## 7. A panel that shows live data

```python
from PySide6.QtWidgets import QLabel


class MyPanel(QLabel):
    def __init__(self, context):
        super().__init__("waiting…")
        # Runs on the GUI thread; ends automatically when this widget is destroyed.
        context.subscribe("telemetry", self.on_event, owner=self)

    def on_event(self, name, payload):
        self.setText(f"state RMS {payload.get('state_rms', 0):.4f} · {payload.get('interface_metrics')}")
```

**Events:**

- `model_loaded`;
- `session_started`;
- `experiment_started`, `experiment_completed`, `experiment_cancelled`, `experiment_failed`;
- `token`;
- `telemetry` (at most 20 per second);
- `configuration_changed`;
- `continuous_progress`.

**Panel behaviour:**

- You never call Qt from another thread; the context delivers events on the GUI thread.
- Closing your panel destroys it, and **View** reopens it.
- Its position is remembered between launches.
- For parameters, `ParameterForm(spec.parameters)` from `liquidaion_fly.ui_plugins` builds
  validated controls.

## 8. Testing headless

```python
from liquidaion_fly.plugin_api import build_registry
from liquidaion_fly.observatory_engine import ObservatoryEngine, ObservatoryConfig

registry = build_registry(enabled=["my_steering"])           # finds ~/.liquidaion/plugins etc.
engine = ObservatoryEngine("Qwen/Qwen3-0.6B-Base", "data/cx512", device="cuda", registry=registry,
                           config=ObservatoryConfig(model_profile="qwen3-0.6b-base",
                                                    revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                                                    components={"neural_interface": "my_steering.interface"}))
print(engine.model_info()["components"]["neural_interface"])
print(engine.generate_pair("Once upon a time", max_new_tokens=16)["fly"]["text"])
```

Loading plugins and the engine never imports Qt. For unit tests without a model, use
`registry.register_plugin("test", register)` and the tiny random-weight engine in
`tests/conftest.py` (see `tests/test_plugin_system.py`).

## 9. Troubleshooting

| Message | Meaning |
|---|---|
| `… returned (…); the reservoir expects a numpy vector of shape (64,)` | `observe` or `feedback` must return `features` floats as a numpy array |
| `… returned torch.Size([…]); expected a tensor of shape …` | `steer` changed the shape: return the whole layer output |
| `… returned torch.float32 on cpu; expected torch.bfloat16 on cuda:0` | cast and move your delta: `.to(h.dtype)`, keep `h.device` |
| `Nonfinite hidden activation after intervention` | your formula produced NaN or inf, e.g. a division by a near-zero norm |
| `Neural interface was built for hidden size …` | use `context.hidden_size`; do not hard-code model sizes |
| `Component '…' is not registered … nothing is substituted` | the plugin is not enabled or failed to load; see the plugin console |
| `… must start with 'my_steering.'` | component ids are namespaced by the plugin id |
| `targets plugin API 2.0` | the manifest's `api_version` must be compatible with `liquidaion_fly.plugin_api.API_VERSION` |
| `… does not support extra per-neuron input` | your reservoir does not declare the `extra_input` capability needed by the DA cue |

## Examples in this repository

| Directory | Shows |
|---|---|
| `plugins/packages/example_normalized_interface/` | A: an alternative interface (unit-RMS steering, experimental) and a comparison experiment |
| `plugins/packages/example_telemetry_panel/` | B: a dockable panel with live telemetry, safe subscription and cleanup |
| `plugins/packages/example_research_interface/` | C: one plugin with an interface, its configuration and visualization panel, and an experiment |

The examples demonstrate replaceability. None of them is claimed to be better, smarter or more
biological than the original.
