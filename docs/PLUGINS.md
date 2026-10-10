# Plugin architecture

LiquidAion is a small execution kernel surrounded by replaceable components. Every scientific
assumption that the kernel does not need to own is a component in **one registry**:

- the language-model integration;
- the connectome;
- the reservoir dynamics;
- the mathematics between the model and the reservoir;
- the coupling timing;
- stimulation;
- experiments;
- GUI panels.

The built-in components reproduce the published reference configuration exactly. They are
selected by default and registered like any other plugin.

For a step-by-step guide to writing a plugin, see
[PLUGIN_DEVELOPMENT_GUIDE.md](PLUGIN_DEVELOPMENT_GUIDE.md).

## Kernel and components

**The kernel stays in core.** It consists of `ObservatoryEngine`, `ObservatoryWindow` and the
plugin loader:

| Responsibility | Where |
|---|---|
| The single loaded model, its lifetime and GPU memory | `ObservatoryEngine` (`switch_model`) |
| KV caches and their invalidation | `ObservatoryEngine` sessions |
| Installing and removing every transformer hook | `coupling.install_hooks`, called from the engine's forward pass |
| Token loop, cancellation, Clean/Fly ordering, safe boundaries for changes | engine lock; the `configure` and `switch_model` boundaries |
| Validation of component outputs (shapes, dtype, device, finite values) | `interfaces.check_features` / `check_steered` |
| Recording, metadata, reproducibility | `SessionRecording`, `model_info()`, `components_info()` |
| Plugin discovery, enabling, dependency and API checks | `plugin_api` |
| Window shell, dock workspace, event delivery to panels | `ObservatoryWindow`, `ui_plugins` |

**Why this split.** These pieces guard correctness across all components. A component must
never be able to:

- leave a hook behind;
- reuse a cache across incompatible states;
- run a second model copy;
- change the reservoir mid-token.

**Components** are registered in `plugin_api.PluginRegistry`:

| Category | Built-in id | Replaces |
|---|---|---|
| `model_adapter` | `builtin.model.qwen3` | model family: profiles, loading, layers, chat or raw rendering |
| `connectome` | `builtin.connectome.flywire_cx512`, `builtin.connectome.random_control` | topology and neuron metadata |
| `reservoir` | `builtin.reservoir.leaky_tanh` | neuronal dynamics |
| `neural_interface` | `builtin.interface.original` | LM ↔ reservoir mathematics |
| `coupling` | `builtin.coupling.auto`, `.legacy_single`, `.synchronous_multi` | when the interface runs in a forward pass |
| `modulation` | `builtin.modulation.neuron_cue` | stimulation groups (the DA cue) |
| `experiment` | (from plugins) | runnable `run(api)` experiments |
| `ui_panel` | `builtin.ui.*` (ten panels) | dockable Qt panels |

**What stays an ordinary library module:**

- serialization;
- recording;
- diagnostics (`steering_diagnostics`, `leak_dynamics`, `dopamine_field`);
- continuous-generation bookkeeping;
- numerical helpers.

They are tools that experiments call, not replaceable behaviours.

## Default selection and reproducibility

`ObservatoryConfig.components` defaults to `builtin_plugins.DEFAULT_COMPONENTS`, and
`component_params` maps a component id to its parameters. A saved selection or parameter that
is not available produces an error that names the missing component. It is never replaced by
another implementation. To recover explicitly, use **Plugins → Restore built-in components**
and restart.

**Recorded with every session, Continuous run and plugin result:**

- in `model_info()` → `components`: for each category, the id, version, plugin, API version
  and parameters;
- the coupling semantics (updates per token, timing);
- the connectome fingerprint and neuron count;
- the neural-interface fingerprint and whether it is recurrent;
- `plugin_api_version`.

Every change at runtime is recorded as a `components_changed` event.

**Exact reproduction** also depends on the GPU, CUDA, PyTorch and transformers versions. The
recording stores the environment, but a configuration file alone does not guarantee identical
numbers on other hardware.

## Component contracts

**Factories and contexts.** Factories receive one context object; its fields per category are
listed below. Factories run when an experiment configuration is resolved, never per token.
The hot path calls direct method references.

### Neural interface

The interface is the mathematics between hidden states and the reservoir. The contract is in
[`interfaces.py`](../src/liquidaion_fly/interfaces.py).

**Context fields:**

- `hidden_size`, `reservoir_neurons`, `features`, `seed`;
- `params`;
- `model`: the profile dictionary;
- `num_layers`, `device`.

**Attributes:**

- `hidden_size`, `reservoir_neurons`, `features`: validated against the model and reservoir;
- `fingerprint`;
- `recurrent`: `True` if the interface keeps its own state; this is recorded separately from
  the reservoir dynamics.

**Methods:**

| Method | Called | Must return |
|---|---|---|
| `observe(hidden_last)` | on each capture layer, last position | numpy `(features,)`, finite |
| `feedback(steered_last)` | after steering, if feedback is on | numpy `(features,)`; the coupling multiplies it by the feedback strength |
| `steer(h, state, strength)` | on each steering layer | `(tensor with h's shape, dtype and device, internals dict)` |
| `reset()` | on new session and state reset | – |
| `describe()` | metadata | JSON-serializable dict |

`internals` may name `steering_vector`, which the steering diagnostics use. `internals["metrics"]`
holds scalar floats that appear in telemetry as `interface_metrics`.

**What an interface must not do:**

- install hooks;
- run the model;
- own caches;
- advance the reservoir.

**Validation.** Wrong shapes, dtypes or devices and nonfinite values raise errors that name the
method. Hooks are always removed afterwards.

**`OriginalNeuralInterface`** is the reference implementation, moved out of the old coupling
class with identical arithmetic and random-number order. Its fingerprints equal the ones
recorded before the migration, for hidden sizes 1024, 2048 and 2560 (tested).

### Coupling policy

**Context fields:**

- `reservoir`, `interface`;
- `bypasses`: the active unique pairs;
- `strength`, `feedback`;
- `mode`: `fly`, `random`, `static` or `none`;
- `hidden_size`, `seed`, `params`.

The policy returns an object derived from `coupling.Coupling`. It declares:

- `hook_plan()`: a list of `(layer, callable)` pairs; the kernel installs them, bounds-checked
  and all or nothing;
- `execution_mode`;
- the record and update semantics.

The spec's `metadata["semantics"]` is recorded. The built-in policies keep their exact
behaviour:

- **legacy single bypass:** forward, self and backward pairs; capture and feedback updates;
- **synchronous multi-bypass:** one snapshot, mean per unique layer, one update.

`builtin.coupling.auto` is the historical rule: one active pair runs legacy, several run
synchronous. The explicit policies make the choice visible:
- legacy refuses more than one pair;
- synchronous also runs a single pair.

### Connectome

**Context fields:**

- `params`;
- `graph_dir`: the launch graph directory;
- `seed`;
- `load(component_id, params)`: for composing providers.

**Return value:** a `connectomes.Connectome` with:

- `weights` (`W[post, pre]`, sparse) and `node_ids`;
- optional `nt_types`, `nt_confidence`, `positions` and `synapses`;
- `provenance`.

Missing metadata is `None`, never invented.

**Built-in providers:**

- **`flywire_cx512`** reads the historical export exactly as before (`connectome.load_graph`).
- **`random_control`** applies `reservoir.random_control` to a base connectome, with the seed
  defaulting to the configuration seed. It keeps the number of nodes and edges, the weight
  multiset and the number of self-loops, and draws uniform off-diagonal positions. Neuron ids
  and labels are kept. The scientific comparisons ("Fly vs random topology", the Continuous
  `random` condition) use this provider on the selected connectome.

### Reservoir

**Context fields:** `connectome`, `features`, `seed`, `leak`, `params`.

**Required attributes:**

- `state`: numpy array of length `connectome.neurons`, assignable;
- `time`, `dynamics.dt`, `leak`, `fingerprint`;
- `set_leak(leak)`;
- `advance(features, steps=1, extra=None)`;
- `reset()`.

**Optional:**

- `input` and `dynamics.input_strength`, for the mapped-input diagnostics;
- `snapshot` / `restore`.

**Capabilities** are declared in the `capabilities` tuple. `extra_input` is required by the
neuron cue. A reservoir without it gets a clear error; it is never emulated.

**The built-in reservoir** is the unchanged leaky-tanh `Reservoir`. Its parameters `dt`,
`gain` and `input_strength` keep the historical defaults, so its fingerprint is unchanged.

### Model adapter

**Context field:** `params`.

The built-in `Qwen3Adapter` provides:

- the pinned profiles: 0.6B Instruct, 0.6B, 1.7B and 4B Base, at the same revisions and with
  the same quantization;
- `load(...)` through the kernel's `LocalLLM`;
- `layers(model)` and `hidden_size(model)`;
- `render(messages, profile, tokenizer)`: the Instruct chat template, or raw continuation for
  Base models.

The adapter changes only through `switch_model` (a model reload). `configure` refuses to
change it.

### Modulation

**Context fields:** `connectome`, `target` (`da`, `random`, `off`), `seed`.

**Return value:** `(indices, root_ids)`.

The kernel uses `builtin.modulation.neuron_cue` for the Continuous Generation cue, with
selection identical to the historical `nodes.csv` rule. The dopamine-field prototype remains a
shadow-mode research plugin and is not connected to the reservoir.

### Experiments

`ExperimentSpec.factory` is `run(api)`. Registered experiments and plain experiment scripts
(`Run Python Script…`) share one runner (`plugins._execute`), one API object and the same
cancellation. Registered experiments save under `results/plugins/<experiment id>/<time>/`.

**API additions** (see [MULTI_BYPASS_AND_PLUGINS.md](MULTI_BYPASS_AND_PLUGINS.md)):

- `api.list_components(category)`;
- `api.components()`;
- `api.select_components(neural_interface=..., coupling=..., reservoir=..., connectome=...,
  params={...})`;
- `api.set_component_params(id, **params)`.

These use the same validated path as the GUI.

## UI plugins

The Observatory window is a shell: header, status, prompt row, session row and menus. Around
it sits a dock workspace (`ui_plugins.Workspace`, a `QMainWindow` of `QDockWidget`s). Every
scientific panel is a `ui_panel` component:

- the ten built-in panels: chats, anatomy, model and bypass, coupling and leak, measurements,
  selected neuron, experiments, plugin console, components;
- any panels from plugins.

**Panel spec.** `registrar.panel(id, title, factory, area="right", visible=True)`.

- `factory(context)` runs on the GUI thread and returns a `QWidget`.
- Import Qt inside the factory, so loading a plugin headless never imports Qt.

**Panel context** (`ui_plugins.PanelContext`):

- `subscribe(events, callback, owner)`: delivery on the GUI thread. The subscription ends when
  `owner` (the panel widget) is destroyed.
- `command(name, payload)`, `configure(**settings)`, `select_component(category, id, params)`,
  `set_component_params(id, params)`, `run_experiment(id)`: queued to the engine worker and
  applied between operations.
- `model_info()`, `is_busy()`, `log(msg)`.
- `settings`: a QSettings group per panel.
- `registry`.

**Events.** Payloads are small dicts; no tensors are sent.

| Event | Payload |
|---|---|
| `model_loaded` | model information |
| `session_started` | – |
| `experiment_started` | `{"operation": …}` |
| `experiment_completed`, `experiment_cancelled`, `experiment_failed` | outcome |
| `token` | `{"branch", "text"}` |
| `telemetry` | engine snapshot, throttled to ≤ 20 per second |
| `configuration_changed` | engine settings after `configure` |
| `continuous_progress` | Continuous Generation progress |

The `telemetry` snapshot carries:

- the reservoir `state` list;
- `state_rms`, `delta_relative_rms`, `steering_vector_norm`, `input_norm`, `simulation_time`;
- `interface_metrics`;
- GPU memory, status.

**Threading.** The worker never touches widgets. Its Qt signals reach window slots on the GUI
thread, which publish to the hub. This is tested from a separate `QThread`.

**Lifecycle:**

- A plugin panel is built when it is shown. Closing it destroys the widget and its
  subscriptions; reopening rebuilds it.
- Built-in panels are hidden on close, because the shell drives them.
- A factory that raises shows an error label, is reported in the plugin console, and leaves
  everything else running.

**Layout.** It is saved per stable panel id (`QMainWindow.saveState`) and restored at launch.
**View → Reset layout** returns to the default, which resembles the earlier fixed window.
Plugin panels open as tabs next to Experiments, and every dock can be moved, tabbed or
floated.

**Parameter editors.** `ui_plugins.ParameterForm(parameters)` generates validated controls from
`Parameter` declarations. Types: `bool`, `int`, `float`, `str`, `enum`, `path`, `layer`
(range from the loaded model) and `range`. The **Components** panel uses it for every
component's parameters.

## Discovery, enabling and trust

**Plugin layout.** A plugin is a directory that contains:

- `plugin.json`: `id`, `name`, `version`, `api_version`, `description`, `requires`, `entry`;
- `plugin.py`: `register(registrar)`.

**Discovery roots:**

- `plugins/packages/` in the source checkout (bundled examples);
- `~/.liquidaion/plugins/`;
- `LIQUIDAION_PLUGIN_PATH` (path list).

Manifests are read without executing code.

**Enabling.** **Only explicitly enabled plugins are imported**:

- **Plugins → Manage plugins…** (saved; applies at the next launch);
- `--plugin ID` on the command line;
- `LIQUIDAION_PLUGINS=a,b`.

**Loading.**

- Plugins load in dependency order as the package `liquidaion_plugins.<id>`, so relative
  imports work and `sys.path` is not touched.
- A plugin's component ids must start with `<id>.`.
- A duplicate id, an API-version mismatch, a missing dependency or an exception in `register()`
  leaves that plugin unloaded, with its message in the plugin console. Nothing is half
  registered, and no built-in is ever replaced.

**Trust.** Plugins are trusted Python code with your full user permissions. Error handling is
not a sandbox, and native crashes (CUDA, segfaults) cannot be isolated in-process. No plugin
is downloaded or enabled automatically.

**No hot reload.** Enabling or disabling a plugin takes effect after a restart.

## Regression evidence (2026-10-10)

The migration was checked against a frozen copy of the pre-migration commit (`1b035ff`).

**Exact equality on deterministic captures**, with wall-clock fields excluded:

- **CPU, tiny random-weight model:** legacy forward (0→2), self (1→1), backward (3→1) and
  synchronous multi-bypass. For each: two chat turns (token ids, per-token coupling records,
  reservoir state and time), both controlled comparisons, the persistence demo, idle dynamics,
  and Continuous Generation in `fly`, `random` and `fly_reset` conditions with an injection,
  the DA cue, a leak schedule, probes and full telemetry.
- **GPU, Qwen3-0.6B-Base:** the same battery for 6 → 14 legacy and for 2 → 2 + 26 → 26
  synchronous.

**Fixtures:**

- The historical golden regression and the real-model tests: 6 of 6, fixtures unmodified.
- The original interface reproduces the recorded projection fingerprints.

**Throughput** on Qwen3-0.6B-Base, RTX 3070 Ti, measured while another Observatory process
held the GPU. Runs alternated before and after:

| | Before | After |
|---|---|---|
| Clean+Fly pair, 2 × 128 tokens | 43.4–45.4 tok/s | 44.1–46.9 tok/s |
| Continuous Generation | 28.7–31.2 tok/s | 28.2–30.2 tok/s |

The two are within noise. One copy that grew with the new fields was removed from the
per-token path: the telemetry settings snapshot is now cached per configuration, and component
selections are not repeated in every snapshot.

## Known limitations

- **Model families.** Only the Qwen3 adapter exists. A new family needs its own adapter, and
  its profiles are not yet listed in the GUI model selector.
- **Connectome changes in the GUI** apply at the next launch. The 3D anatomy view works only
  when the connectome's neurons are the registered CX512 anatomy; otherwise a placeholder is
  shown. Headless, a connectome can be changed at any time (the reservoir restarts at zero).
- **Custom coupling policies** subclass `coupling.Coupling` and provide a hook plan. There is
  no declarative execution-plan language.
- **Modulation.** The kernel consumes only the built-in neuron cue. Other modulation
  components can be registered and used by experiments, but the kernel does not call them.
- **Diagnostics.** The projection analysis in `steering_diagnostics` needs an interface that
  exposes `readout` and `decoder`. Other interfaces are measured through the delta they apply.
- **Legacy CLI.** `experiments.py`, the v0.1 reproduction CLI, keeps its own small pipeline and
  is not routed through the registry.
- **Continuous dialog.** It reads cue-group names from the graph files for display only.
- **Hot reload** is not supported. There is no sandbox.
- **Test coverage.** The four-model GUI acceptance script (`scripts/gui_acceptance_models.py`)
  was not rerun after the migration. The window was checked with an offscreen real-model smoke
  test and the automated GUI tests.
