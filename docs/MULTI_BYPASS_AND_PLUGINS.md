# Multi-Bypass and Python plugins

Both features belong to the development version (0.3.0.dev0).

## Multi-Bypass

A **Neural Bypass** is one pair of transformer layers:

- **Capture:** the last-token hidden state of this layer is projected into the reservoir.
- **Steering:** this layer receives the residual derived from the reservoir state.

All bypasses share **one** FlyWire reservoir: the same 512 neurons, the same connectome, the
same projections and one state vector. Nothing is cloned per bypass.

### Using the panel

**MODEL COUPLING · NEURAL BYPASS** in the main window holds one row per bypass:
`[enabled] capture → steering [✕]`.

- **+ Add Bypass** appends a *disabled* row with the last row's layers. Set its layers, then
  enable it.
- **✕** removes a row. The last remaining row cannot be removed.
- Both layers lie in `0 … N−1`, where N is the loaded model's real layer count. Either order is
  allowed: forward (capture < steering), self (equal) or backward (capture > steering). The
  label beside each row shows which one it is.
- Below the list, the panel shows the active count and the coupling mode that will run.
- At least one row must be enabled. Otherwise nothing is applied.
- Edits apply between operations, never in the middle of a token. The controls are locked while
  generation or a plugin runs.
- Changing the active set rebuilds the coupling and drops the Fly KV cache. The reservoir state
  is kept, and the change is recorded as `layers_changed`.
- The rows are saved per model profile.

**Duplicate rows.** The executed configuration is the **sorted set of unique enabled pairs**.
Identical rows are applied once, and the panel says so. Row order never matters.

**Model switching.** The rows saved for the target profile are validated against the new
model's layer count after loading. If any row (enabled or not) references a layer that does
not exist:

- the switch fails with an error that names those rows;
- the previous model is restored;
- no pair is ever substituted.

Hooks are detached and verified absent before the old model is released.

### Two execution modes

| Active unique pairs | Mode (`coupling_mode`) | Behaviour |
|---|---|---|
| 1 | `legacy_single` | Exactly the established single bypass, unchanged; see the hook-order table in [MODELS.md](MODELS.md). The capture update and the feedback update are separate reservoir steps: two updates per token with feedback, one without. |
| ≥ 2 | `synchronous_multi` | Synchronous Multi-Bypass, described below. One reservoir update per token, whatever the number of bypasses. |

The single-bypass default, 6 → 14 on Qwen3-0.6B, therefore runs the original code path. The
golden regression is unchanged.

**Synchronous Multi-Bypass.** Each forward pass is one token. In that pass:

1. The first hooked layer to run takes a snapshot `x` of the shared reservoir state.
2. Every steering layer adds the established residual computed from that same `x`. The rule
   is unchanged: readout, decoder, centring and normalization, `strength × RMS(h)`.
3. Every capture layer projects its last-token activation to 64 features. On a layer that also
   steers, capture reads the output *before* the residual is added.
4. With feedback enabled, every steering layer projects its *steered* activation to 64
   features.
5. After the last hooked layer, the reservoir advances **once**:

   ```text
   u      = mean(capture features over unique capture layers)
          + feedback × mean(steered features over unique steering layers)
   x_next = (1 − leak)·x + leak·tanh(W·x + input_strength·U·u + cue)
   ```

   The means are taken in ascending layer order. `input_strength`, leak and gain are each
   applied once, by the unchanged reservoir equation. No clipping and no new gains are added.

There is one hook per unique layer, a layer-oriented dispatcher. A pass interrupted by an
error never reaches step 5, so the reservoir is not updated.

**Temporal semantics.** In synchronous mode, an observation influences steering **from the
next token on**, for forward, self and backward pairs alike. This is a deliberate one-token
delay. Synchronous forward coupling is therefore *not* equivalent to the legacy forward bypass:
there, capture updates the reservoir before steering within the same pass.

**What the pairs mean.** The inputs mix in one shared state, and one decoder serves every
steering layer: each steering layer gets the same direction, scaled by its own RMS. So only
the **set of capture layers** and the **set of steering layers** matter.

- `{2→10, 5→20}` and `{2→20, 5→10}` are mathematically identical.
- So are any reordering of rows and any duplicate row.

The tests check this to exact equality. The rows are attachment points, not isolated channels.

**Recorded with every session, run and plugin result.** Model information includes:

- `bypasses` (the rows) and `active_bypasses`;
- `coupling_mode`;
- `capture_layers` and `steer_layers`;
- `reservoir_updates_per_token` and `synchronization` (the description above);
- `coupling_order`: the legacy pair order, or `"synchronous"`.

**Per token.** Telemetry and traces record:

- the layers that fired, as lists in synchronous mode;
- per-layer capture RMS and steering metrics;
- `reservoir_updates`.

**Restoring.** A live session restores only into the same active bypass set. Recordings made
before Multi-Bypass restore as single bypasses. Exact replay of Continuous Generation from a
schedule (`--inject`, `--leak-at`) remains a single-bypass CLI feature. The CLI has no
multi-bypass option.

## Python plugins

A plugin is an ordinary `.py` file that defines `run(api)`:

```python
def run(api):
    api.log("Plugin started")
```

**Trusted code.** A plugin runs inside the Observatory process with **your full user
permissions**. There is no sandbox. Run only scripts you wrote or have read. Nothing is
discovered or executed automatically.

**Running.**

1. Click **Experiments → Run Python Script…** and choose a file.
2. The script runs on the engine worker thread. It owns the engine until it returns:
   - generation, model loading and coupling controls are locked;
   - the window stays responsive;
   - messages appear in the plugin log under the button.

**Stopping.** **Stop script** (or Cancel) requests cooperative cancellation:

- a running generation stops at the next token;
- the next `api.check_cancelled()` raises and ends the script.

Python code that never checks cannot be forcibly stopped.

**When the script fails.** A script error does not stop the Observatory:

- the traceback appears in the plugin log;
- the engine is released and the controls are restored;
- any forward hook left behind by the script is removed;
- the start and end of every script are recorded in the session (`plugin_started`,
  `plugin_finished`, with the script's SHA-256).

Afterwards, the panel shows the bypass configuration the script left. If the script used the
chat histories, the chat panes are cleared.

### API

| Call | Purpose |
|---|---|
| `api.set_bypasses([(c, s), ...])` | Replace the bypass list. Rows may also be `(c, s, enabled)`. It uses the same validation and the same configuration object as the panel. Returns the executed configuration (`coupling_mode`, `active_bypasses`, `reservoir_updates_per_token`, …). |
| `api.set_coupling(strength=, feedback=, steering_enabled=, feedback_enabled=, leak_rate=)` | Coupling settings, the same as the GUI controls. |
| `api.new_session()` | Clean start without reloading weights: empty Clean and Fly histories, invalidated KV caches, a zeroed reservoir, cleared coupling traces and counters, and a new session recording. |
| `api.reset_state()` | Zero the reservoir and drop the Fly cache; the histories are kept. |
| `api.generate(prompt, mode="fly", max_tokens=None)` | One turn through the chat path the GUI uses (greedy). `mode` is `"fly"`, `"clean"` or `"both"`; `"both"` returns `{"clean": …, "fly": …}`. The turn is appended to that branch's history. Returns a `GenerationResult` (below). |
| `api.run_continuous(name=None, **ContinuousConfig fields)` | One Continuous Generation condition, for example `seed_text`, `total_tokens`, `window`, `condition`, `decoding="sample"`, `temperature`, `top_p` or `sampling_seed`. This is the way to use sampling. Output goes into the plugin's run directory. Returns the summary, with `output` set. |
| `api.get_config()` | A copy of the settings, model information and executed coupling. |
| `api.get_reservoir_state()` | A copy of the shared reservoir: a read-only `state` array, `rms`, `simulation_time`, `leak`, `fingerprint`. |
| `api.save_result(name, result, extra=None, overwrite=False)` | Write `<run dir>/<name>.json`. It contains the result plus the model, coupling, settings, reservoir summary and session id. Existing files are never replaced unless `overwrite=True`. |
| `api.log(message)` | Add a line to the plugin log. |
| `api.is_cancelled()`, `api.check_cancelled()` | Cooperative cancellation. |
| `api.output_dir` | The run directory. You may write other files there with ordinary Python. |

**`GenerationResult`** has these fields:

- `branch`, `prompt`, `text`;
- `token_ids`, `prompt_ids`;
- `stop_reason`, `seconds`, `tokens_per_second`, `cancelled`;
- `decoding`, `coupling`, `model`;
- `telemetry`: per-token coupling records, Fly branch only;
- and `to_dict()`.

**Output.** Each script run gets its own directory, `results/plugins/<script name>/<start
time>/`. It is never reused. It contains the saved results and `plugin_run.json` (script path,
SHA-256, status, saved files).

### Examples

- `plugins/examples/bypass_sweep.py` sweeps capture layers 1…6 into steering layer 10. For each
  value it starts a new session, generates on the Fly branch, saves the result and logs it, and
  it honours Stop.
- `plugins/examples/two_bypasses.py` runs Clean and Fly with two simultaneous bypasses,
  2 → 10 and 20 → 5.
