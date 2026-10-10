# Research plugins: leak dynamics and a shadow dopamine field

Two trusted plugins in `plugins/examples/`. Run them with **Experiments → Run Python
Script…**. All settings sit at the top of each file. Neither plugin changes:

- the reservoir equation;
- the coupling or the projections;
- decoding;
- the existing DA cue.

The leak plugin sets the leak rate through the same configuration as the RESERVOIR DYNAMICS
control. The dopamine plugin is **shadow mode**: it reads recorded states and never writes to
the reservoir. It verifies this on every run and raises an error if the state changed.

Both run in the Observatory's engine worker, on the model already loaded. They never start a
second model instance. Stop requests cooperative cancellation. Each run writes to a new
directory and never overwrites an earlier one.

## 1. Leak Rate Dynamics — `leak_rate_dynamics.py`

**Purpose.** Separate four things that change together with the leak rate:

- how the reservoir state develops and how long it is retained;
- the **actual** steering size;
- the steering direction;
- the measured effect on Qwen's next-token distributions.

**Settings.**

| Setting | Meaning |
|---|---|
| `MODEL` | Profile key, or `None` to keep the loaded model. A profile that is already loaded is never reloaded. |
| `BYPASSES` | Bypass pairs; `None` keeps the current ones. The default (2,2), (34,34) needs the 36-layer 4B model. |
| `STEERING_STRENGTH`, `FEEDBACK_STRENGTH` | Configured values. |
| `LEAK_RATES` | Run in sequence, with no model reload. |
| `MODE` | `"forced_tokens"`: controlled, the same ids for every leak and for Clean and Fly. `"free_generation"`: Fly picks its own greedy tokens. |
| `PROMPT`, `MAX_TOKENS`, `FORCED_TOKENS` | Measurement input. Without `FORCED_TOKENS`, the first run's Clean greedy continuation is reused for every leak. |
| `WARMUP_TOKENS`, `WARMUP_PROMPT` | Optional fixed forced warm-up before the measurement. It is the same id sequence for every leak and is stored as phase `warmup` with index < 0, not counted as measurement. |
| `IDLE_STEPS` | Optional idle reservoir updates after the measurement: no input, dt simulation time each. These are **not tokens**. |
| `RESET_BETWEEN_RUNS` | `new_session()` before each leak (zero state, empty histories and caches). If `False`, runs continue from the previous state and are marked as not independent. |
| `COUNTERFACTUAL` | Per-token Fly-off run on a copy of the Fly cache: the current-token steering effect. |
| `SAVE_VECTORS`, `SAVE_PLOTS` | Optional outputs. |

**Measured per token** (`token_metrics.csv`, one row per leak, phase and index):

- **Reservoir:** RMS, norm and change norm; consecutive cosine and cosine to the reference.
- **Steering:** norm, applied delta norm and **applied ‖Δ‖/‖h‖**; consecutive cosine and cosine
  to the reference; parallel and orthogonal components; whether normalization was active.
- **Distributions (forced mode):** KL(Clean‖Fly), immediate KL(Fly-off‖Fly), history
  KL(Clean‖Fly-off), top-1 changes.
- **Free mode:** only the immediate KL(Fly-off‖Fly). Clean and Fly then have different
  histories, so no Clean-vs-Fly logit metric is computed.
- The leak rate and the reservoir update count per token.

**Other outputs.**

- `summary.csv`, one row per leak:
  - medians of the measured quantities;
  - configured strength next to the actual relative intervention;
  - number of steps with normalization active;
  - top-1 changes;
  - idle half-life (idle steps and simulation time until ‖x‖ halves);
  - repetition: distinct-1/2, self-repeat rate, best period;
  - the start of the Fly text.
- `matched_amplitude.json` — offline only:
  - geometry of the recorded steering vectors scaled to unit norm;
  - same-step cosines between leak conditions, meaningful in forced mode.

  This is **not** a matched-amplitude experiment. Rescaling recorded vectors cannot reproduce
  a coupled run, because the applied delta also drives feedback and therefore future states. A
  causal matched-amplitude run would need a separate controlled intervention mechanism in the
  coupling, which is not implemented.
- `median_kl_immediate_per_rel2` in the summary is descriptive. For small perturbations, KL
  grows roughly with the square of the perturbation. It is not a causal normalization.
- `reservoir_states.npz` and `steering_vectors.npz` hold all leak rates, keyed
  `leak_<value>__<name>`.
- Per-run probe folders `leak-<value>/{warmup,measure}/`.
- `plots/`:

  | Plot | Content |
  |---|---|
  | 01 | Reservoir RMS |
  | 02 | Applied relative intervention |
  | 03 | Consecutive steering cosine |
  | 04 | Steering cosine to the reference |
  | 05 | State change norm |
  | 06 | KL(Clean‖Fly) |
  | 07 | Immediate KL |
  | 08 | Idle retention |

  The warm-up boundary is a dashed line at index 0.

**Reading it.**

- The configured steering strength is not the applied size. While RMS(centered) < 1 the
  normalization is inactive, and ‖Δ‖/‖h‖ scales with the reservoir state.
- A leak effect that persists in the immediate KL *relative to* the applied size is the
  interesting signal. An effect that tracks the size alone is the amplitude confound.
- The legacy single bypass updates the reservoir twice per token, the synchronous multi-bypass
  once, so equal leak rates are different per-token time scales.
- Idle decay describes the state without input. It is not evidence of memory during active
  generation.

Saved metadata, abridged:

```json
{"mode": "forced", "leak_rates": [0.005, 0.2], "warmup_tokens": 4, "idle_steps": 10,
 "reset_between_runs": true, "independence": "new_session() before every leak rate",
 "coupling": {"coupling_mode": "synchronous_multi", "active_bypasses": [[2, 2], [26, 26]],
              "reservoir_updates_per_token": 1},
 "phases": {"warmup": "fixed forced token sequence (same ids for every leak), index < 0 …",
            "idle": "reservoir updates with no input …; each idle step is dt simulation time, not a token"},
 "configured_vs_actual": "configured steering strength is not the applied size …"}
```

## 2. Spatial Dopamine Field Prototype — `dopamine_field_prototype.py`

### Anatomical data audit (CX512, verified)

Every run writes this audit to `data_availability.json`.

| Item | Present | Detail |
|---|---|---|
| Neuron ids | yes | 512 FlyWire root ids (materialization 783) |
| Neurotransmitter | predicted | ACH 220, GABA 123, GLUT 93, DA **16**, SER 6, unassigned 54. These are Codex classifier predictions with scores; for DA the scores are 0.41–0.73. |
| Coordinates | 1 per neuron | Community-marked positions (1,250 markers in total), FlyWire/FAFB14.1, nm. **Not soma centroids and not release sites.** |
| Skeletons | 16 neurons | The first 16 in graph order. **None of the DA neurons.** |
| Connectivity | yes | Synapse counts per directed pair. |
| Synapse or release-site positions | **no** | |
| DA axon or neurite geometry | **no** | |
| Receptor expression | **no** | |

**Consequence.** A dopamine field grounded in real release sites cannot be built from this
repository. The plugin offers a **marker-proxy** illustration: sources sit at the DA neurons'
marker points, inside a real coordinate frame. The field is explicitly not a reconstruction.

A grounded field would need, at least:

- DA presynaptic (T-bar) coordinates from the FlyWire synapse table;
- DA neuron skeletons or meshes;
- per-neuron receptor expression;
- extracellular volume fraction and tortuosity, for an effective D;
- DAT density, for clearance.

If any DA neuron lacks coordinates, the plugin writes only the audit and stops. It never
creates geometry.

### Model

- **Activity.** A short Continuous Generation run on the loaded model, optionally with the
  existing DA cue and one injection, saved with one state per token. Alternatively, an existing
  run directory saved with `snapshot_interval=1`.
- **Release proxy.** `target_i = g(gain · (x_i − threshold))`, with `g` = ReLU or softplus,
  for the 16 DA-classified neurons.
  - Reservoir values are signed model activations, not firing rates. Negative values release
    nothing.
  - Release has its own kinetics: `dr/dt = (target − r)/τ`.
- **Field.** `dC/dt = D∇²C − kC + S` on a voxel grid around all 512 markers, with a
  configurable margin.
  - Explicit finite differences, 7-point Laplacian.
  - Boundary: no-flux (Neumann) or absorbing.
  - Each token step is a fixed field time (`TOKEN_INTERVAL_S`), split into equal substeps with
    `dt·(6D/h² + k) ≤ cfl < 1`. This keeps every update coefficient nonnegative, so C ≥ 0.
  - The timing never depends on the number of hooks.
  - After the activity, `POST_SECONDS` of source-off integration measures spreading and
    clearance.
- **Units.** Lengths in µm (from FlyWire nm), time in seconds of *field time*, concentration in
  uncalibrated release units per µm³. D, k, the gain and the token-to-time mapping are
  illustrative settings, not measured constants.
- **Hypothetical leak.**
  - `alpha_i = f(C_i)` uses a smooth tanh curve (or linear then clipped) around a baseline
    concentration, with a sensitivity setting. Direction +1 raises leak with concentration,
    −1 lowers it, within `[LEAK_MIN, LEAK_MAX]` and anchored at `LEAK_BASELINE`.
  - **It is never applied.** No receptor identities are assigned.

### Outputs

The run directory is `results/plugins/dopamine-field/<time>/`. It contains:

- `data_availability.json`;
- `metadata.json`: settings, grid, substeps, positivity limit, minimum concentration, mass
  balance error (no-flux), boundary loss, leak range, spread by distance, clearance half-time
  and its k-only expectation;
- `field_series.csv`, per step:
  - release;
  - total amount, max, mean, variance;
  - mean C at DA markers and at receivers;
  - gradients;
  - phase;
  - whether the token was cued;
- `field_arrays.npz`: C and hypothetical leak at every neuron per step, release rates,
  positions, and z-max projections at snapshot times;
- `distance_vs_graph.json`: spatial distance to the nearest DA marker next to synaptic graph
  hops. These are different quantities; graph hops are not diffusion;
- `activity-run/`: the source Continuous Generation run.

**Plots** (`plots/`):

| Plot | Content |
|---|---|
| 01 | DA release proxy |
| 02 | Concentration evolution |
| 03 | Spatial snapshots: z-max projections with all markers and DA markers |
| 04 | Gradients, and peak C against distance |
| 05 | Hypothetical leak percentiles over neurons |
| 06 | The configured response curve and the concentration range reached |
| 07 | Spatial distance against graph hops |

**Not produced, because the data are missing:**

- release-site-resolved maps;
- receptor-specific responses;
- anything along neurites.

**Known approximation.** The clearance half-time includes the release decaying with τ after
the stop and a 0.1 s sampling step. It therefore exceeds ln2/k when τ > 0.

## Engineering checks performed

**Unit tests** (`tests/test_research_plugins.py`, tiny fixtures):

- No-flux mass conservation with k = 0.
- Exact exponential clearance.
- Spreading is gradual, not instantaneous.
- Absorbing boundary loss.
- The positivity limit is enforced.
- C stays nonnegative even with negative activations.
- The leak response stays within its limits in both directions.
- Anatomy audit on the real data: 16 DA neurons, 512 positions, no DA skeletons, no synapse
  positions.
- Missing anatomy is handled.
- Shadow mode leaves the state unchanged.
- The leak plugin:
  - resets between runs and reuses the forced tokens;
  - does not reload the model;
  - writes all outputs;
  - handles warm-up indexing and free mode.

**On the GPU.** A short smoke run of both plugins on Qwen3-0.6B-Base:

- leak plugin: 2 leak rates × 16 tokens, plus warm-up and idle steps;
- dopamine plugin: 60 tokens with the DA cue.

These are smoke checks, not results.
