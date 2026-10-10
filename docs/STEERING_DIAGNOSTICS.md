# Steering diagnostics: direction and persistence of the reservoir's influence

This is a measurement instrument. It changes nothing in the coupling, and its output is
identical whether it is on or off (this is tested). This document covers what the instrument
records, how to run it, and a small preliminary pilot.

## What is measured

The steering rule, as implemented in `llm.Coupling._intervene`, for reservoir state `x` (512):

```text
compact   = x @ readout                       readout 512×64, fixed (seed 42), rank 64
raw       = compact @ decoder                 decoder 64×hidden, fixed (seed 42), rank 64
centered  = raw − mean(raw)
direction = centered / max(RMS(centered), 1)
delta     = strength · RMS(h_last) · direction    added to the last position only
```

There is no nonlinearity between `x` and `delta`. **Steering vector** `v_t` means `centered`:
every later step multiplies it by a positive scalar, so its direction is the direction of the
applied delta. The direction of the delta actually added in the model dtype is measured
separately (`applied_vs_steering_cos`).

`centered` always lies in the image of the centered decoder, at most 64 of the hidden
dimensions. The readout sees only a 64-dimensional subspace of the 512 reservoir dimensions;
the remaining 448 form its null space.

**How it is collected.** The coupling has an optional `observer` attribute, `None` by default.
When it is set, the hooks pass float64 CPU copies of the tensors they actually compute to the
observer: the captured hidden state, the features, the state used for steering, compact,
raw, centered, direction, the hidden state before and after steering, the feedback input, and
the synchronous aggregate input. The observer never writes to engine tensors.

**How it runs.** `steering_diagnostics.run_probe` drives tokens through the engine's own
`_forward`, with the live coupling and reservoir:

1. **Clean.** The prompt plus a token sequence, uncoupled. The sequence is Clean's greedy
   continuation, unless `forced_tokens` is given.
2. **Fly.** The *same* token ids, coupled. The reservoir evolves legitimately through them.
3. **Fly-off** (immediate counterfactual). At each step, the same input runs uncoupled on a
   copy of the Fly KV cache: Fly's history, no steering at this token, no reservoir update.
   - `KL(Fly-off‖Fly)` is the **immediate** effect of this token's steering.
   - `KL(Clean‖Fly-off)` is the effect **carried in the cache** from earlier interventions.

   The KV cache is copied with `copy.deepcopy`. There is no other change to inference.

**Step alignment.** Step k is forward pass k.

- Pass 0 is the prompt prefill. Pass k ≥ 1 feeds token k−1.
- The logits of pass k predict token k.
- The hooks act on the last input position of the pass.

Three states are recorded per step:

| State | Meaning |
|---|---|
| `state_before` | The state at the start of pass k. |
| `state_used` | The state the steering rule read. In legacy forward mode this is after the token's capture update. In legacy backward mode and in synchronous mode it equals `state_before`. |
| `state_after` | The state at the end of pass k, which is the input to pass k+1. |

**Per-token columns** in `token_metrics.csv`:

- **Reservoir:** RMS, norm, change norm, cosine to the previous state and to the reference
  state, and the fraction of the state, and of its change, that lies in the readout's subspace.
- **Readout:** cosines of the 64-dimensional vector.
- **Steering:** norm, RMS, whether normalization was active (RMS > 1), cosine to the previous
  vector and to the reference, parallel and orthogonal components with their ratio, relative
  norm change, applied delta norm, hidden-state norm, and relative intervention.
- **Inputs:** capture and feedback norms, and the cosine of consecutive reservoir inputs.
- **Logits:** KL, Jensen–Shannon, top-1 and whether it changed, top-1 probability and margin,
  logit-difference L2, and the immediate and history terms.

**Conventions.**

- A cosine involving a vector with norm ≤ 1e-12 is **undefined**. It is stored as NaN, empty in
  the CSV, and marked on the plots; it is never set to 0 or 1.
- The reference vector is the first nonzero vector.

**Offline geometry** in `metadata.json` → `geometry`:

- **Centered PCA** gives the explained-variance ratios, effective rank (exp of the entropy) and
  participation ratio. It shows *how many directions the variation spans*.
- **Uncentered top-1 energy fraction** is the share of Σ‖v‖² along one direction. A value of 1
  means one fixed direction.

A trajectory can be low-dimensional without being fixed-direction. Rank estimates from about
48 samples are coarse.

## Running it

**From the GUI.** Edit the settings at the top of `plugins/examples/steering_dynamics_probe.py`,
then use **Experiments → Run Python Script…**. The settings are:

- model profile;
- bypasses;
- steering strength and feedback;
- the leak rates to compare (default 0.005 and 0.200);
- prompt and number of tokens;
- optional forced token ids.

For each leak rate, the probe starts a new session (zero reservoir) and uses the same token
sequence. Output goes to `results/plugins/steering_dynamics_probe/<time>/`, with one directory
per leak rate:

- `metadata.json`, `summary.json`, `summary.md`;
- `token_metrics.csv`;
- `reservoir_states.npz` and `steering_vectors.npz`;
- `plots/01…07`.

A `comparison.md` sits next to them. Nothing is ever overwritten.

**Outside the GUI.** `scripts/steering_probe_pilot.py` runs the same plugin. Its settings can be
overridden: `--profile`, `--bypasses 2:2,26:26`, `--leaks`, `--tokens`, and `--legacy 6:14` for
one extra single-bypass run on the same tokens.

**From your own plugin.** Call `api.steering_probe(prompt, max_tokens, name, forced_tokens,
counterfactual)`. To switch models from a plugin, use `api.load_model(profile, bypasses)`.

**Leak rate and time scale.** The legacy single bypass updates the reservoir twice per token;
the synchronous mode updates it once. Equal leak rates are therefore *not* equal per-token time
scales across the two modes. Every output records the update count.

## Preliminary pilot (2026-10-10)

**Not the requested model.** The pilot ran on **Qwen3-0.6B-Base**. The planned Qwen3-4B-Base
did not fit on the GPU, because the Observatory window held 3.2 GB with 4B loaded. The plugin's
defaults are the 4B configuration (2→2, 34→34), so it can be run from that window directly.

**Setup.**

- Synchronous bypasses 2→2 and 26→26: the same positions in a 28-layer stack, first and
  next-to-last layer.
- Strength 0.90, feedback 0.65, sampling not used.
- Prompt «Однажды в студёную зимнюю пору»; 48 forced tokens, Clean's greedy continuation.
- Zero initial state.
- Leak rates 0.005 and 0.200.
- One extra legacy single-bypass run, 6→14 at leak 0.200, on the same tokens.

**Output.** `results/steering-dynamics-pilot/20261010-203252-qwen3-0.6b-base/`.

This is one prompt, one model and 48 tokens. It is an engineering pilot, not evidence.

### Direct measurements

The first table is per token; it gives medians over the steps.

| Run | state cos(t,t−1) | steering cos(t,t−1) | steering cos to reference (last) | orthogonal / total | relative intervention ‖Δ‖/‖h‖ | KL(Clean‖Fly) | immediate KL(Fly-off‖Fly) | history KL(Clean‖Fly-off) | top-1 changed |
|---|---|---|---|---|---|---|---|---|---|
| sync, leak 0.005 | 0.9987 | 0.9992 | 0.84 | 0.56 | 0.027 | 0.0018 | 0.0019 | 0.0010 | 3/48 |
| sync, leak 0.200 | 0.9786 | 0.9863 | 0.75 | 0.63 | 0.27 | 0.18 | 0.15 | 0.0079 | 10/48 |
| legacy 6→14, leak 0.200 | 0.9619 | 0.9762 | 0.36 | 0.92 | 0.22 | 0.021 | 0.0083 | 0.0058 | 9/48 |

The second table gives the offline geometry over the 48 steps.

| Run | steering vectors: uncentered top-1 energy | steering: centered effective rank | reservoir change vectors: effective rank |
|---|---|---|---|
| sync, leak 0.005 | 0.990 | 1.2 | 22 |
| sync, leak 0.200 | 0.901 | 7.6 | 23 |
| legacy 6→14, leak 0.200 | 0.835 | 8.6 | 21 |

**Facts beyond the tables.**

- **Normalization was never active.** RMS(centered) stayed below 1 in every step of every run,
  so `delta = strength · RMS(h) · centered`. The steering magnitude is proportional to the
  reservoir state's magnitude, and the configured strength 0.9 is never reached:
  - at leak 0.005 the state RMS is about 0.03 and the relative intervention about 2.7%;
  - at leak 0.200 the state RMS is about 0.26 and the relative intervention about 27%.
- **The projection preserves direction.**
  - Steering cosines track the 64-dimensional readout cosines almost exactly (plot 06).
  - The centered decoder has rank 64 and condition number 1.6.
  - The readout captures 34–38% of the norm of each state change, close to √(64/512) ≈ 0.35,
    the value expected for a random 64-dimensional subspace. No sign of selective suppression
    or enhancement.
- **One direction for all steering layers.** In synchronous mode the steering direction is
  identical across steering layers (maximum deviation 0).
- **Influence persists.** At leak 0.200 (sync), KL(Clean‖Fly) stays measurable through all 48
  steps; its median in the second half is 0.20 against 0.18 in the first. Most of it is
  immediate: the Fly-off vs Fly term follows the total closely (plot 07). The history term is
  one to two orders of magnitude smaller.
- **Precision at low leak.** At leak 0.005 the applied bf16 delta deviates from the computed
  direction on the earliest steps (minimum cosine 0.886). There the delta is tiny compared with
  the hidden state, so rounding distorts its direction.

### Supported inferences

- **Is the direction nearly fixed?** In this pilot, yes over short spans: consecutive steering
  vectors are almost parallel, with median cosine 0.986–0.999. At leak 0.005 the direction is
  close to a single fixed vector (uncentered energy 0.99). It still drifts slowly: the cosine
  to the first vector falls to 0.84 at 0.005 and to 0.75 at 0.200.
- **Where the stability comes from.** It originates in the **reservoir state**, not in the
  projection (Case A). The states are as directionally stable as the steering vectors, and the
  linear projection carries their cosines through. The state *changes* are high-dimensional
  (effective rank about 22). The *state* is dominated by its slowly accumulated component, and
  more strongly so at low leak.
- **Why low leak gives a weak effect.** Low leak mainly makes the intervention **small**, not
  just stable. Because normalization never engages, the magnitude is proportional to state
  RMS, about 10 times smaller at leak 0.005. The distribution effect is about 100 times smaller
  (KL 0.002 against 0.18). In this pilot, a weak, almost constant bias at low leak is the
  measured situation.
- **Legacy versus synchronous.** The legacy 6→14 run drifts much further from its reference
  (0.36 against 0.75). It also has a smaller immediate KL at a similar relative intervention.
  Its layers and update count differ, so this is not a like-for-like comparison.

### Open hypotheses (not established)

- **Low leak and repetition.** That low leak causes repetition through a fixed steering
  direction is not tested here. These runs used forced tokens, so repetition could not appear.
  The measured low-leak effect is mainly weak. Whether weak plus fixed or merely weak matters
  needs free-generation runs at matched intervention magnitude.
- **Topology.** Whether any of this depends on the FlyWire topology is not tested. That needs
  the random-topology control, which is available as condition `random` in Continuous
  Generation.
- **Generality.** Whether the 4B model, other prompts and longer runs behave the same is
  unknown.
- **What is not shown.** A high cosine does not mean no influence. At leak 0.200, steering
  changes the top-1 token on 10 of 48 steps despite consecutive cosines of about 0.99.

**Limitations.**

- The counterfactual removes steering at one token but keeps the earlier Fly cache. It does not
  remove the reservoir's own history.
- Clean's tokens are greedy; the Fly branch is forced onto them.
- Step 0 of the synchronous mode reads the zero state, so its direction is undefined. The
  reference is the first nonzero vector, step 1.
