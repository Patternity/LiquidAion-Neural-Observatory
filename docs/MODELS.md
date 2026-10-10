# Language models and coupling layers

LiquidAion couples one FlyWire reservoir to an interchangeable Qwen3 language substrate.

- **The reservoir is fixed.** It always has 512 neurons, the same 16,731 FlyWire edges, the
  same dynamics and the same 64-feature input interface, whichever model is loaded.
- **The operator chooses three things:** the model, the layer to observe (capture), and the
  layer to steer.
- **Everything else is automatic:** the engine derives every tensor dimension from the loaded
  model.

## Profiles

Defined in `src/liquidaion_fly/models.py`. Revisions are immutable Hugging Face commits.

| Profile | Model | Type | Revision | Loading | Attention | Default capture → steering |
|---|---|---|---|---|---|---|
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | instruct (chat) | `c1899de289a04d12100db370d81485cdf75e47ca` | float16 (published baseline, unchanged) | eager | 6 → 14 |
| `qwen3-0.6b-base` | Qwen/Qwen3-0.6B-Base | base | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | bfloat16 | sdpa | 6 → 14 |
| `qwen3-1.7b-base` | Qwen/Qwen3-1.7B-Base | base | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | bfloat16 | sdpa | 6 → 14 |
| `qwen3-4b-base` | Qwen/Qwen3-4B-Base | base | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | 4-bit NF4 (bitsandbytes), compute bfloat16 | sdpa | 8 → 18 |

**Why these precisions.** The original profile keeps its exact loading path (float16, eager
attention), so the published results and the golden regression reproduce bit for bit. The
Base models were trained in bfloat16. The RTX 3070 Ti (Ampere, compute 8.6) supports
bfloat16 natively (`torch.cuda.is_bf16_supported()`), so they run in it. Qwen3-4B-Base needs
about 8 GB in 16-bit, which is the entire card. It is therefore loaded as 4-bit NF4
(`bitsandbytes` 0.48.2, no double quantization), with compute in bfloat16.

- The quantization backend places the weights itself; the model is never moved with `.to()`.
- Loading fails if any part lands off the GPU. There is no silent CPU offload.
- NF4 is unavailable on CPU, and the 4B profile refuses to start there rather than substituting
  another model.

**Default layers.** 6 → 14 is the original pair. For the 36-layer 4B model, 8 → 18 places the
two hooks at the same relative depths (6/28 and 14/28 of the stack). These are starting
points, not tuned values.

## Base versus Instruct

**Qwen3-0.6B is instruction-tuned.**

- The Classic Observatory keeps its original chat protocol for it: the Qwen3 chat template
  with `enable_thinking=False`.
- Continuous Generation may use the pinned "Продолжи текст." prefix.

**The Base models are plain pretrained causal language models** and are used as a raw
generative substrate.

- **Classic Observatory.** Each branch's history is one raw text. Every submission and every
  model reply is appended exactly as written, and the model continues it. There is no chat
  template, role marker, system prompt, persona or instruction; the tokenizer is called with
  `add_special_tokens=False`. The GUI shows "Interaction: Raw continuation / Base".
- **Continuous Generation.** Raw continuation is the only policy available: `chat_prefix` is
  rejected for Base profiles. A seed such as «Однажды в студёную зимнюю пору» is tokenized as
  plain text and continued.

## Choosing capture and steering layers

The **MODEL COUPLING** panel in the main window exposes the two existing attachment points.
No other hooks exist.

- **Capture layer.** The last-token hidden state at the output of this decoder layer is
  normalized and projected to 64 reservoir features.
- **Steering layer.** The reservoir-derived residual is added to the last-token hidden state
  at the output of this layer. The modified activation also drives the feedback update.

**Rules.**

- Indices are zero-based. For a model with N layers, capture lies in 0…N−2 and steering in
  capture+1…N−1.
- The spin boxes take these limits from the loaded model's real layer count. Steering moves
  up automatically if capture passes it.
- Invalid pairs are rejected by the engine as well.

**When changes apply.**

- Changes are allowed only when the engine is idle; the controls are disabled during
  generation.
- A change applies before the next operation. The panel shows "Pending" until the engine
  confirms "Applied".
- The reservoir state is kept, and the Fly KV cache is discarded.
- Every change is recorded as `layers_changed`.

**A change of layers is a change of experimental condition.** The same pair is not
scientifically equivalent across models: the layer count, width and representation differ.
Nothing is selected automatically, and no layer choice is optimized for the quality of the
output.

## Why dimensions adapt automatically

The coupling uses three fixed random projections, all drawn from seed 42 with the original
algorithm:

| Projection | Shape |
|---|---|
| Encoder | hidden × 64 |
| Readout | 512 × 64 |
| Decoder | 64 × hidden |

`hidden` is the model's `hidden_size`. It is the same at every standard decoder block of a
model, because the residual stream keeps one width throughout:

| Model | Hidden size | Decoder layers |
|---|---|---|
| Qwen3-0.6B, 0.6B-Base | 1024 | 28 |
| Qwen3-1.7B-Base | 2048 | 28 |
| Qwen3-4B-Base | 2560 | 36 |

So:

- **Moving the hooks within one model** never changes the projections. This is verified: the
  projection fingerprint is identical before and after a layer change.
- **Switching to a model with another width** rebuilds them for that width.

The hidden size is unrelated to the sequence length and to the sliding window. Dimensions are
read from the loaded model's configuration, never from the profile name. Every recording
stores the projection fingerprint.

## Switching models

Select a profile under **LANGUAGE MODEL** and press **Load**. A confirmation explains the
consequences. Then the engine:

1. Accepts the request only when idle; the controls are disabled during any operation.
2. Detaches hooks, verifies that none remain, and releases the previous model and its caches
   (`gc.collect()`, `torch.cuda.empty_cache()`). Only one model occupies the GPU.
3. Loads the new model at its pinned revision. A model is downloaded once into the standard
   Hugging Face cache on first use; nothing is downloaded at startup.
4. Applies the operator's last layer pair for that profile, or the profile defaults.
5. Rebuilds the projections, starts a **zero reservoir state**, clears both chats and starts a
   new recording.

The reservoir state was driven by the previous model's activations, so it is never carried
over. The connectome itself is untouched. If loading fails, the real error is shown and the
previous model is reloaded. No other model is ever substituted.

The preferred profile and each profile's layer pair are saved with `QSettings`. Without a
saved choice, the original Qwen3-0.6B with 6 → 14 is used.

Continuous Generation always shows the main window's active model and layers. It has no
selector of its own. Its runs are stored under
`results/continuous/gui/<profile>-c<capture>-s<steer>/`, so runs of different substrates
never collide.

## Recording and replay

**Recordings store:**

- model information: profile, model id, revision, dtype, quantization, attention, layer count,
  hidden size, vocabulary, maximum positions, EOS ids;
- capture and steering layers, and the projection fingerprint;
- connectome hashes;
- reservoir dynamics and leak;
- cue configuration;
- decoding settings and seeds;
- every injection with its token ids.

**Rules for restoring and replaying:**

- A live session restores only into the same profile, revision and layer pair. Anything else
  is rejected.
- Offline replay never loads a model.
- Token-exact comparisons are meaningful only within one profile and loading path, never
  across different weights, quantization or architectures.

## Returning to the original baseline

Choose **Qwen3-0.6B (Instruct)** and set capture 6 → steering 14 (its defaults), with leak
0.200. Alternatively, start without saved preferences or run
`--model-profile qwen3-0.6b`. The golden regression (`tests/test_golden_fly_question.py`)
reproduces the published Clean (66 tokens) and Fly (59 tokens) replies exactly under this
profile.

## Measured on the RTX 3070 Ti (2026-10-10)

Measurements come from `scripts/model_acceptance.py`, with per-profile output in
`results/model-acceptance/<profile>/`.

- PyTorch 2.8.0+cu128, transformers 4.56.2, bitsandbytes 0.48.2, accelerate 1.10.1.
- Every profile ran entirely on `cuda:0`, with no CPU offload, from its pinned revision.
- Speeds are for sampled Continuous Generation: window 128, 200 tokens, T=1.0, top-p 0.95,
  seed 42.
- Peak memory was measured during the Fly run.

| Profile | Layers / hidden | Weights | Load (cached) | Allocated after load | Peak allocated / reserved | Clean tok/s | Fly tok/s |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 / 1024 | float16 | 0.9 s | 1.11 GiB | 1.16 / 1.44 GiB | 37.0 | 32.8 |
| Qwen3-0.6B-Base | 28 / 1024 | bfloat16 | 0.5 s | 1.11 GiB | 1.16 / 1.43 GiB | 39.5 | 35.6 |
| Qwen3-1.7B-Base | 28 / 2048 | bfloat16 | 0.9 s | 3.21 GiB | 3.26 / 3.59 GiB | 36.2 | 31.7 |
| Qwen3-4B-Base | 36 / 2560 | NF4, compute bfloat16 | 6.5 s | 2.65 GiB | 2.72 / 2.88 GiB | 15.1 | 13.6 |

**Steering effect on identical forced tokens.** Same-prefix Clean vs Fly comparison, 24
tokens, strength 0.25, feedback 0.25. The second pair of each model is an alternative tested
on purpose:

| Profile | Pair | Mean KL(Clean‖Fly) | Argmax disagreement | ‖Δh‖/‖h‖ |
|---|---|---|---|---|
| Qwen3-0.6B | 6 → 14 / 14 → 26 | 3.3e-3 / 5.5e-3 | 0% / 4% | 4.9% / 4.6% |
| Qwen3-0.6B-Base | 6 → 14 / 14 → 26 | 2.5e-3 / 8.0e-3 | 8% / 17% | 4.7% / 4.6% |
| Qwen3-1.7B-Base | 6 → 14 / 14 → 26 | 1.6e-3 / 4.2e-3 | 0% / 0% | 4.2% / 4.4% |
| Qwen3-4B-Base | 8 → 18 / 18 → 34 | 8.3e-4 / 4.0e-3 | 0% / 4% | 3.6% / 4.1% |

**Observations.** At a matched relative intervention (about 4–5% of the hidden-state RMS), a
later steering layer moves the output distribution more in every model. At the default pair,
the per-token effect is smaller for larger models. One prompt per model: these are
measurements, not a ranking of substrates.

**Clean vs Fly on the raw seed.** Greedy decoding, 48 tokens, «Однажды в студёную зимнюю
пору». The branches agree for the first clause and then may diverge:

- 0.6B-Base: «…в студенческой библиотеке появилась новая книга» (Clean) versus «…в классе
  появился один из самых умных учеников» (Fly).
- 1.7B-Base: «…вдруг наступила тёплая весна» versus «…вдруг появилась волна».
- 4B-Base: identical within 48 tokens.

The quality of the prose is not evidence of anything about the reservoir.

**GUI acceptance.** `scripts/gui_acceptance_models.py`, output in
`results/gui-acceptance-models/`. The script drove real Qt widgets offscreen, with real models,
and passed 24 of 24 checks:

- every switch through the selector;
- layer ranges following 28 or 36 layers;
- layers set to 10 → 20 with the spin boxes and then **used**: every token's telemetry records
  hooks firing on exactly 10 and 20;
- NF4 4B entirely on GPU;
- memory returning to 1.12 GiB after switching from 4B back to 0.6B.

The 3D view was paused during the automated run: offscreen VTK EGL rendering takes about 2 s
per frame on this machine, measured before any model was loaded. Native on-screen rendering
uses QtInteractor and was not part of this automated test.
