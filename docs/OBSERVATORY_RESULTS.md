# LiquidAion Neural Observatory: verified results

Validation date: 2026-10-09. Version 0.2.0. All inference measurements below
use actual pretrained Qwen3-0.6B on CPU, measured in an earlier CPU-only build environment
(historical pre-release record). RTX 3070 Ti results: `docs/GPU_ACCEPTANCE.md`, `README.md`.

## Executed checks

- Full suite: **52 passed in 16.57s**, process exit 0. The original 21 tests remain included.
- Main launcher preparation: graph/anatomy integrity checks completed, exit 0.
- Explicit CUDA startup without GPU: clear startup failure, exit 2, no silent fallback.
- Real model smoke: shared-model dual generation, all three experiments, save/reload, exit 0.
- Real desktop worker: two turns, 133 GUI heartbeat callbacks, genuine rendering, idle decay, reset, normal exit 0.
- Offline Qt worker replay: 139 actual snapshots available, representative start/peak/middle/final state identity checked, model not loaded, final conversations match, exit 0.
- Unit regression also seeks every frame in the actual smoke recording.
- Dashboard and anatomy PNGs verified structurally; exports contain actual telemetry.

Software tests use tiny random-weight transformers and isolated GUI fixtures.
They test architecture and lifecycle, not pretrained language ability. Runtime
and desktop probes use the real model and genuine anatomy. Headless Qt displays
real VTK EGL renders; native QtInteractor interaction still needs a Linux display.

## Identity and configuration

Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`.

Connectome edge SHA-256: `5c524dd9aca9557ab94ad04f36668e6a86d108c905f31d909b43cc916e5bae54`.

Node SHA-256: `62f91abd6f9da4a4f5f81f8307c2a4be0e894c8d95ad240b8cba63178f1a20f4`.

One model instance serves independent Clean/Fly histories, with hooks only on
Fly forwards. Greedy decoding, seed 42, features 64, layer 6 capture / layer 14
steering, strength 0.25 and feedback 0.25. Context cap 2048. The 512-node sparse
reservoir uses row-sum gain 0.9, dt 0.2, tau 1, input strength 0.5.

The authentic graph has 16,731 directed edges and 310,999 synapses. All 512 IDs
map to published representative positions in exact graph order. Two compatible
FlyWire anatomical envelopes and 16 full published skeletons share the registered
FlyWire/FAFB14.1 nanometer space. Markers are not soma centroids; optional graph
lines are not reconstructed axons. Sources, hashes, registration and licenses are
in the anatomy manifest and `docs/ANATOMY.md`.

## Controlled measurements

| Experiment | Matched steps | Logit RMS difference | Mean KL | Argmax disagreement |
|---|---:|---:|---:|---:|
| A: Clean vs Fly | 8 | 0.113750935 | 0.00163115228 | 0 |
| B: Fly vs random | 8 | 0.0353380591 | 0.000112614454 | 0 |
| C: Retained vs reset | 8 | 0.0362332761 | 0.00010866324 | 0 |

All paired metrics use identical prompt IDs and a shared forced continuation.
Free texts are independently generated and kept separate from these metrics.
Identical short greedy answers do not imply identical logits.

A descriptive free outputs:

- clean: `The sky appears blue due to the scattering` (8 tokens, 2.424 s).
- fly: `The sky appears blue due to the scattering` (8 tokens, 1.744 s).

B descriptive free outputs:

- fly: `The sky appears blue due to the scattering` (8 tokens, 2.079 s).
- random: `The sky appears blue due to the scattering` (8 tokens, 1.656 s).

C uses prompts `A friend helped me fix a bug.` and `Write one short thank-you sentence.`, with fresh context for B. RMS before 10 actual idle steps is 0.143116098; retained RMS afterward is 0.057631474.

The short smoke continuations are capped at eight tokens and can end mid-sentence.
The live desktop probe produces complete short responses:

- Turn 1: Clean `Hello, world!`, Fly `Hello, world!`.
- Turn 2: Clean `Goodbye!`, Fly `Goodbye!`.

During the live probe, idle RMS falls from 0.147516970 to 0.066648186. Reset gives RMS 0.0, maximum activity alpha 0. Exactly zero activity is transparent.

## Interpretation and remaining limits

These measurements establish genuine hidden-state intervention and fading-state
influence, not improved language quality, topology superiority, semantic memory,
emotion or consciousness. Positive synapse counts do not provide physiological
signs. The engineered rate dynamics are not a faithful fly simulation. Random
controls change degrees and normalized effective weights. One seed and a small
prompt set cannot support general performance claims.

CPU timing comes from a shared host and includes incidental load; it is not a
reliable performance comparison. No GPU throughput or frame-rate claim is made.
The actual environment and process RAM measurements are stored in `smoke.json`
and recording metadata. CUDA memory counters and float16 inference are implemented
but remain target-device checks. Follow `docs/GPU_ACCEPTANCE.md`.

## Reproduce

From the extracted project root on a Linux desktop:

```bash
./scripts/launch_observatory.sh
./scripts/run_demo_smoke.sh
```

Explicit CPU and model-free recording routes:

```bash
./scripts/launch_observatory.sh --cpu
./scripts/launch_observatory.sh --replay results/observatory/session.json
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

The archive includes source, exact graph/anatomy caches, recordings, measured
screenshots, activity plots, licenses and English/Russian publication drafts.
Model weights download separately at the pinned revision. Nothing was pushed,
released or published remotely.
