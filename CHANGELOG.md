# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/); the project is experimental (0.x).

## [Unreleased]

### Added
- Four selectable language substrates with pinned revisions (`liquidaion_fly.models`):
  - Qwen3-0.6B (instruct, the unchanged baseline);
  - Qwen3-0.6B-Base and Qwen3-1.7B-Base (bfloat16);
  - Qwen3-4B-Base (4-bit NF4 via bitsandbytes).
  Model switching from the GUI keeps one model in GPU memory. Base models use raw
  continuation, with no chat template or role markers.
- Adjustable capture and steering layers in a MODEL COUPLING panel.
  - Projection sizes follow the loaded model's hidden size.
  - The selected layers are recorded per run and per token.
- `scripts/model_acceptance.py` and `scripts/gui_acceptance_models.py`; `--model-profile` option.
- Live external injection into continuous generation.
  - Operator text is tokenized as typed and teacher-forced into the stream at token
    boundaries, with no role markers.
  - Token provenance (AUTONOMOUS / EXTERNAL INJECTION / POST-INJECTION) is tracked by window
    membership, colored in the GUI and exported.
  - Interventions are recorded with stream positions and reproduce exactly from a schedule
    (`--inject`, `--leak-at`).
- Optional external neural cue: an extra input to the 16 `nt_type = DA` neurons, or a random
  size-matched control, while injected tokens are processed.
- Runtime reservoir leak-rate control in Continuous Generation and in a new RESERVOIR
  DYNAMICS panel of the classic GUI.
  - The state is kept and changes are recorded.
  - The fingerprint includes the dynamics, so sessions restore their leak.
- `scripts/injection_experiment.py` and `scripts/gui_acceptance_injection.py`.
- Continuous sliding-window generation ("Infinite Nekrasov"). The model sees only the last N
  tokens; the reservoir persists across thousands of tokens.
  - Conditions: Clean, persistent Fly, periodically reset Fly, random topology.
  - Same-suffix checkpoint probes on cloned reservoirs.
  - Streaming telemetry, text and state snapshots written to disk.
  - CLI `liquidaion-continuous`, GUI window "Continuous generation", and
    `scripts/run_continuous_experiments.sh`. See `docs/CONTINUOUS_GENERATION.md`.
- Zenodo DOIs in the README and `CITATION.cff`: concept `10.5281/zenodo.23265197` and v0.2.0 `10.5281/zenodo.23265198`.

## [0.2.0] - 2026-10-09

First public release.

### Added
- LiquidAion Neural Observatory: a PySide6 desktop application. One frozen Qwen3-0.6B
  instance serves a Clean chat and a Fly chat. The Fly chat is coupled to a 512-neuron
  FlyWire-derived rate reservoir (observation at layer 6, steering at layer 14, feedback).
- A registered 3D FlyWire anatomy view with live reservoir state, plus telemetry, session
  recording, offline replay and dashboard export.
- Controlled experiments A (coupling), B (random topology) and C (persistent vs reset state).
- Configurable `max_new_tokens` (default 512, up to 4096) and context window (default 4096),
  both persisted in the GUI. Every answer reports a stop reason.
- An opt-in inference debug log (`--debug-log summary|full`) of the exact model inputs.
- An attractor diagnostic (`liquidaion-attractor`) for repetition, input sensitivity, the
  turn-to-turn map, and recovery.
- Regression tests for context construction, branch isolation, KV-cache handling, the Qwen3
  template, and an opt-in real-model regression.
- `scripts/capture_publication_assets.py`, which captures real screenshots and animations.
- CI, citation metadata, contribution guide and third-party notices.

### Fixed
- An oversized prompt was silently cut from the left, breaking the chat template. It is now
  rejected with an explicit error.
- Context trimming no longer deletes conversation history. Only the model input is trimmed.
- Cancellation and branch failures no longer leave the Clean and Fly histories misaligned.
- Generation now also stops on `<|endoftext|>`.
- A Clean forward pass refuses to run if a forward hook is present.
- CUDA runs of the paired CLI experiments failed with deterministic algorithms enabled.
  `CUBLAS_WORKSPACE_CONFIG` is now set.
- The launcher detects a CPU-only PyTorch build in GPU mode, checks free disk space, and
  installs without the pip cache.

## 0.1.0 (unreleased internal prototype)

- CPU prototype: paired fixed-context experiments with `none`, `static`, `random` and `fly`
  controls, and cross-prompt persistence (`liquidaion-fly run`). See `docs/CPU_PROTOTYPE.md`.

[0.2.0]: https://github.com/Patternity/LiquidAion-Neural-Observatory/releases/tag/v0.2.0
