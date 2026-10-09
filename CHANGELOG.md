# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/); the project is experimental (0.x).

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
