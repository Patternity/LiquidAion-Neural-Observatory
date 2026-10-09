# Reproducibility guide

This guide lists everything needed to reproduce the results reported for release v0.2.0.

## 1. Fixed identities

| Item | Value |
|---|---|
| Software | LiquidAion Neural Observatory 0.2.0. Use the git tag `v0.2.0` or the Zenodo archive. |
| Language model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, pretrained, frozen. float16 on CUDA, float32 on CPU. Eager attention. |
| Chat template | Qwen3 template with `enable_thinking=False`. Generation stops on `<\|im_end\|>` (151645) or `<\|endoftext\|>` (151643). |
| Decoding | Greedy argmax. No sampling, temperature or repetition penalty. |
| Connectome | FlyWire FAFB materialization 783 (Codex export). 512 neurons, 16,731 edges, 310,999 synapses. |
| Graph hashes | `edges.csv` SHA-256 `5c524dd9…5bae54`, `nodes.csv` SHA-256 `62f91abd…a20f4` (full values in `data/cx512/provenance.json`) |
| Neuron IDs | `data/cx512/nodes.csv` (root IDs, in graph order). The anatomy files use the same order. |
| Coordinates | FlyWire / FAFB14.1 nanometers, unmodified (see `docs/ANATOMY.md`) |
| Reservoir | `W` = row-normalized synapse counts × gain 0.9. `dt = 0.2`, `τ = 1.0`, `input_strength = 0.5`, leaky tanh rate dynamics. Fingerprint in each recording's metadata. |
| Coupling | Capture at layer 6, steering at layer 14 (zero-based). 64 features. Default strength 0.25 and feedback 0.25. |
| Seeds | 42 for the projections, the reservoir input matrix and the random-topology control (the CLI experiments use seed + 1 for their shuffled control). |
| File hashes | `MANIFEST.sha256.json` lists SHA-256 values for every data and configuration file. |

## 2. Tested environment

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS, Linux 7.0.0-34 |
| CPU / GPU | AMD Ryzen 7 7700X / NVIDIA GeForce RTX 3070 Ti 8 GB, driver 595.71.05 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 (CUDA 12.8, cuDNN 9.10.2) |
| transformers | 4.56.2 |
| numpy / scipy / pandas | 2.3.5 / 1.17.0 / 2.2.3 |
| PySide6 / pyvista / vtk | 6.12.0 / 0.46.5 / 9.5.2 |

`requirements-cpu.txt` pins the same versions with the CPU-only PyTorch build. CI uses it.

## 3. Commands and expected outputs

Run everything from the repository root after `./scripts/launch_observatory.sh` has created
`.venv`. Alternatively, follow the manual installation in the README.

| Result | Command | Output | Runtime (RTX 3070 Ti) |
|---|---|---|---|
| Software tests | `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` | 77 passed, 2 skipped (real-model tests) | ~10 s |
| Real-model context regression | `LIQUIDAION_REAL_MODEL=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_real_model_regression.py` | 2 passed | ~6 s |
| Paired CLI experiments (none/static/random/fly, persistence) | `PYTHONPATH=src .venv/bin/python -m liquidaion_fly.cli run --config configs/cuda.json --output results/cuda` | `results/cuda/summary.json` | ~15 s |
| GUI engine smoke (chat and experiments A/B/C, save/reload) | `./scripts/run_demo_smoke.sh --max-tokens 16 --output results/observatory-cuda` | `results/observatory-cuda/smoke.json` | ~20 s |
| Attractor diagnostic | `PYTHONPATH=src .venv/bin/python -m liquidaion_fly.attractor --output results/attractor` | JSON, CSV and PNG in `results/attractor/` | ~10 min |
| Screenshots and animation | `QT_QPA_PLATFORM=offscreen PYTHONPATH=src .venv/bin/python scripts/capture_publication_assets.py --output publication/hero --prompt "…"` | `dashboard.png`, `activity.gif`, `session.json` | ~15 s |

The paired CLI experiments call `torch.use_deterministic_algorithms(True)` and set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA. A repeat run is bitwise identical
(`summary.json → repeat.bitwise_equal_logits`). Numbers can differ slightly between GPU
models, drivers, and CPU versus GPU. The CPU results in `results/cpu/` agree with the CUDA
results to about two significant digits.

The exact commands used for the publication assets are in each `publication/*/capture.json`.

## 4. Measurement definitions

- **‖Δh‖/‖h‖** (`delta_relative_rms`). RMS of the change at the steered position divided by
  the RMS of the original layer-14 hidden state.
- **Mean KL(baseline‖condition).** Mean over generated steps of the KL divergence between
  next-token distributions, with both conditions forced through identical tokens.
- **Logit RMS difference.** RMS over the vocabulary and steps of the raw logit difference.
- **Argmax disagreement.** Fraction of steps where the most likely next token differs.
- **State RMS.** Root mean square of the 512-dimensional reservoir state.
- **Attractor diagnostic metrics** (char-3-gram similarity, JS divergence, period detection)
  are defined in `docs/CONTEXT_AND_ATTRACTOR.md`.

## 5. Controls

- **none.** The hooks are attached with strength 0. This must reproduce the baseline logits
  bit for bit (asserted).
- **static.** A frozen reservoir state, to separate state dynamics from a constant bias.
- **random.** A random directed graph that matches node count, edge count, the raw weight
  multiset and the number of self-loops. Degree sequence and strongly connected components
  are not preserved.
- **feedback off.** The reservoir is driven by observation only.
- **retained vs reset.** The same text context and KV cache, with only the initial reservoir
  state differing.

## 6. Known sources of variation

- float16 CUDA kernels differ numerically from float32 CPU. Greedy text can diverge after a
  near-tie between two tokens.
- The GUI keeps idle dynamics running between prompts. The exact reservoir state at the
  moment you press **Send** therefore depends on timing. The controlled experiments and the
  capture script start from defined states (zero, or a stated copy).
- Codex annotations can change upstream. The bundled subset and its hashes fix the graph used
  here.
