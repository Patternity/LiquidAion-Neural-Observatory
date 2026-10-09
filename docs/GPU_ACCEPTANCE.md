# Target hardware acceptance: NVIDIA RTX 3070 Ti

## Status for v0.2.0 (2026-10-09)

Hardware: RTX 3070 Ti 8 GB, driver 595.71.05, PyTorch 2.8.0+cu128, Ubuntu 24.04, Python 3.12.3.

**Verified on this hardware by scripted runs** (real pretrained weights; artifacts in the repository):

- CUDA kernel probe and model load in float16. There is no CPU fallback.
- GUI engine smoke: dual chat, experiments A/B/C, save and reload
  (`results/observatory-cuda/smoke.json`).
- Paired CLI experiments. A repeat run is bitwise identical (`results/cuda/summary.json`).
- Real-model multi-turn context regression (`tests/test_real_model_regression.py`).
- Attractor diagnostic, about 10 min (`results/attractor/`).
- Full desktop window, rendered through Qt offscreen with VTK EGL on this machine:
  two-turn chat, real anatomy, telemetry, export and recording (`publication/hero/`,
  `publication/habr-fly-question/`).

**Measured.**

- Throughput: about 50–55 tokens/s per branch for short answers with telemetry enabled.
- Memory: about 1.2 GiB allocated for weights and the KV cache, and a 3.66 GiB peak for a
  4096-token prefill. 6144-token prefills run out of memory with about 1.5 GB used by the
  desktop.

**Owner-run, not independently scripted:** native interactive rendering (QtInteractor),
including rotation, zoom, pan and neuron picking with the mouse. The owner ran the
application on this machine. The checklist below is kept for future target devices.

## Checklist for a new device

1. Record `nvidia-smi` card, driver and actual memory. Use Python 3.12+ and a
   CUDA-enabled PyTorch 2.8.0 build. The launcher checks availability and executes
   a real kernel. It does not silently fall back to CPU.
2. Run `./scripts/launch_observatory.sh`. Confirm the GPU report shows actual
   allocated/reserved and free/total memory. One model serves sequential branches
   using float16. Record peak VRAM during two turns and A/B/C demonstrations.
3. Send `Reply with exactly: Hello, world!`, then a second prompt. Confirm both
   panels stream text, report tokens/time/throughput and maintain separate histories.
4. Rotate, zoom and pan the center anatomy. Pick a neuron, compare its root ID
   and position with `data/anatomy/positions.csv`, and toggle real surfaces and
   abstract marker connections. Camera motion must not change coordinates.
5. Observe actual idle decay. Pause/resume the view while confirming simulation
   time advances. Reset to zero, change steering/feedback controls and cancel.
6. Run A/B/C. Inspect identical-prefix metrics and genuine intermediate states.
   C must include actual idle evolution and fresh contexts. No quality claim
   follows from a nonzero KL or logit difference.
7. Save/export and close normally. Open
   `./scripts/launch_observatory.sh --replay /absolute/path/to/session.json`
   without weights. Play/pause, change speed and seek recorded states/tokens.
   Restore the compatible session in a live model and send another message.
8. Run `./scripts/run_demo_smoke.sh` and `python -m pytest -q`. Preserve hardware,
   `smoke.json`, session, screenshots and test log as target-device evidence.

The launcher installs Python dependencies, not the host NVIDIA driver. If an
installed CUDA wheel is incompatible with the driver, select a compatible official
PyTorch build and repeat the actual kernel probe. Existing CPU artifacts are not target acceptance.
