# Continuous generation ("Infinite Nekrasov"): article assets

All files come from real runs on an RTX 3070 Ti. Method and full results:
[docs/CONTINUOUS_GENERATION.md](../../docs/CONTINUOUS_GENERATION.md). Raw data:
`results/continuous/` (one directory per series, one subdirectory per condition).

| File | Content |
|---|---|
| `sliding-window.svg` / `.png` | Sliding window and persistent reservoir, schematic |
| `reservoir-dynamics.png` | Reservoir RMS over 5000 tokens (sampled Fly, reset control) and 91-token phase locking |
| `ui/continuous-dialog.png` | The continuous-generation window after a real 1000-token run (Fly, T=0.8, top-p 0.9, seed 42) |
| `ui/observatory.png` | Main window at the end of the same run |
| `ui/run/` | That run's telemetry, text and states |

Reproduce everything: `./scripts/run_continuous_experiments.sh` (about 1.5 h). Then rebuild the
summary and figure with `.venv/bin/python scripts/summarize_continuous.py` and
`PYTHONPATH=src .venv/bin/python scripts/plot_continuous_publication.py`.
