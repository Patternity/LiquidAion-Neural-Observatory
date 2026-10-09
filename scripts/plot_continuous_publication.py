"""Publication figure: reservoir dynamics over thousands of tokens, from saved continuous runs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from liquidaion_fly.continuous import load_run, repetition_metrics

root = Path("results/continuous")
out = Path("publication/continuous")
ink, muted, grid, shade = "#0b0b0b", "#52514e", "#e4e3df", "#fde8de"
blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
                     "axes.titlecolor": ink, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), gridspec_kw={"height_ratios": [1.2, 1, 1]})
run = load_run(root / "sample-t1.0-p0.95-seed42/fly")
rows = run["telemetry"]
ax = axes[0]
for interval in repetition_metrics(run["tokens"]["generated"])["intervals"]["list"]:
    ax.axvspan(interval["start"], interval["end"], color=shade, lw=0)
ax.plot([r["step"] for r in rows], [r["state_rms"] for r in rows], color=blue, lw=0.6)
ax.set_title("Fly, persistent reservoir · sampling T=1.0, top-p 0.95, seed 42 · 128-token window "
             "(shaded: repetition-dominated text)", loc="left")
ax.set_ylabel("reservoir state RMS")
ax.grid(color=grid, lw=0.5)

reset = load_run(root / "sample-t1.0-p0.95-seed42/fly_reset_512")
ax = axes[1]
ax.plot([r["step"] for r in reset["telemetry"]], [r["state_rms"] for r in reset["telemetry"]], color=orange, lw=0.6)
for step in reset["summary"]["resets"]:
    ax.axvline(step, color=muted, lw=0.6, alpha=0.6)
ax.set_title("Control: same run, reservoir reset to zero every 512 tokens (grey lines)", loc="left")
ax.set_ylabel("reservoir state RMS")
ax.set_xlabel("generated token")
ax.grid(color=grid, lw=0.5)

chat = load_run(root / "chat-s0.25-f0.25/fly")
ax = axes[2]
window = [r for r in chat["telemetry"] if 2000 <= r["step"] < 2364]
ax.plot([r["step"] for r in window], [r["state_rms"] for r in window], color=aqua, lw=1.2)
for start in range(2000, 2364, 91):
    ax.axvline(start, color=muted, lw=0.6, ls="--", alpha=0.7)
ax.set_title("Phase locking: greedy run whose text repeats one 91-token paragraph; "
             "reservoir RMS repeats with the same 91-token period (dashed lines)", loc="left")
ax.set_ylabel("reservoir state RMS")
ax.set_xlabel("generated token")
ax.grid(color=grid, lw=0.5)
fig.tight_layout()
fig.savefig(out / "reservoir-dynamics.png", dpi=140)
print("saved", out / "reservoir-dynamics.png")
