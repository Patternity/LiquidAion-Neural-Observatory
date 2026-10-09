"""Export publication plots directly from a genuine portable session recording."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/observatory/recorded-activity.png"))
    args = parser.parse_args()
    recording = json.loads(args.recording.read_text())
    snapshots = [event["snapshot"] for event in recording["events"] if "snapshot" in event]
    if not snapshots:
        raise ValueError("Recording has no measured snapshots")
    plt.style.use("dark_background")
    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout="constrained")
    colors = ("#76d5bf", "#f4bc67", "#b394e6")
    measurements = (("state_rms", "Reservoir state RMS"),
                    ("delta_relative_rms", "Relative hidden perturbation"),
                    ("steering_vector_norm", "Steering vector norm"))
    for axis, color, (key, label) in zip(axes, colors, measurements):
        values = np.asarray([snapshot.get(key, 0.0) for snapshot in snapshots])
        axis.plot(np.arange(len(values)), values, color=color, linewidth=1.4)
        axis.set_ylabel(label)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Recorded snapshot index (live and temporary scientific conditions)")
    figure.suptitle("LiquidAion: actual recorded CPU activity\n"
                   "Condition resets are part of the experiment; this is not biological activity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
