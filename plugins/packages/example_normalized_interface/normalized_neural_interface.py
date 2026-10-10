"""EXPERIMENTAL example: the original interface with unit-RMS steering normalization.

Original:   direction = centered / max(RMS(centered), 1)   (inactive while RMS < 1)
This file:  direction = centered / RMS(centered)            (always unit RMS)
            and no steering at all when RMS(centered) < min_rms (a near-zero reservoir state
            has no defined direction, so dividing by it would only amplify rounding noise).

Everything else (encoder, readout, decoder, seeds, feedback) is inherited unchanged. This is a
demonstration of replaceability, not a claim that it is better or more biological.
"""
from __future__ import annotations

import torch

from liquidaion_fly.interfaces import OriginalNeuralInterface


class UnitRmsNeuralInterface(OriginalNeuralInterface):
    def __init__(self, hidden_size: int, reservoir_neurons: int, features: int, seed: int, min_rms: float):
        super().__init__(hidden_size, reservoir_neurons, features, seed)
        self.min_rms = min_rms

    def steer(self, h, state, strength):
        compact = torch.from_numpy((state @ self.readout).astype("float32")).to(h.device)
        raw = compact @ self.decoder.to(h.device)
        centered = raw - raw.mean()
        rms = float(centered.square().mean().sqrt())
        if strength == 0 or rms < self.min_rms:
            return h, {"centered": centered, "steering_vector": centered, "metrics": {"centered_rms": rms}}
        direction = centered / rms
        baseline = h[:, -1, :]
        baseline_rms = baseline.float().square().mean(dim=-1, keepdim=True).sqrt()
        modified = h.clone()
        modified[:, -1, :] = baseline + (strength * baseline_rms * direction).to(h.dtype)
        return modified, {"compact": compact, "raw": raw, "centered": centered, "direction": direction,
                          "steering_vector": centered, "metrics": {"centered_rms": rms}}

    def describe(self):
        return {**super().describe(), "steering": "delta = strength * RMS(h) * centered / RMS(centered); "
                "no steering when RMS(centered) < min_rms", "min_rms": self.min_rms, "experimental": True}
