"""EXPERIMENTAL example: steering gated by how active the reservoir is.

    centered  = original readout/decoder output, mean removed (inherited)
    gate      = tanh(gate_gain * RMS(reservoir state))           in [0, 1)
    delta     = strength * gate * RMS(h) * centered / max(RMS(centered), floor)

Copy this file, rename the class, change the expression in steer(), and register it in
plugin.py. The kernel validates the returned tensor (shape, dtype, device, finite values).
"""
from __future__ import annotations

import numpy as np
import torch

from liquidaion_fly.interfaces import OriginalNeuralInterface


class GatedSteeringInterface(OriginalNeuralInterface):
    def __init__(self, hidden_size, reservoir_neurons, features, seed, gate_gain, floor):
        super().__init__(hidden_size, reservoir_neurons, features, seed)
        self.gate_gain, self.floor = gate_gain, floor

    def steer(self, h, state, strength):
        state_rms = float(np.sqrt(np.mean(state ** 2)))
        gate = float(np.tanh(self.gate_gain * state_rms))
        compact = torch.from_numpy((state @ self.readout).astype("float32")).to(h.device)
        raw = compact @ self.decoder.to(h.device)
        centered = raw - raw.mean()
        direction = centered / centered.square().mean().sqrt().clamp_min(self.floor)
        internals = {"compact": compact, "centered": centered, "steering_vector": centered,
                     "metrics": {"gate": gate, "state_rms": state_rms}}
        if strength == 0 or gate == 0:
            return h, internals
        baseline = h[:, -1, :]
        baseline_rms = baseline.float().square().mean(dim=-1, keepdim=True).sqrt()
        modified = h.clone()
        modified[:, -1, :] = baseline + (strength * gate * baseline_rms * direction).to(h.dtype)
        return modified, internals

    def describe(self):
        return {**super().describe(), "experimental": True, "gate_gain": self.gate_gain, "floor": self.floor,
                "steering": "delta = strength * tanh(gate_gain * RMS(x)) * RMS(h) * centered / max(RMS(centered), floor)"}
