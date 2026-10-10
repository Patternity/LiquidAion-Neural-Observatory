"""Neural interfaces: the mathematics between language-model activations and the reservoir.

Contract (duck-typed; see docs/PLUGINS.md, "Neural interface"):

    hidden_size: int           model hidden size the interface was built for
    reservoir_neurons: int     reservoir state dimension it reads
    features: int              reservoir input dimension it produces
    fingerprint: str           identifies its fixed parameters (recorded with every run)
    recurrent: bool            True if it keeps its own state between calls (recorded)

    observe(hidden_last) -> np.ndarray (features,)
        hidden_last: the last-position hidden vector (hidden,) on the model device, model dtype
    feedback(steered_last) -> np.ndarray (features,)
        the steered last-position vector; the coupling scales the result by feedback strength
    steer(hidden, state, strength) -> (modified, internals)
        hidden: (1, seq, hidden) layer output; state: np.ndarray (reservoir_neurons,)
        returns a tensor of the same shape, dtype and device, and a small dict of tensors/floats
        for optional diagnostics (key "steering_vector" names the direction it added; key
        "metrics" holds scalar floats that are copied into telemetry)
    reset() -> None            clear interface state (new session / state reset)
    describe() -> dict         JSON-serializable parameters for experiment metadata

The kernel decides WHEN these run (coupling policy and hooks) and validates every output.
An interface never installs hooks, owns caches or advances the reservoir.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


class OriginalNeuralInterface:
    """The reference LiquidAion interface, unchanged from the original coupling.

    observe:  tanh((h / RMS(h)) @ encoder)                               encoder hidden x features
    steer:    compact = x @ readout; raw = compact @ decoder;           readout neurons x features
              centered = raw - mean(raw);                                decoder features x hidden
              direction = centered / max(RMS(centered), 1);
              delta = strength * RMS(h_last) * direction  (last position only)
    feedback: observe(steered last vector)
    The three matrices are drawn once from numpy default_rng(seed), in this order.
    """
    recurrent = False

    def __init__(self, hidden_size: int, reservoir_neurons: int, features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.hidden_size, self.reservoir_neurons, self.features, self.seed = hidden_size, reservoir_neurons, features, seed
        self.encoder = torch.from_numpy(rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, features)).astype("float32"))
        self.readout = rng.normal(0, 1 / np.sqrt(reservoir_neurons), (reservoir_neurons, features))
        self.decoder = torch.from_numpy(rng.normal(0, 1 / np.sqrt(features), (features, hidden_size)).astype("float32"))

    @property
    def fingerprint(self) -> str:
        """Identifies the fixed encoder, readout and decoder (the historical projection fingerprint)."""
        digest = hashlib.sha256()
        for array in (self.encoder.numpy(), self.readout, self.decoder.numpy()):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def observe(self, hidden_last: torch.Tensor) -> np.ndarray:
        x = hidden_last.detach().float()
        x = x / x.square().mean().sqrt().clamp_min(1e-8)
        projection = self.encoder.to(device=x.device)
        return torch.tanh(x @ projection).cpu().numpy()

    def feedback(self, steered_last: torch.Tensor) -> np.ndarray:
        return self.observe(steered_last)

    def steer(self, h: torch.Tensor, state: np.ndarray, strength: float) -> tuple[torch.Tensor, dict[str, Any]]:
        compact = torch.from_numpy((state @ self.readout).astype("float32")).to(h.device)
        raw = compact @ self.decoder.to(h.device)
        centered = raw - raw.mean()
        direction = centered / centered.square().mean().sqrt().clamp_min(1.0)
        baseline = h[:, -1, :]
        baseline_rms = baseline.float().square().mean(dim=-1, keepdim=True).sqrt()
        delta = (strength * baseline_rms * direction).to(h.dtype)
        if strength == 0:
            modified = h
        else:
            modified = h.clone()
            modified[:, -1, :] = baseline + delta
        return modified, {"compact": compact, "raw": raw, "centered": centered, "direction": direction,
                          "steering_vector": centered}

    def reset(self) -> None:
        """Stateless."""

    def describe(self) -> dict[str, Any]:
        return {"hidden_size": self.hidden_size, "reservoir_neurons": self.reservoir_neurons,
                "features": self.features, "seed": self.seed, "encoder_shape": list(self.encoder.shape),
                "readout_shape": list(self.readout.shape), "decoder_shape": list(self.decoder.shape),
                "steering": "delta = strength * RMS(h) * centered / max(RMS(centered), 1)",
                "fingerprint": self.fingerprint}


def check_features(vector: Any, expected: int, where: str) -> np.ndarray:
    """Reservoir input from an interface: a finite float vector of the declared size."""
    if not isinstance(vector, np.ndarray) or vector.shape != (expected,):
        shape = getattr(vector, "shape", type(vector).__name__)
        raise ValueError(f"{where} returned {shape}; the reservoir expects a numpy vector of shape ({expected},)")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{where} returned nonfinite reservoir input")
    return vector


def check_steered(h: torch.Tensor, modified: Any, where: str) -> torch.Tensor:
    """A steered layer output must keep shape, dtype and device, and stay finite."""
    if not isinstance(modified, torch.Tensor) or modified.shape != h.shape:
        raise ValueError(f"{where} returned {getattr(modified, 'shape', type(modified).__name__)}; "
                         f"expected a tensor of shape {tuple(h.shape)}")
    if modified.dtype != h.dtype or modified.device != h.device:
        raise ValueError(f"{where} returned {modified.dtype} on {modified.device}; expected {h.dtype} on {h.device}")
    if not torch.isfinite(modified).all():
        raise FloatingPointError("Nonfinite hidden activation after intervention")
    return modified
