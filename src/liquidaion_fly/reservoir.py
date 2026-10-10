"""Sparse leaky rate dynamics; not a biophysical simulation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Dynamics:
    dt: float = 0.2
    tau: float = 1.0
    gain: float = 0.9
    input_strength: float = 0.5

    def __post_init__(self) -> None:
        if not all(np.isfinite(x) for x in asdict(self).values()):
            raise ValueError("Dynamics parameters must be finite")
        if self.tau <= 0 or not 0 < self.dt / self.tau <= 1:
            raise ValueError("Require 0 < dt/tau <= 1")
        if not 0 <= self.gain < 1 or self.input_strength < 0:
            raise ValueError("Require 0 <= gain < 1 and nonnegative input strength")


class Reservoir:
    """The built-in reservoir (builtin.reservoir.leaky_tanh). Capabilities are declared for plugins."""
    capabilities = ("idle", "extra_input", "set_leak", "snapshot")

    def __init__(self, graph: sparse.csr_matrix, features: int = 64,
                 seed: int = 42, dynamics: Dynamics = Dynamics()):
        graph = graph.astype(np.float64).tocsr()
        graph.sum_duplicates()
        graph.sort_indices()
        if graph.shape[0] != graph.shape[1] or graph.nnz == 0 or features < 1:
            raise ValueError("A nonempty square graph and positive feature count are required")
        if not np.isfinite(graph.data).all():
            raise ValueError("Graph contains nonfinite weights")
        rows = np.asarray(abs(graph).sum(axis=1)).ravel()
        self.w = (sparse.diags(1 / np.maximum(rows, 1)) @ graph * dynamics.gain).tocsr()
        self.dynamics = dynamics
        self.input = np.random.default_rng(seed).normal(0, 1 / np.sqrt(features), (graph.shape[0], features))
        self.state = np.zeros(graph.shape[0])
        self.time = 0.0
        self._update_fingerprint()

    def _update_fingerprint(self) -> None:
        # Covers graph, input projection and dynamics, so a state never restores under other dynamics.
        self.fingerprint = hashlib.sha256(
            self.w.data.tobytes() + self.w.indices.tobytes() + self.w.indptr.tobytes()
            + self.input.tobytes() + json.dumps(asdict(self.dynamics), sort_keys=True).encode()).hexdigest()

    @property
    def input_features(self) -> int:
        return self.input.shape[1]

    @property
    def leak(self) -> float:
        return self.dynamics.dt / self.dynamics.tau

    def set_leak(self, leak: float) -> None:
        """Change the leak dt/tau of the update equation in place, keeping the current state and dt."""
        if not np.isfinite(leak) or not 0 < leak <= 1:
            raise ValueError("Leak rate must lie in (0, 1]")
        if leak == self.leak:
            return
        self.dynamics = replace(self.dynamics, tau=self.dynamics.dt / leak)
        self._update_fingerprint()

    def advance(self, features: np.ndarray | None = None, steps: int = 1,
                extra: np.ndarray | None = None) -> np.ndarray:
        """Leaky tanh update; `extra` is an additional per-neuron input added to the forcing."""
        if steps < 0 or not isinstance(steps, int):
            raise ValueError("Steps must be a nonnegative integer")
        u = np.zeros(self.input.shape[1]) if features is None else np.asarray(features, dtype=float)
        if u.shape != (self.input.shape[1],) or not np.isfinite(u).all():
            raise ValueError("Invalid input features")
        forcing = self.dynamics.input_strength * (self.input @ u)
        if extra is not None:
            extra = np.asarray(extra, dtype=float)
            if extra.shape != self.state.shape or not np.isfinite(extra).all():
                raise ValueError("Invalid extra reservoir input")
            forcing = forcing + extra
        leak = self.dynamics.dt / self.dynamics.tau
        for _ in range(steps):
            self.state = (1 - leak) * self.state + leak * np.tanh(self.w @ self.state + forcing)
            self.time += self.dynamics.dt
        return self.state.copy()

    def reset(self) -> None:
        self.state.fill(0)
        self.time = 0.0

    def snapshot(self, path: Path) -> None:
        np.savez(path, state=self.state, time=self.time, fingerprint=self.fingerprint)

    def restore(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            state = data["state"]
            if str(data["fingerprint"]) != self.fingerprint or state.shape != self.state.shape:
                raise ValueError("Snapshot belongs to a different graph, adapter or dynamics")
            if not np.isfinite(state).all() or np.max(abs(state)) > 1 + 1e-9:
                raise ValueError("Invalid snapshot state")
            time = float(data["time"])
            if not np.isfinite(time) or time < 0:
                raise ValueError("Invalid snapshot time")
            self.state = state.copy()
            self.time = time


def random_control(graph: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """Match node count, edge count, raw weight multiset and loop count."""
    rng = np.random.default_rng(seed)
    n = graph.shape[0]
    diagonal = int(np.count_nonzero(graph.diagonal()))
    off_count = graph.nnz - diagonal
    # Sample directed off-diagonal positions without replacement.
    positions = rng.choice(n * (n - 1), size=off_count, replace=False)
    row = positions // (n - 1)
    col = positions % (n - 1)
    col += col >= row
    loops = rng.choice(n, size=diagonal, replace=False)
    row = np.concatenate([row, loops])
    col = np.concatenate([col, loops])
    return sparse.csr_matrix((rng.permutation(graph.data), (row, col)), shape=graph.shape)
