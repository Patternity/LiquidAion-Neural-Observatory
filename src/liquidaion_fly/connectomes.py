"""Connectome providers: topology and neuron metadata, independent of reservoir dynamics.

A provider returns a Connectome. Optional metadata (neurotransmitter labels, coordinates,
synapse tables) is None when the source does not have it; nothing is invented.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .connectome import load_graph, sha256
from .reservoir import random_control


@dataclass(frozen=True)
class Connectome:
    component_id: str
    weights: sparse.csr_matrix               # W[post, pre]: presynaptic state drives the postsynaptic node
    node_ids: list[str]
    nt_types: list[str | None] | None = None  # neurotransmitter labels per node, or None if unknown
    nt_confidence: list[float | None] | None = None
    positions: np.ndarray | None = None      # (n, 3) coordinates if the provider has them
    position_units: str | None = None
    synapses: Any = None                     # synapse-level table if available
    provenance: dict[str, Any] = field(default_factory=dict)
    source_dir: Path | None = None           # directory with nodes.csv / edges.csv, if file-based
    fingerprint: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.node_ids)
        if self.weights.shape != (n, n):
            raise ValueError(f"{self.component_id}: weights {self.weights.shape} do not match {n} neurons")
        if len(set(self.node_ids)) != n or n < 2:
            raise ValueError(f"{self.component_id}: duplicate or insufficient neuron ids")
        for name in ("nt_types", "nt_confidence"):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"{self.component_id}: {name} has {len(values)} entries for {n} neurons")
        if self.positions is not None and np.asarray(self.positions).shape != (n, 3):
            raise ValueError(f"{self.component_id}: positions must be ({n}, 3)")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", weights_fingerprint(self.weights, self.node_ids))

    @property
    def neurons(self) -> int:
        return len(self.node_ids)

    def describe(self) -> dict[str, Any]:
        return {"id": self.component_id, "neurons": self.neurons, "edges": int(self.weights.nnz),
                "fingerprint": self.fingerprint, "has_nt_types": self.nt_types is not None,
                "has_positions": self.positions is not None, "has_synapses": self.synapses is not None,
                "parameters": self.parameters, "provenance": self.provenance}


def weights_fingerprint(weights: sparse.csr_matrix, node_ids: list[str]) -> str:
    w = weights.tocsr().copy()
    w.sum_duplicates()
    w.sort_indices()
    digest = hashlib.sha256()
    for array in (w.data.astype(np.float64), w.indices.astype(np.int64), w.indptr.astype(np.int64)):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update("\n".join(node_ids).encode())
    return digest.hexdigest()


def flywire_cx512(context: Any) -> Connectome:
    """The historical FlyWire CX512 export (nodes.csv, edges.csv, provenance.json), read as before."""
    directory = Path(context.params.get("directory") or context.graph_dir)
    weights, node_ids = load_graph(directory)
    nodes = pd.read_csv(directory / "nodes.csv", dtype={"root_id": str})
    nt = [None if pd.isna(v) else str(v) for v in nodes["nt_type"]] if "nt_type" in nodes else None
    score = [None if pd.isna(v) else float(v) for v in nodes["nt_type_score"]] if "nt_type_score" in nodes else None
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    fingerprint = hashlib.sha256((sha256(directory / "edges.csv") + sha256(directory / "nodes.csv")).encode()).hexdigest()
    return Connectome("builtin.connectome.flywire_cx512", weights, node_ids, nt, score, provenance=provenance,
                      source_dir=directory, fingerprint=fingerprint, parameters={"directory": str(directory)})


def random_control_connectome(context: Any) -> Connectome:
    """reservoir.random_control applied to a base connectome: same node count, edge count, raw weight
    multiset and self-loop count, with off-diagonal positions drawn uniformly without replacement
    (numpy default_rng(seed)). Neuron ids and labels are kept; only the topology is randomized."""
    base = context.load(context.params["base"])
    seed = context.params["seed"] if context.params.get("seed") is not None else context.seed
    return replace(base, component_id="builtin.connectome.random_control", weights=random_control(base.weights, seed),
                   fingerprint="", positions=None, synapses=None,
                   parameters={"base": base.component_id, "base_fingerprint": base.fingerprint, "seed": seed,
                               "method": "reservoir.random_control: uniform off-diagonal positions, permuted "
                                         "weights, same loop count"})


def cue_group(connectome: Connectome, target: str, seed: int = 42) -> tuple[list[int], list[str]]:
    """Built-in modulation group. "da": nodes labelled DA; "random": as many non-DA nodes drawn with
    a fixed seed; "off": none. Identical to the historical nodes.csv rule (graph order)."""
    labels = connectome.nt_types
    if target in {"da", "random"} and labels is None:
        raise ValueError(f"{connectome.component_id} has no neurotransmitter labels; the DA cue needs them")
    da = [i for i, label in enumerate(labels or []) if label == "DA"]
    if target == "da":
        chosen = da
    elif target == "random":
        others = [i for i in range(connectome.neurons) if i not in set(da)]
        chosen = sorted(np.random.default_rng(seed).choice(others, size=len(da), replace=False).tolist())
    elif target == "off":
        chosen = []
    else:
        raise ValueError(f"Unknown cue target {target!r}")
    return chosen, [connectome.node_ids[i] for i in chosen]


def load_connectome(registry: Any, component_id: str, component_params: dict[str, dict], graph_dir: Path,
                    seed: int, params: dict | None = None) -> Connectome:
    """Resolve and build a connectome through the registry (composable: providers may load others)."""
    from types import SimpleNamespace
    spec = registry.get(component_id, "connectome")
    resolved = spec.resolve_params(params if params is not None else component_params.get(component_id))
    connectome = spec.factory(SimpleNamespace(
        params=resolved, graph_dir=Path(graph_dir), seed=seed,
        load=lambda other, other_params=None: load_connectome(registry, other, component_params, graph_dir, seed,
                                                              other_params)))
    if not isinstance(connectome, Connectome):
        raise TypeError(f"{component_id} must return a liquidaion_fly.connectomes.Connectome")
    return connectome
