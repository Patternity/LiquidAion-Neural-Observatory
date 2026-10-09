"""Prepare an induced, directed FlyWire circuit with explicit provenance."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def largest_scc(matrix: sparse.csr_matrix) -> np.ndarray:
    count, labels = connected_components(matrix, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=count)
    return np.flatnonzero(labels == int(sizes.argmax()))


def prepare(raw: Path, output: Path, neurons: int = 512) -> dict:
    if neurons < 2:
        raise ValueError("At least two neurons are required")
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in ("classification", "neurons", "connections"):
        path = raw / f"{name}.csv.gz"
        url = BASE_URL + path.name
        if not path.exists():
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        sources.append({"name": path.name, "url": url, "sha256": sha256(path),
                        "bytes": path.stat().st_size})
    classification = pd.read_csv(raw / "classification.csv.gz", dtype=str).fillna("")
    # CX is the central-complex class supplied by the source annotations.
    candidates = classification.loc[classification["class"] == "CX", "root_id"].tolist()
    if not candidates:
        raise ValueError("The source has no CX-annotated neurons")
    candidate_set = set(candidates)
    parts = []
    for chunk in pd.read_csv(raw / "connections.csv.gz", chunksize=250_000,
                             dtype={"pre_root_id": str, "post_root_id": str}):
        keep = chunk.pre_root_id.isin(candidate_set) & chunk.post_root_id.isin(candidate_set)
        parts.append(chunk.loc[keep, ["pre_root_id", "post_root_id", "syn_count"]])
    edges = pd.concat(parts, ignore_index=True)
    if edges.empty or (edges.syn_count <= 0).any():
        raise ValueError("Invalid or absent source synapse counts")
    edges = edges.groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum()
    index = {root: i for i, root in enumerate(candidates)}
    matrix = sparse.csr_matrix((edges.syn_count.to_numpy(float),
                               (edges.pre_root_id.map(index), edges.post_root_id.map(index))),
                              shape=(len(candidates), len(candidates)))
    component = largest_scc(matrix)
    # Select by total within-component synapse weight, breaking ties by root ID.
    sub = matrix[component][:, component]
    degree = np.asarray(sub.sum(axis=0)).ravel() + np.asarray(sub.sum(axis=1)).ravel()
    rank = sorted(range(len(component)), key=lambda i: (-degree[i], candidates[component[i]]))
    selected = component[np.asarray(rank[:neurons])]
    induced = matrix[selected][:, selected]
    recurrent = largest_scc(induced)
    selected = selected[recurrent]
    if len(selected) < 2:
        raise ValueError("Selected subset has no nontrivial strongly connected component")
    roots = [candidates[i] for i in selected]
    keep = edges.pre_root_id.isin(roots) & edges.post_root_id.isin(roots)
    retained = edges.loc[keep].sort_values(["pre_root_id", "post_root_id"])
    retained.to_csv(output / "edges.csv", index=False)
    annotation = classification.set_index("root_id").loc[roots].reset_index()
    transmitter = pd.read_csv(raw / "neurons.csv.gz", dtype={"root_id": str})
    annotation = annotation.merge(transmitter[["root_id", "nt_type", "nt_type_score"]], on="root_id", how="left")
    annotation.to_csv(output / "nodes.csv", index=False)
    graph, ids = load_graph(output)
    manifest = {
        "dataset": "FlyWire FAFB", "materialization": 783,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "license": "CC BY-NC 4.0", "license_source": "https://flywire.ai/guidelines",
        "selection": "CX class; largest SCC; top weighted degree; largest induced SCC",
        "requested_nodes": neurons, "candidate_nodes": len(candidates),
        "candidate_largest_scc": len(component), "nodes": len(ids), "edges": int(graph.nnz),
        "synapses": int(graph.sum()), "strongly_connected": len(largest_scc(graph)) == len(ids),
        "edge_sha256": sha256(output / "edges.csv"), "node_sha256": sha256(output / "nodes.csv"),
        "transformations": ["aggregate neuropil rows per directed pair", "induce selected nodes",
                            "remove all connections to omitted neurons"],
        "weight_semantics": "Positive synapse counts; no neurotransmitter sign assignment",
        "annotation_warning": "Codex annotations can change independently of materialization; hashes freeze this export",
        "synapse_warning": "Materialization alone does not identify the synapse-prediction release; export hashes are authoritative",
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_graph(directory: Path) -> tuple[sparse.csr_matrix, list[str]]:
    nodes = pd.read_csv(directory / "nodes.csv", dtype={"root_id": str})
    ids = nodes.root_id.tolist()
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError("Duplicate or insufficient neuron IDs")
    index = {root: i for i, root in enumerate(ids)}
    edges = pd.read_csv(directory / "edges.csv", dtype={"pre_root_id": str, "post_root_id": str})
    if not set(edges.pre_root_id).union(edges.post_root_id).issubset(index):
        raise ValueError("An edge endpoint is missing from nodes.csv")
    values = edges.syn_count.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Synapse counts must be finite and positive")
    # W[post, pre] transports presynaptic state into the postsynaptic node.
    graph = sparse.csr_matrix((values, (edges.post_root_id.map(index), edges.pre_root_id.map(index))),
                              shape=(len(ids), len(ids)))
    return graph, ids
