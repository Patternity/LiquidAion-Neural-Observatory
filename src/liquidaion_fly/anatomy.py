"""Registered public FlyWire anatomy with exact graph-index mapping in nm."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import re
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

COORDINATES_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/coordinates.csv.gz"
FLYBRAINS_COMMIT = "273333c8d8bf5adeebebd274e554621462e388bd"
FLYBRAINS_BASE = f"https://raw.githubusercontent.com/navis-org/navis-flybrains/{FLYBRAINS_COMMIT}/flybrains/"
SKELETON_BASE = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783"
SPACE = "FLYWIRE/FAFB14.1"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path, max_bytes: int = 12_000_000) -> None:
    """Download atomically, without credentials or placeholder geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=45) as response, temporary.open("wb") as out:
            count = 0
            while block := response.read(128 * 1024):
                count += len(block)
                if count > max_bytes:
                    raise ValueError(f"Anatomy download exceeds byte limit: {url}")
                out.write(block)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _graph_ids(graph_dir: Path) -> list[str]:
    with (graph_dir / "nodes.csv").open(newline="") as stream:
        ids = [row["root_id"] for row in csv.DictReader(stream)]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Graph neuron IDs must be present and unique")
    return ids


def decode_skeleton(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode upstream Neuroglancer vertices, edges and radius, all in nm."""
    if len(data) < 8:
        raise ValueError("Truncated skeleton header")
    n_vertices, n_edges = struct.unpack_from("<II", data)
    if n_vertices == 0 or len(data) != 8 + n_vertices * 16 + n_edges * 8:
        raise ValueError("Invalid skeleton length or absent vertices")
    vertices = np.frombuffer(data, dtype="<f4", count=n_vertices * 3, offset=8).reshape(-1, 3).copy()
    offset = 8 + n_vertices * 12
    edges = np.frombuffer(data, dtype="<u4", count=n_edges * 2, offset=offset).reshape(-1, 2).copy()
    radius = np.frombuffer(data, dtype="<f4", count=n_vertices, offset=offset + n_edges * 8).copy()
    if not np.isfinite(vertices).all() or not np.isfinite(radius).all():
        raise ValueError("Nonfinite skeleton geometry")
    if len(edges) and int(edges.max()) >= n_vertices:
        raise ValueError("Skeleton endpoint outside vertex array")
    return vertices, edges, radius


@dataclass(frozen=True)
class Anatomy:
    directory: Path
    neuron_ids: list[str]
    positions: np.ndarray
    meshes: dict[str, Path]
    skeletons: dict[str, Path]
    manifest: dict
    units: str = "nm"
    coordinate_space: str = SPACE

    @property
    def mesh_paths(self) -> tuple[Path, ...]:
        return tuple(self.meshes.values())

    def read_skeleton(self, root_id: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.skeletons[str(root_id)], allow_pickle=False) as data:
            return data["vertices"].copy(), data["edges"].copy()

    @classmethod
    def load(cls, directory: Path | str) -> "Anatomy":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("coordinate_space") != SPACE or manifest.get("units") != "nm":
            raise ValueError("Unsupported anatomical coordinate reference")
        for entry in manifest["files"]:
            path = directory / entry["path"]
            if not path.is_file() or _hash(path) != entry["sha256"]:
                raise ValueError(f"Missing or changed anatomical cache file: {path.name}")
        with (directory / "positions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [row["root_id"] for row in rows]
        positions = np.asarray([[float(row[k]) for k in ("x_nm", "y_nm", "z_nm")] for row in rows])
        if ids != manifest["neuron_ids"] or len(set(ids)) != len(ids):
            raise ValueError("Neuron index mapping differs from anatomy manifest")
        if positions.shape != (len(ids), 3) or not np.isfinite(positions).all():
            raise ValueError("Invalid anatomical positions")
        return cls(directory, ids, positions,
                   {k: directory / v for k, v in manifest["meshes"].items()},
                   {k: directory / v for k, v in manifest.get("skeletons", {}).items()}, manifest)


def load_anatomy(directory: Path | str) -> Anatomy:
    return Anatomy.load(directory)


def ensure_anatomy(graph_dir: Path | str, cache_dir: Path | str,
                   skeleton_count: int = 16) -> Anatomy:
    """Load a verified cache, or fetch registered public anatomy once.

    All graph neurons require real coordinates. A changed graph or geometry
    fails explicitly. No extrapolated position or generated mesh replaces it.
    """
    graph_dir, cache_dir = Path(graph_dir), Path(cache_dir)
    ids, node_hash = _graph_ids(graph_dir), _hash(graph_dir / "nodes.csv")
    if (cache_dir / "manifest.json").exists():
        anatomy = Anatomy.load(cache_dir)
        if anatomy.neuron_ids != ids or anatomy.manifest["graph_node_sha256"] != node_hash:
            raise ValueError("Anatomical cache belongs to a different graph export")
        return anatomy
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    coordinates = cache_dir / "coordinates.csv.gz"
    if not coordinates.exists():
        _download(COORDINATES_URL, coordinates)
    sources.append({"kind": "community_marked_positions", "url": COORDINATES_URL,
                    "sha256": _hash(coordinates), "bytes": coordinates.stat().st_size})
    matches = {root: [] for root in ids}
    with gzip.open(coordinates, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["root_id"] in matches:
                values = [int(value) for value in re.findall(r"-?\d+", row["position"])]
                if len(values) != 3:
                    raise ValueError("Malformed marked FlyWire position")
                matches[row["root_id"]].append({"root_id": row["root_id"],
                    "x_nm": values[0], "y_nm": values[1], "z_nm": values[2],
                    "supervoxel_id": row["supervoxel_id"]})
    absent = [root for root in ids if not matches[root]]
    if absent:
        raise ValueError(f"No authentic coordinate for {len(absent)} graph neurons: {absent[:5]}")
    columns = ["root_id", "x_nm", "y_nm", "z_nm", "supervoxel_id"]
    for filename, rows in [("positions.csv", [matches[root][0] for root in ids]),
                           ("marked_positions.csv", [row for root in ids for row in matches[root]])]:
        with (cache_dir / filename).open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    meshes = {}
    for key, name in [("neuropil", "FLYWIRE.ply"), ("whole_brain", "FLYWIRE_whole_brain.ply")]:
        path, url = cache_dir / name, FLYBRAINS_BASE + "meshes/" + name
        if not path.exists():
            _download(url, path)
        if not path.read_bytes().startswith(b"ply\n"):
            raise ValueError("Downloaded brain geometry is not a PLY mesh")
        meshes[key] = name
        sources.append({"kind": key, "url": url, "commit": FLYBRAINS_COMMIT,
                        "sha256": _hash(path), "bytes": path.stat().st_size})
    skeletons, failures = {}, {}
    def fetch_skeleton(root):
        binary = cache_dir / "skeletons" / f"{root}.bin"
        target, url = binary.with_suffix(".npz"), f"{SKELETON_BASE}/{root}"
        try:
            _download(url, binary, max_bytes=3_000_000)
            raw_hash = _hash(binary)
            vertices, edges, radius = decode_skeleton(binary.read_bytes())
            np.savez_compressed(target, vertices=vertices, edges=edges, radius=radius)
            binary.unlink()
            return root, {"kind": "skeleton", "root_id": root, "url": url,
                          "download_sha256": raw_hash, "cache_sha256": _hash(target),
                          "vertices": len(vertices), "edges": len(edges)}, None
        except (OSError, ValueError) as exc:
            binary.unlink(missing_ok=True)
            return root, None, str(exc)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for root, source, error in pool.map(fetch_skeleton, ids[:max(0, skeleton_count)]):
            if source:
                skeletons[root] = f"skeletons/{root}.npz"
                sources.append(source)
            else:
                failures[root] = error
    files = []
    for name in ["positions.csv", "marked_positions.csv", *meshes.values(), *skeletons.values()]:
        path = cache_dir / name
        files.append({"path": name, "sha256": _hash(path), "bytes": path.stat().st_size})
    manifest = {
        "schema_version": 1, "dataset": "FlyWire FAFB", "materialization": 783,
        "coordinate_space": SPACE, "units": "nm", "neuron_ids": ids,
        "graph_node_sha256": node_hash, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "position_semantics": "First community-marked coordinate per root in the public Codex export; not a soma or geometric centroid",
        "multiple_position_policy": "First source row; all source positions retained in marked_positions.csv",
        "registration": "Identity. Upstream FLYWIRE meshes already transformed from FAFB14 using the FAFB14->FlyWire re-alignment vector field. No further spatial transform applied.",
        "mesh_semantics": {"neuropil": "Synapse-density-derived neuropil envelope; cleaned, smoothed and downsampled upstream", "whole_brain": "EM tissue-mask-derived brain tissue envelope; upstream transformed into FlyWire space"},
        "skeleton_semantics": "Exact public materialization-783 vertices, edges and radius; lossless NPZ conversion, identity transform",
        "skeleton_selection": f"First {min(max(0, skeleton_count), len(ids))} graph IDs in nodes.csv order; optional context subset",
        "skeleton_failures": failures, "meshes": meshes, "skeletons": skeletons,
        "sources": sources, "files": files,
        "license_notes": "FlyWire CC BY-NC 4.0; navis-flybrains GPL-3.0 repository. See docs/ANATOMY.md.",
    }
    target = cache_dir / "manifest.json"
    temporary = target.with_suffix(".part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    return Anatomy.load(cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and validate registered FlyWire anatomy")
    parser.add_argument("--graph", type=Path, default=Path("data/cx512"))
    parser.add_argument("--cache", type=Path, default=Path("data/anatomy"))
    parser.add_argument("--skeletons", type=int, default=16)
    args = parser.parse_args()
    anatomy = ensure_anatomy(args.graph, args.cache, args.skeletons)
    print(json.dumps({"neurons": len(anatomy.neuron_ids), "meshes": list(anatomy.meshes),
                      "skeletons": len(anatomy.skeletons), "units": anatomy.units,
                      "coordinate_space": anatomy.coordinate_space}, indent=2))


if __name__ == "__main__":
    main()
