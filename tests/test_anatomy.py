"""Verify graph registration against bundled real data, not generated layouts."""
import csv
import gzip
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from liquidaion_fly.anatomy import Anatomy, decode_skeleton, ensure_anatomy

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/anatomy"
GRAPH = ROOT / "data/cx512"


def _ply_vertices(path):
    raw = path.read_bytes()
    offset = raw.index(b"end_header\n") + len(b"end_header\n")
    count = int(re.search(rb"element vertex (\d+)", raw[:offset])[1])
    return np.frombuffer(raw, dtype="<f4", count=count * 3, offset=offset).reshape(-1, 3)


def test_all_graph_nodes_have_exact_source_coordinates_in_graph_order():
    anatomy = Anatomy.load(CACHE)
    with (GRAPH / "nodes.csv").open() as stream:
        ids = [row["root_id"] for row in csv.DictReader(stream)]
    assert anatomy.neuron_ids == ids
    assert len(ids) == 512
    expected = {}
    with gzip.open(CACHE / "coordinates.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            if row["root_id"] in ids and row["root_id"] not in expected:
                expected[row["root_id"]] = [int(v) for v in re.findall(r"\d+", row["position"])]
    np.testing.assert_array_equal(anatomy.positions, np.asarray([expected[root] for root in ids]))
    # Real export anchor catches accidental voxel scaling, centering or mirroring.
    assert ids[0] == "720575940633145903"
    np.testing.assert_array_equal(anatomy.positions[0], [593640, 300232, 76280])


def test_registered_reference_units_and_actual_mesh_bounds():
    anatomy = Anatomy.load(CACHE)
    assert anatomy.coordinate_space == "FLYWIRE/FAFB14.1"
    assert anatomy.units == "nm"
    assert anatomy.manifest["materialization"] == 783
    tissue = _ply_vertices(anatomy.meshes["whole_brain"])
    neuropil = _ply_vertices(anatomy.meshes["neuropil"])
    assert tissue.shape == (48995, 3)
    assert neuropil.shape == (25047, 3)
    assert np.all(anatomy.positions >= tissue.min(0))
    assert np.all(anatomy.positions <= tissue.max(0))
    assert "already transformed from FAFB14" in anatomy.manifest["registration"]


def test_actual_skeleton_topology_and_nanometer_geometry():
    anatomy = Anatomy.load(CACHE)
    assert list(anatomy.skeletons) == anatomy.neuron_ids[:16]
    assert not anatomy.manifest["skeleton_failures"]
    for root in anatomy.skeletons:
        vertices, edges = anatomy.read_skeleton(root)
        assert np.isfinite(vertices).all()
        assert vertices.shape[1] == 3 and edges.shape[1] == 2
        assert int(edges.max()) < len(vertices)
        assert len(edges) == len(vertices) - 1
    vertices, edges = anatomy.read_skeleton(anatomy.neuron_ids[0])
    assert (len(vertices), len(edges)) == (16938, 16937)
    # The official marked point is actually near this neuron's skeleton.
    distance = np.linalg.norm(vertices - anatomy.positions[0], axis=1).min()
    assert distance < 2000


def test_offline_cache_reuse_needs_no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Verified cache must be usable offline")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    assert ensure_anatomy(GRAPH, CACHE).neuron_ids == Anatomy.load(CACHE).neuron_ids


def test_changed_graph_and_cache_fail_instead_of_repositioning(tmp_path):
    graph = tmp_path / "graph"
    graph.mkdir()
    shutil.copy(GRAPH / "nodes.csv", graph / "nodes.csv")
    with (graph / "nodes.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="different graph"):
        ensure_anatomy(graph, CACHE)
    cache = tmp_path / "cache"
    shutil.copytree(CACHE, cache)
    with (cache / "positions.csv").open("a") as stream:
        stream.write("corrupt data\n")
    with pytest.raises(ValueError, match="changed anatomical"):
        Anatomy.load(cache)


def test_bad_coordinate_reference_and_truncated_skeleton_fail(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = json.loads((CACHE / "manifest.json").read_text())
    manifest["coordinate_space"] = "FAFB14"
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coordinate reference"):
        Anatomy.load(cache)
    with pytest.raises(ValueError, match="Truncated"):
        decode_skeleton(b"123")
    with pytest.raises(ValueError, match="length"):
        decode_skeleton(struct.pack("<II", 3, 2))
