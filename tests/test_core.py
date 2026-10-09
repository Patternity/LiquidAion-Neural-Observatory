"""Synthetic fixtures test mechanics only; they are never research results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from liquidaion_fly.connectome import load_graph, largest_scc
from liquidaion_fly.reservoir import Dynamics, Reservoir, random_control


@pytest.fixture
def graph():
    return sparse.csr_matrix(np.array([[0., 2., 0.], [0., 0., 4.], [3., 0., 0.]]))


def test_source_target_orientation(tmp_path):
    pd.DataFrame({"root_id": ["720575940000000001", "720575940000000002"]}).to_csv(tmp_path / "nodes.csv", index=False)
    pd.DataFrame({"pre_root_id": ["720575940000000001"], "post_root_id": ["720575940000000002"], "syn_count": [7]}).to_csv(tmp_path / "edges.csv", index=False)
    w, ids = load_graph(tmp_path)
    assert ids[0] == "720575940000000001"
    np.testing.assert_array_equal(w @ np.array([1., 0.]), [0., 7.])


def test_missing_endpoint_rejected(tmp_path):
    pd.DataFrame({"root_id": ["1", "2"]}).to_csv(tmp_path / "nodes.csv", index=False)
    pd.DataFrame({"pre_root_id": ["3"], "post_root_id": ["2"], "syn_count": [7]}).to_csv(tmp_path / "edges.csv", index=False)
    with pytest.raises(ValueError, match="endpoint"):
        load_graph(tmp_path)


def test_known_integration_step(graph):
    r = Reservoir(graph, features=2)
    u = np.array([0.4, -0.2])
    expected = 0.2 * np.tanh(0.5 * r.input @ u)
    np.testing.assert_allclose(r.advance(u), expected)
    assert r.time == 0.2


def test_zero_state_stays_zero_without_input(graph):
    r = Reservoir(graph)
    np.testing.assert_array_equal(r.advance(steps=100), np.zeros(3))


def test_bounded_long_run_and_idle_decay(graph):
    r = Reservoir(graph)
    r.advance(np.ones(64), steps=1000)
    assert np.max(abs(r.state)) <= 1
    before = np.max(abs(r.state))
    r.advance(steps=1000)
    assert np.max(abs(r.state)) < before * 1e-4


def test_recurrence_changes_second_step(graph):
    a = Reservoir(graph)
    b = Reservoir(graph * 0.1, dynamics=Dynamics(gain=0))
    np.testing.assert_array_equal(a.advance(np.ones(64)), b.advance(np.ones(64)))
    assert not np.array_equal(a.advance(np.ones(64)), b.advance(np.ones(64)))


def test_snapshot_continuation_and_reset(graph, tmp_path):
    r = Reservoir(graph)
    r.advance(np.ones(64), steps=3)
    path = tmp_path / "state.npz"
    r.snapshot(path)
    expected = r.advance(np.zeros(64), steps=4)
    r.restore(path)
    np.testing.assert_array_equal(r.advance(np.zeros(64), steps=4), expected)
    r.reset()
    assert r.time == 0
    assert np.count_nonzero(r.state) == 0


def test_snapshot_identity_rejected(graph, tmp_path):
    r = Reservoir(graph)
    path = tmp_path / "state.npz"
    r.snapshot(path)
    with pytest.raises(ValueError, match="different"):
        Reservoir(graph, seed=99).restore(path)


@pytest.mark.parametrize("kwargs", [{"tau": 0}, {"dt": 2}, {"gain": 1}, {"input_strength": -1}, {"dt": float("nan")}])
def test_invalid_dynamics(kwargs):
    with pytest.raises(ValueError):
        Dynamics(**kwargs)


def test_control_matches_graph_budget(graph):
    c = random_control(graph, 43)
    assert c.shape == graph.shape and c.nnz == graph.nnz
    np.testing.assert_array_equal(np.sort(c.data), np.sort(graph.data))
    assert np.count_nonzero(c.diagonal()) == np.count_nonzero(graph.diagonal())
    np.testing.assert_array_equal(c.toarray(), random_control(graph, 43).toarray())


def test_authentic_packaged_graph():
    directory = Path(__file__).resolve().parents[1] / "data/cx512"
    w, ids = load_graph(directory)
    manifest = json.loads((directory / "provenance.json").read_text())
    assert len(ids) == manifest["nodes"] == 512
    assert w.nnz == manifest["edges"] == 16731
    assert int(w.sum()) == manifest["synapses"] == 310999
    assert len(largest_scc(w)) == len(ids)
    assert all(root.startswith("720575940") for root in ids)
