"""Research plugins: field numerics on tiny fixtures, anatomy audit, shadow mode, leak-dynamics runs."""
import csv
import json
import math
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from scipy import sparse

from conftest import assert_no_hooks
from liquidaion_fly.dopamine_field import Field3D, FieldConfig, LeakResponse, audit_anatomy, simulate
from liquidaion_fly.plugins import ExperimentAPI, load_plugin

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "plugins" / "examples"


def point_source_field(boundary="noflux", k=0.0):
    config = FieldConfig(diffusion_um2_per_s=10.0, clearance_per_s=k, voxel_um=1.0, boundary=boundary)
    field = Field3D(np.zeros(3), np.full(3, 8.0), config)
    field.C[4, 4, 4] = 1.0
    return field


def test_field_diffusion_clearance_positivity_and_stability():
    field = point_source_field()
    dt = 0.9 * field.dt_limit
    for _ in range(200):
        field.step(dt, np.empty((0, 3), int), np.empty(0))
    assert field.C.min() >= 0 and field.C.sum() == pytest.approx(1.0, rel=1e-12), "No-flux, k = 0 conserves mass"
    assert field.C[4, 4, 4] < 0.05 and field.C[0, 0, 0] > 0, "The pulse spreads; it is not broadcast instantly"
    early = point_source_field()
    early.step(dt, np.empty((0, 3), int), np.empty(0))
    assert early.C[0, 0, 0] == 0 and early.C[4, 4, 5] > 0, "One step reaches only neighbouring voxels"
    cleared = point_source_field(k=2.0)
    dt = 0.9 * cleared.dt_limit
    for _ in range(50):
        cleared.step(dt, np.empty((0, 3), int), np.empty(0))
    assert cleared.C.sum() == pytest.approx((1 - 2.0 * dt) ** 50, rel=1e-10), "Clearance removes k·C exactly"
    absorbing = point_source_field("absorbing")
    for _ in range(400):
        absorbing.step(0.9 * absorbing.dt_limit, np.empty((0, 3), int), np.empty(0))
    assert absorbing.C.sum() < 0.9 and absorbing.boundary_loss > 0.1 and absorbing.C.min() >= 0
    with pytest.raises(ValueError, match="positivity"):
        point_source_field().step(2 * field.dt_limit, np.empty((0, 3), int), np.empty(0))


def test_leak_response_limits_and_directions():
    c = np.concatenate([np.linspace(-10, 10, 101), [1e9, -1e9]])
    for direction in (1, -1):
        for saturating in (True, False):
            response = LeakResponse(baseline_concentration=1.0, sensitivity=3.0, direction=direction,
                                    saturating=saturating)
            leak = response(c)
            assert leak.min() >= 0.05 and leak.max() <= 0.35
            assert response(np.array([1.0]))[0] == pytest.approx(0.20)
            assert (response(np.array([2.0]))[0] > 0.2) == (direction == 1)
    with pytest.raises(ValueError):
        LeakResponse(leak_min=0.3, leak_baseline=0.2).validate()


def tiny_geometry():
    positions = np.array([[0, 0, 0], [30, 0, 0], [0, 30, 0], [30, 30, 10], [60, 10, 5]], dtype=float)
    graph = sparse.csr_matrix((np.ones(5), ([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])), shape=(5, 5))
    return {"positions_um": positions, "da_indices": [0], "graph": graph}


def test_simulation_on_a_tiny_fixture_is_nonnegative_bounded_and_mass_consistent():
    states = np.zeros((30, 5))
    states[:10, 0] = 0.8          # DA neuron active for 10 steps
    states[:, 1] = -0.9           # negative activation of a non-DA neuron: no release, no negative C
    config = FieldConfig(diffusion_um2_per_s=50.0, clearance_per_s=1.0, voxel_um=5.0, margin_um=10.0,
                         token_interval_s=0.1, post_seconds=3.0, release_tau_s=0.0)
    result = simulate(states, tiny_geometry(), config, LeakResponse(sensitivity=100.0))
    assert result["min_concentration"] >= 0 and result["max_mass_balance_error"] < 1e-9
    assert 0.05 <= result["leak_range"][0] and result["leak_range"][1] <= 0.35
    c = result["concentration_at_neurons"]
    assert c[10, 0] > c[10, 4] > 0, "Nearer the source the concentration is higher"
    assert result["release_rates"][:10].min() > 0 and result["release_rates"][10:].max() == 0
    assert result["clearance_half_time_s"] == pytest.approx(math.log(2), abs=0.15), "No-flux: total decays at rate k"
    assert result["distance_vs_graph"]["graph_hops_from_nearest_da"][1] == 1


def test_anatomy_audit_on_real_data_and_missing_anatomy(tmp_path):
    audit = audit_anatomy(ROOT / "data" / "cx512", ROOT / "data" / "anatomy")
    assert audit["da_neurons"]["count"] == 16 and audit["neurons"] == 512
    assert audit["coordinates"]["neurons_with_position"] == 512 and audit["coordinates"]["units"] == "nm"
    assert audit["skeletons"]["da_neurons"] == 0 and not audit["synapse_positions"]
    assert audit["spatial_field_possible"] and "marker-proxy" in audit["grounding"]
    missing = audit_anatomy(ROOT / "data" / "cx512", tmp_path / "no-anatomy")
    assert not missing["spatial_field_possible"] and missing["coordinates"] is None


def run_example(engine, tmp_path, name, **overrides):
    run = load_plugin(EXAMPLES / name)
    run.__globals__.update(overrides)
    lines = []
    api = ExperimentAPI(engine, EXAMPLES / name, Event(), lines.append, output_root=tmp_path / "out")
    run(api)
    return api, lines


def test_dopamine_plugin_shadow_mode_with_and_without_anatomy(engine, tmp_path):
    api, lines = run_example(engine, tmp_path, "dopamine_field_prototype.py", TOKENS=12, INJECTION=(3, "Hi"),
                             ANATOMY_DIR=str(tmp_path / "none"))
    assert (api.output_dir / "data_availability.json").exists() and not (api.output_dir / "metadata.json").exists()
    assert any("only data_availability.json" in line for line in lines)
    anatomy = tmp_path / "anatomy"
    anatomy.mkdir()
    (anatomy / "manifest.json").write_text(json.dumps({"coordinate_space": "test", "units": "nm",
                                                       "position_semantics": "fixture", "skeletons": {}}))
    (anatomy / "positions.csv").write_text("root_id,x_nm,y_nm,z_nm\n1,0,0,0\n2,20000,0,0\n3,0,20000,10000\n")
    engine.new_session()
    api, lines = run_example(engine, tmp_path, "dopamine_field_prototype.py", TOKENS=12, INJECTION=(3, "Hi"),
                             ANATOMY_DIR=str(anatomy), POST_SECONDS=1.0, SENSITIVITY=10.0)
    metadata = json.loads((api.output_dir / "metadata.json").read_text())
    assert metadata["min_concentration"] >= 0 and metadata["source"]["cued_passes"] == metadata["source"]["injected_passes"] > 0
    assert 0.05 <= metadata["leak_range"][0] <= metadata["leak_range"][1] <= 0.35
    assert len(list((api.output_dir / "plots").glob("*.png"))) == 7
    assert_no_hooks(engine)


def test_leak_dynamics_plugin_resets_reuses_tokens_and_does_not_reload(engine, tmp_path):
    loads = len(engine._test_load_calls)
    api, lines = run_example(engine, tmp_path, "leak_rate_dynamics.py", MODEL=engine.profile.key, BYPASSES=[(0, 2), (2, 1)],
                             LEAK_RATES=[0.1, 0.5], MAX_TOKENS=4, WARMUP_TOKENS=0, IDLE_STEPS=3)
    assert len(engine._test_load_calls) == loads, "The loaded model is reused"
    assert api.output_dir.parent.name == "leak-rate-dynamics"
    for name in ("metadata.json", "token_metrics.csv", "summary.csv", "idle_metrics.csv", "matched_amplitude.json",
                 "reservoir_states.npz", "steering_vectors.npz", "plots/02_relative_intervention.png",
                 "plots/08_idle_retention.png"):
        assert (api.output_dir / name).exists(), name
    runs = sorted(api.output_dir.glob("leak-*/measure"))
    first, second = (json.loads((run / "metadata.json").read_text()) for run in runs)
    assert first["token_ids"] == second["token_ids"], "Every leak rate sees the same forced tokens"
    assert first["reservoir"]["initial_state_rms"] == second["reservoir"]["initial_state_rms"] == 0, "Reset between runs"
    assert (first["reservoir"]["leak"], second["reservoir"]["leak"]) == (0.1, 0.5)
    with (api.output_dir / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert [float(r["leak"]) for r in summary] == [0.1, 0.5] and all(r["reservoir_updates_per_token"] == "1" for r in summary)
    idle = list(csv.DictReader((api.output_dir / "idle_metrics.csv").open()))
    assert len(idle) == 6 and float(idle[0]["simulation_time_since_stop"]) == pytest.approx(0.2)
    api, _ = run_example(engine, tmp_path, "leak_rate_dynamics.py", MODEL=None, BYPASSES=None, LEAK_RATES=[0.2],
                         MAX_TOKENS=3, WARMUP_TOKENS=2, MODE="free_generation", SAVE_VECTORS=False, SAVE_PLOTS=False)
    rows = list(csv.DictReader((api.output_dir / "token_metrics.csv").open()))
    assert [r["phase"] for r in rows].count("warmup") == 2 and [int(r["index"]) for r in rows][:2] == [-2, -1]
    assert all(not r.get("kl_clean_fly") for r in rows if r["phase"] == "measure"), "No Clean-vs-Fly KL in free mode"
    assert not (api.output_dir / "steering_vectors.npz").exists() and not (api.output_dir / "plots").exists()
    assert_no_hooks(engine)


def test_state_trace_records_every_forward_pass_without_changing_the_run(engine, tmp_path):
    from liquidaion_fly.continuous import ContinuousConfig, ContinuousRun, load_run
    base = dict(seed_text="Once", window=8, total_tokens=6, checkpoints=(), probe_tokens=1, cue_target="da",
                injections=((2, "Hi!"),))
    runs = {}
    for traced in (False, True):
        engine.new_session()
        ContinuousRun(engine, ContinuousConfig(**base, state_trace=traced), tmp_path / str(traced)).run()
        runs[traced] = load_run(tmp_path / str(traced))
    assert runs[False]["state_trace"] is None
    assert runs[False]["tokens"] == runs[True]["tokens"], "Recording only: the run is unchanged"
    trace = runs[True]["state_trace"]
    summary = runs[True]["summary"]
    assert len(trace["states"]) == summary["forward_calls"] == summary["stream_tokens"]
    assert int((~trace["autonomous"]).sum()) == summary["external_tokens"] == 3
    assert trace["cue_applied"].sum() == 3
