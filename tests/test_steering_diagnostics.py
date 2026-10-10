"""Steering diagnostics: numerics, step alignment and zero effect on inference (tiny random weights)."""
import json
import math
from threading import Event

import numpy as np
import pytest
import torch

from conftest import assert_no_hooks
from liquidaion_fly.observatory_engine import ChatSession
from liquidaion_fly.steering_diagnostics import (SteeringObserver, _forced_loop, cosine, decompose, run_probe,
                                                 spectrum)


def test_cosine_norm_and_zero_conventions():
    assert cosine(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(1)
    assert cosine(np.array([1.0, 0]), np.array([0, 3.0])) == pytest.approx(0)
    assert cosine(np.array([1.0, 0]), np.array([-1.0, 0])) == pytest.approx(-1)
    assert math.isnan(cosine(np.zeros(3), np.ones(3))) and math.isnan(cosine(None, np.ones(3)))
    parallel, orthogonal, ratio = decompose(np.array([3.0, 4.0]), np.array([2.0, 0.0]))
    assert (parallel, orthogonal, ratio) == pytest.approx((3, 4, 0.8))
    assert all(math.isnan(x) for x in decompose(np.zeros(2), np.array([1.0, 0])))
    fixed = spectrum(np.outer(np.linspace(1, 2, 10), [1.0, 2.0, 0.5]))
    assert fixed["uncentered"]["top1_energy_fraction"] == pytest.approx(1)
    assert spectrum(np.ones((2, 3)))["samples"] == 2 and "centered" not in spectrum(np.ones((2, 3)))


def prompt_ids(engine, text="Hi"):
    return engine._build_context(ChatSession([]), text, 4).ids


@pytest.mark.parametrize("bypasses", [[[0, 2, True]], [[2, 0, True]], [[0, 2, True], [1, 1, True]]])
def test_observer_has_no_effect_and_aligns_steps(engine, bypasses):
    engine.configure(bypasses=bypasses, strength=0.6)
    ids, tokens = prompt_ids(engine), [5, 9, 2, 7]
    runs = []
    for observed in (False, True):
        engine.reset_state()
        observer = SteeringObserver() if observed else None
        engine.coupling.observer = observer
        try:
            out = _forced_loop(engine, ids, tokens, 4, engine.coupling, observer, False, Event())
        finally:
            engine.coupling.observer = None
        runs.append((np.stack(out["logits"]), engine.reservoir.state.copy(), observer))
    np.testing.assert_array_equal(runs[0][0], runs[1][0]), "Identical logits with diagnostics on and off"
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    passes = runs[1][2].passes
    assert len(passes) == 4
    mode = engine.coupling_info()["coupling_mode"]
    for index, data in enumerate(passes):
        if index:
            np.testing.assert_array_equal(data["state_before"], passes[index - 1]["state_after"])
        steering = next(iter(data["steering"].values()))
        if mode == "synchronous_multi" or bypasses[0][0] > bypasses[0][1]:
            np.testing.assert_array_equal(steering["state_used"], data["state_before"])
        else:  # legacy forward: steering reads the state after this token's capture update
            assert not np.array_equal(steering["state_used"], data["state_before"])
        compact = steering["state_used"] @ engine.coupling.readout
        expected = compact @ engine.coupling.decoder.numpy().astype(np.float64)
        np.testing.assert_allclose(steering["centered"], expected - expected.mean(), atol=1e-4)
        assert cosine(steering["applied_delta"], steering["centered"]) > 0.999 or \
            np.linalg.norm(steering["centered"]) < 1e-12
    assert_no_hooks(engine)


def test_probe_loop_matches_the_engine_scientific_path_and_counterfactual(engine, tmp_path):
    engine.configure(bypasses=[[0, 2, True], [2, 1, True]])
    engine.new_session()
    result = run_probe(engine, "Hi", 4, tmp_path / "probe", forced_tokens=[5, 9, 2, 7])
    assert engine.coupling.observer is None and not engine.coupling.handles
    assert_no_hooks(engine)
    rows = result["rows"]
    assert [r["step"] for r in rows] == [0, 1, 2, 3] and result["metadata"]["token_ids"] == [5, 9, 2, 7]
    assert rows[0]["kl_clean_off_history"] == 0 and not rows[0]["top1_changed_history"], \
        "Step 0 without steering is exactly the Clean prefill"
    assert math.isnan(rows[0]["steering_cos_prev"]) and math.isnan(rows[0]["steering_cos_ref"]), \
        "Synchronous step 0 steers from the zero state: direction undefined, not 0 or 1"
    assert rows[1]["steering_cos_ref"] == pytest.approx(1) and result["metadata"]["reference_steps"]["steering"] == 1
    assert all(r["direction_max_dev_across_layers"] == 0 for r in rows), "One decoder for every steering layer"
    for name in ("token_metrics.csv", "metadata.json", "summary.md", "reservoir_states.npz", "steering_vectors.npz",
                 "plots/04_steering_cos_ref.png", "plots/07_kl.png"):
        assert (tmp_path / "probe" / name).exists(), name
    metadata = json.loads((tmp_path / "probe" / "metadata.json").read_text())
    assert metadata["coupling"]["coupling_mode"] == "synchronous_multi" and metadata["reservoir"]["updates_per_token"] == 1
    with pytest.raises(FileExistsError):
        run_probe(engine, "Hi", 2, tmp_path / "probe")


def test_probe_fly_logits_equal_engine_forced_generation(engine):
    """The diagnostic loop is the engine's own forced path: same logits as _generate(scientific=True)."""
    engine.configure(strength=0.7)
    engine.reset_state()
    initial = engine.reservoir.state.copy()
    expected = engine._generate("fly", "Hi", 4, Event(), None, None, session=ChatSession([]),
                                coupling=engine.coupling, forced_tokens=[5, 9, 2, 7], scientific=True)["logits"]
    engine.reservoir.state = initial.copy()
    engine.coupling.clear_trace()
    out = _forced_loop(engine, prompt_ids(engine), [5, 9, 2, 7], 4, engine.coupling, None, False, Event())
    np.testing.assert_allclose(np.stack(out["logits"]), expected.astype(np.float64), rtol=0, atol=1e-6)
