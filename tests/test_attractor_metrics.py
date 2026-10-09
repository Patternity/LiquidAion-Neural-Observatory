"""Pure metric helpers of the attractor diagnostic (no model)."""
import numpy as np

from liquidaion_fly.attractor import (answered, default_conditions, detect_period, js_divergence,
                                      kl_divergence, repeated_ngram_fraction, text_similarity, transient)


def test_period_detection_distinguishes_fixed_points_cycles_and_drift():
    a, b, c = (1,), (2,), (3,)
    assert detect_period([c, a, a, a, a, a], tail=4) == 1
    assert transient([c, a, a, a, a, a], 1) == 1
    assert detect_period([c, a, b, a, b, a, b, a], tail=4) == 2
    assert transient([c, a, b, a, b, a, b, a], 2) == 1
    assert detect_period([a, b, c, (4,), (5,), (6,)], tail=4) is None


def test_text_metrics():
    assert abs(text_similarity("Столица Франции — Париж.", "Столица Франции — Париж.") - 1.0) < 1e-12
    assert text_similarity("КВАРЦ", "56") == 0.0
    assert repeated_ngram_fraction("a b c d a b c d a b c d") > 0.5
    assert repeated_ngram_fraction("one two three four five") == 0.0
    assert answered("Столица — **Париж**.", ["париж"]) is True
    assert answered("Кварт.", ["кварц"]) is False
    assert answered("anything", None) is None


def test_divergences_are_zero_on_identity_and_positive_otherwise():
    x = np.array([1.0, 2.0, 3.0])
    assert abs(js_divergence(x, x)) < 1e-12 and abs(kl_divergence(x, x)) < 1e-12
    assert js_divergence(x, x[::-1]) > 0 and kl_divergence(x, x[::-1]) > 0
    assert js_divergence(x, x[::-1]) <= np.log(2)


def test_default_conditions_cover_required_controls():
    names = [c.name for c in default_conditions()]
    assert names[:2] == ["clean", "fly_off"]
    for s in ("0.05", "0.10", "0.25", "0.50"):
        assert f"steer_{s}" in names and f"steer_fb_{s}" in names
    assert any(c.reset_between_turns for c in default_conditions())
    assert any(c.mode == "random" for c in default_conditions())
