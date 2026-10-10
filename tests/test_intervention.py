"""Random-weight tiny transformer tests validate hooks, not scientific outcomes."""
import numpy as np
import pytest
import torch
from scipy import sparse
from transformers import Qwen3Config, Qwen3ForCausalLM
from liquidaion_fly.llm import Coupling, compare_logits
from liquidaion_fly.reservoir import Reservoir


@pytest.fixture
def model():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def coupling(strength=0.4, mode="fly"):
    graph = sparse.csr_matrix([[0., 2., 0.], [0., 0., 4.], [3., 0., 0.]])
    return Coupling(Reservoir(graph, features=8), 32, 0, 1, strength, mode=mode)


def test_zero_strength_identical_and_hooks_cleaned(model):
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        original = model(ids).logits
        c = coupling(0)
        c.attach(model)
        altered = model(ids).logits
        c.detach()
    assert torch.equal(original, altered)
    assert c.records[0]["delta_relative_rms"] == 0
    assert c.states[0].shape == (3,)
    assert len(model.model.layers[1]._forward_hooks) == 0


def test_real_intervention_changes_last_token_only_and_cached_decode(model):
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        base = model(ids, use_cache=True).logits
        c = coupling()
        c.attach(model)
        steered = model(ids, use_cache=True)
        assert torch.equal(base[:, :-1], steered.logits[:, :-1])
        assert not torch.equal(base[:, -1], steered.logits[:, -1])
        model(torch.tensor([[5]]), past_key_values=steered.past_key_values, use_cache=True)
        c.detach()
    assert [r["sequence_length"] for r in c.records] == [4, 1]
    assert all(0 < r["delta_relative_rms"] <= c.strength + 1e-6 for r in c.records)
    assert not np.array_equal(c.states[0], c.states[1])


def test_baseline_cached_matches_uncached(model):
    with torch.inference_mode():
        first = model(torch.tensor([[1, 2, 3]]), use_cache=True)
        cached = model(torch.tensor([[4]]), past_key_values=first.past_key_values, use_cache=True).logits[:, -1]
        full = model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits[:, -1]
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-7)


def test_static_uses_frozen_state(model):
    c = coupling(mode="static")
    c.attach(model)
    with torch.inference_mode():
        model(torch.tensor([[1, 2, 3]]))
        first = c.static_state.copy()
        model(torch.tensor([[4, 5, 6]]))
    c.detach()
    np.testing.assert_array_equal(first, c.static_state)
    assert not np.array_equal(c.states[0], c.states[1])


def test_tensor_tuple_preservation():
    c = coupling()
    h = torch.randn(1, 3, 32)
    c.capture(None, (), (h, "metadata"))
    result = c.steer(None, (), (h, "metadata"))
    assert isinstance(result, tuple) and result[1] == "metadata"
    assert result[0].shape == h.shape
    assert result[0].dtype == h.dtype
    assert result[0].device == h.device


def test_comparison_rejects_different_context_shape():
    with pytest.raises(ValueError):
        compare_logits(np.zeros((2, 5)), np.zeros((3, 5)))


@pytest.mark.parametrize("capture_layer, steer_layer, order", [
    (0, 2, ["capture", "steer"]),   # capture below steering: the reservoir sees this token before steering
    (1, 1, ["capture", "steer"]),   # same layer: capture reads the output before steering modifies it
    (2, 0, ["steer", "capture"]),   # steering below capture: steering reads the previous token's state
])
def test_any_layer_order_one_record_and_two_updates_per_pass(model, capture_layer, steer_layer, order):
    graph = sparse.csr_matrix([[0., 2., 0.], [0., 0., 4.], [3., 0., 0.]])
    c = Coupling(Reservoir(graph, features=8), 32, capture_layer, steer_layer, 0.4)
    fired, advances = [], []
    capture, steer, advance = c.capture, c.steer, c.reservoir.advance
    c.capture = lambda *a: (fired.append("capture"), capture(*a))[1]
    c.steer = lambda *a: (fired.append("steer"), steer(*a))[1]
    c.reservoir.advance = lambda *a, **k: (advances.append(len(fired)), advance(*a, **k))[1]
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        base = model(ids, use_cache=True).logits
        c.attach(model)
        steered = model(ids, use_cache=True)
        model(torch.tensor([[5]]), past_key_values=steered.past_key_values, use_cache=True)
        c.detach()
    assert fired == order * 2
    assert len(advances) == 4, "Exactly two reservoir updates (capture + feedback) per forward pass"
    assert [r["step"] for r in c.records] == [0, 1] and [r["sequence_length"] for r in c.records] == [4, 1]
    assert len(c.states) == 2
    for record in c.records:
        assert (record["capture_layer"], record["steer_layer"]) == (capture_layer, steer_layer)
        assert {"capture_rms", "features", "delta_relative_rms", "feedback_input_norm"} <= set(record)
    if order[0] == "steer":
        # The first steering read the untouched initial state; nothing from this pass reached it.
        np.testing.assert_array_equal(c.states[0], np.zeros(3))
        assert c.records[0]["delta_relative_rms"] == 0
    else:
        assert not np.array_equal(c.states[0], np.zeros(3))
    assert c.records[1]["delta_relative_rms"] > 0
    assert torch.equal(base[:, :-1], steered.logits[:, :-1])
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)


def test_same_layer_capture_reads_unsteered_output(model):
    ids = torch.tensor([[1, 2, 3]])
    seen = []
    handle = model.model.layers[1].register_forward_hook(lambda m, a, out: seen.append(out[0] if isinstance(out, tuple)
                                                                                        else out))
    with torch.inference_mode():
        model(ids)
    handle.remove()
    unsteered_rms = float(seen[0][0, -1].float().square().mean().sqrt())
    graph = sparse.csr_matrix([[0., 2., 0.], [0., 0., 4.], [3., 0., 0.]])
    c = Coupling(Reservoir(graph, features=8), 32, 1, 1, 0.4)
    c.attach(model)
    with torch.inference_mode():
        model(ids)
    c.detach()
    assert c.records[0]["capture_rms"] == pytest.approx(unsteered_rms, rel=1e-6)
    assert c.records[0]["delta_relative_rms"] > 0


def test_coupling_rejects_layers_outside_the_model(model):
    graph = sparse.csr_matrix([[0., 2., 0.], [0., 0., 4.], [3., 0., 0.]])
    with pytest.raises(ValueError):
        Coupling(Reservoir(graph, features=8), 32, -1, 1)
    c = Coupling(Reservoir(graph, features=8), 32, 3, 0)
    with pytest.raises(ValueError):
        c.attach(model)
    assert not c.handles
