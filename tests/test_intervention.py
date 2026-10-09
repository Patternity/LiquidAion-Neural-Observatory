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
