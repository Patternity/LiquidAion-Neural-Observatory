"""Shared random-weight tiny-transformer fixtures: software tests, not scientific evidence."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from liquidaion_fly.observatory_engine import ObservatoryConfig, ObservatoryEngine

QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class TestTokenizer:
    """One token per character; a plain-text chat template."""
    __test__ = False
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"{m['role']}:{m['content']}\n" for m in messages) + "assistant:"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[ord(c) % 64 for c in text]])}

    def decode(self, tokens, **kwargs):
        return "".join(chr(token + 64) for token in tokens)


def write_graph(directory):
    directory.mkdir(parents=True)
    (directory / "nodes.csv").write_text("root_id,nt_type\n1,ACH\n2,DA\n3,GABA\n")
    (directory / "edges.csv").write_text("pre_root_id,post_root_id,syn_count\n1,2,2\n2,3,3\n3,1,4\n")
    return directory


def build_engine(monkeypatch, tmp_path, tokenizer=None, vocab_size=64, registry=None, **config):
    torch.set_num_threads(1)
    torch.manual_seed(321)
    model_config = Qwen3Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model_config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(model_config).eval()
    calls = []

    def load(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(model=model, tokenizer=tokenizer or TestTokenizer(),
                               device=torch.device("cpu"), revision="software-test")
    monkeypatch.setattr("liquidaion_fly.observatory_engine.LocalLLM", load)
    settings = dict(capture_layer=0, steer_layer=1, features=8, max_context_tokens=96,
                    max_new_tokens=3, threads=1)
    settings.update(config)
    result = ObservatoryEngine("software-test-only", write_graph(tmp_path / "graph"),
                               config=ObservatoryConfig(**settings), registry=registry)
    result._test_load_calls = calls
    return result


@pytest.fixture
def engine(monkeypatch, tmp_path):
    return build_engine(monkeypatch, tmp_path)


def qwen_tokenizer():
    """The real Qwen3 tokenizer and chat template from the local Hugging Face cache, or skip."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=QWEN_REVISION, local_files_only=True)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"Qwen3 tokenizer is not cached locally: {exc}")


@pytest.fixture
def qwen_engine(monkeypatch, tmp_path):
    """Random tiny weights behind the genuine Qwen3 tokenizer, template and special tokens."""
    tokenizer = qwen_tokenizer()
    return build_engine(monkeypatch, tmp_path, tokenizer=tokenizer, vocab_size=len(tokenizer),
                        max_context_tokens=160, max_new_tokens=4)


def assert_no_hooks(engine):
    assert all(not layer._forward_hooks for layer in engine.llm.model.model.layers)
