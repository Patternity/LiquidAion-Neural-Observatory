"""Golden regression: the 'are you a fly?' exchange must reproduce token for token.

Reference token arrays are copied verbatim from publication/habr-fly-question/capture.json
(captured on CUDA, RTX 3070 Ti, float16) into tests/fixtures/fly_question_golden.json.
Verified on 2026-10-10: CUDA float16 and CPU float32 (AMD Ryzen 7 7700X, PyTorch 2.8.0)
both reproduce the arrays exactly. Other CPUs and the `+cpu` PyTorch build are unverified,
so the model test is opt-in (LIQUIDAION_REAL_MODEL=1); CI checks the fixture itself.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from conftest import QWEN_REVISION, qwen_tokenizer

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/fixtures/fly_question_golden.json").read_text(encoding="utf-8"))
CAPTURE = json.loads((ROOT / "publication/habr-fly-question/capture.json").read_text(encoding="utf-8"))
ORIGINAL = json.loads((ROOT / "publication/habr-fly-question/original-session-excerpt.json").read_text(encoding="utf-8"))
EOS = 151645


def test_fixture_is_the_published_capture_and_matches_the_original_session():
    turn = CAPTURE["turns"][0]
    for branch, length in (("clean", 66), ("fly", 59)):
        ids = GOLDEN[branch]["token_ids"]
        assert ids == turn[branch]["token_ids"], "Fixture must be a verbatim copy of capture.json"
        assert len(ids) == length and ids[-1] == EOS
        assert GOLDEN[branch]["text"] == turn[branch]["text"] == ORIGINAL["messages"][branch][1]["content"]
    assert GOLDEN["prompt"] == CAPTURE["prompts"][0] == ORIGINAL["messages"]["clean"][0]["content"]
    settings = ORIGINAL["settings"]
    assert (GOLDEN["strength"], GOLDEN["feedback"], GOLDEN["max_new_tokens"]) == \
        (settings["strength"], settings["feedback"], settings["max_new_tokens"]) == (0.7, 0.45, 512)
    assert GOLDEN["revision"] == settings["revision"] == QWEN_REVISION


def test_fixture_tokens_decode_to_the_recorded_text():
    tokenizer = qwen_tokenizer()
    for branch in ("clean", "fly"):
        ids = GOLDEN[branch]["token_ids"]
        assert tokenizer.decode(ids, skip_special_tokens=True) == GOLDEN[branch]["text"]
        assert tokenizer.decode(ids[-1:]) == "<|im_end|>"


@pytest.mark.skipif(os.environ.get("LIQUIDAION_REAL_MODEL") != "1",
                    reason="Set LIQUIDAION_REAL_MODEL=1 to run real-model golden regressions")
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_real_model_reproduces_golden_tokens_exactly(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from liquidaion_fly.observatory_engine import ObservatoryConfig, ObservatoryEngine
    engine = ObservatoryEngine(GOLDEN["model"], ROOT / "data/cx512", device=device,
                               config=ObservatoryConfig(revision=GOLDEN["revision"],
                                                        max_context_tokens=GOLDEN["max_context_tokens"]))
    engine.configure(strength=GOLDEN["strength"], feedback=GOLDEN["feedback"])
    pair = engine.generate_pair(GOLDEN["prompt"], max_new_tokens=GOLDEN["max_new_tokens"])
    for branch in ("clean", "fly"):
        got, expected = pair[branch]["token_ids"], GOLDEN[branch]["token_ids"]
        first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        assert got == expected, (f"{device} {branch}: first difference at token {first}; "
                                 f"lengths {len(got)} vs {len(expected)}")
