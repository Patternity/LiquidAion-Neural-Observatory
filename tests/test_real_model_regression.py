"""Opt-in multi-turn regression on the real pretrained Qwen3-0.6B and the real connectome.

Run with:  LIQUIDAION_REAL_MODEL=1 python -m pytest tests/test_real_model_regression.py -s
Uses CUDA when available, CPU otherwise (slow). Skipped by default.
"""
import os
from pathlib import Path

import pytest
import torch

from conftest import QWEN_REVISION, assert_no_hooks
from liquidaion_fly.debuglog import read_log

pytestmark = pytest.mark.skipif(os.environ.get("LIQUIDAION_REAL_MODEL") != "1",
                                reason="Set LIQUIDAION_REAL_MODEL=1 to run real-model regressions")

ROOT = Path(__file__).resolve().parents[1]
TURNS = [("Напиши только слово КВАРЦ.", ["кварц"]),
         ("Сколько будет 7 умножить на 8?", ["56"]),
         ("Назови столицу Франции.", ["париж", "paris"]),
         ("Тебе не хватает токенов на ответ.", None)]


@pytest.fixture(scope="module")
def real_engine():
    from liquidaion_fly.observatory_engine import ObservatoryConfig, ObservatoryEngine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ObservatoryEngine("Qwen/Qwen3-0.6B", ROOT / "data/cx512", device=device,
                             config=ObservatoryConfig(max_new_tokens=512, max_context_tokens=4096,
                                                      revision=QWEN_REVISION))


def test_each_branch_receives_its_intended_context(real_engine, tmp_path):
    engine = real_engine
    engine.new_session()
    engine.configure(steering_enabled=True, feedback_enabled=True)
    log = engine.enable_debug_log(tmp_path / "debug.jsonl", "full")
    replies = {"clean": [], "fly": []}
    for prompt, expected in TURNS:
        pair = engine.generate_pair(prompt)
        for branch in ("clean", "fly"):
            replies[branch].append(pair[branch]["text"])
            print(f"[{branch}] {pair[branch]['stop_reason']:>14} {len(pair[branch]['token_ids']):>3} tok: "
                  f"{pair[branch]['text'][:120]!r}")
            assert pair[branch]["stop_reason"] == "eos", "512 new tokens must suffice for these prompts"
        if expected:
            # The latest question must drive the Clean answer.
            assert any(word in pair["clean"]["text"].lower() for word in expected)
    engine.disable_debug_log()
    records = read_log(log)
    marker = "\x00M\x00"
    tail = engine._render([{"role": "user", "content": marker}]).split(marker, 1)[1]
    for index, (prompt, _) in enumerate(TURNS):
        clean, fly = records[2 * index], records[2 * index + 1]
        assert (clean["branch"], fly["branch"]) == ("clean", "fly")
        for record in (clean, fly):
            assert record["rendered_text"].endswith(prompt + tail)
            assert engine.llm.tokenizer.decode(record["input_ids"]) == record["rendered_text"]
            # Every earlier reply of this branch, and only this branch, is in its input.
            for earlier in replies[record["branch"]][:index]:
                assert f"<|im_start|>assistant\n{earlier}<|im_end|>" in record["rendered_text"]
        assert clean["hooks"]["attached"] == 0 and fly["hooks"]["present_after"] == 0
    assert_no_hooks(engine)


def test_fly_with_coupling_disabled_is_token_identical_to_clean(real_engine):
    engine = real_engine
    engine.new_session()
    engine.configure(steering_enabled=False, feedback_enabled=False)
    try:
        for prompt, _ in TURNS:
            pair = engine.generate_pair(prompt)
            assert pair["clean"]["token_ids"] == pair["fly"]["token_ids"]
    finally:
        engine.configure(steering_enabled=True, feedback_enabled=True)
        engine.new_session()
