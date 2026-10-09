"""Context construction and Clean/Fly isolation regressions.

Random tiny weights: these are software tests of what reaches the model, not evidence
about language behavior. Tests using `qwen_engine` exercise the genuine Qwen3 tokenizer,
chat template and special tokens.
"""
from threading import Event

import numpy as np
import pytest
import torch

from conftest import assert_no_hooks, build_engine
from liquidaion_fly.debuglog import read_log
from liquidaion_fly.observatory_engine import ChatSession, ContextOverflowError, template_messages

DIVERSE_PROMPTS = ["Напиши только слово КВАРЦ.", "Сколько будет 7 умножить на 8?",
                   "Назови столицу Франции.", "Тебе не хватает токенов на ответ."]


def generation_prompt_tail(engine):
    marker = "\x00M\x00"
    return engine._render([{"role": "user", "content": marker}]).split(marker, 1)[1]


def expected_input(engine, session_messages, prompt):
    return engine._render(template_messages(session_messages) + [{"role": "user", "content": prompt}])


def spy_forward(engine, monkeypatch):
    """Record the input ids and live hook count of every model forward."""
    calls = []
    original = engine.llm.model.forward

    def forward(*args, **kwargs):
        calls.append({"ids": kwargs["input_ids"][0].tolist(), "hooks": engine._hook_count(),
                      "cache_length": kwargs["past_key_values"].get_seq_length()
                      if kwargs.get("past_key_values") is not None else 0})
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.llm.model, "forward", forward)
    return calls


def test_latest_user_message_reaches_both_branches_every_turn(qwen_engine, tmp_path):
    log = qwen_engine.enable_debug_log(tmp_path / "debug.jsonl", "full")
    expected = {"clean": [], "fly": []}
    for prompt in DIVERSE_PROMPTS:
        before = {branch: list(session.messages) for branch, session in qwen_engine.sessions.items()}
        pair = qwen_engine.generate_pair(prompt)
        for branch in ("clean", "fly"):
            expected[branch].append(expected_input(qwen_engine, before[branch], prompt))
            assert pair[branch]["prompt_ids"] == qwen_engine.llm.tokenizer(expected[branch][-1])["input_ids"]
    records = read_log(log)
    assert [record["branch"] for record in records] == ["clean", "fly"] * len(DIVERSE_PROMPTS)
    tail = generation_prompt_tail(qwen_engine)
    for index, prompt in enumerate(DIVERSE_PROMPTS):
        clean, fly = records[2 * index], records[2 * index + 1]
        assert clean["turn_index"] == fly["turn_index"] == index
        assert clean["latest_user_message"] == fly["latest_user_message"] == prompt
        assert clean["latest_user_sha256"] == fly["latest_user_sha256"]
        for record in (clean, fly):
            # The log holds the actual tokenized input, not the GUI transcript.
            assert record["rendered_text"] == expected[record["branch"]][index]
            assert record["rendered_text"].endswith(prompt + tail)
            assert qwen_engine.llm.tokenizer.decode(record["input_ids"]) == record["rendered_text"]
            assert record["input_token_count"] == len(record["input_ids"])
            assert record["history_messages_in_input"] == 2 * index
        assert not clean["coupled"] and fly["coupled"]
        assert clean["hooks"]["attached"] == 0 and fly["hooks"]["attached"] == fly["hooks"]["detached"] > 0
        assert clean["hooks"]["present_after"] == fly["hooks"]["present_after"] == 0
    assert_no_hooks(qwen_engine)


def test_multi_turn_histories_share_user_turns_but_keep_own_assistant_replies(engine):
    replies = {"clean": [], "fly": []}
    for prompt in ("first question", "second question", "third question"):
        pair = engine.generate_pair(prompt)
        for branch in replies:
            replies[branch].append(pair[branch]["text"])
    for branch, session in engine.sessions.items():
        assert [m["role"] for m in session.messages] == ["user", "assistant"] * 3
        assert [m["content"] for m in session.messages[0::2]] == ["first question", "second question", "third question"]
        assert [m["content"] for m in session.messages[1::2]] == replies[branch]
    assert engine.sessions["clean"].messages is not engine.sessions["fly"].messages


def test_clean_branch_is_bitwise_independent_of_fly_coupling(monkeypatch, tmp_path):
    def clean_tokens(**settings):
        engine = build_engine(monkeypatch, tmp_path / str(len(settings)), **settings)
        return [engine.generate_pair(prompt)["clean"]["token_ids"] for prompt in DIVERSE_PROMPTS]
    weak = clean_tokens(strength=0.0, feedback=0.0, steering_enabled=False, feedback_enabled=False)
    strong = clean_tokens(strength=1.0, feedback=1.0)
    assert weak == strong


def test_clean_inference_never_runs_through_hooks(engine, monkeypatch):
    calls = spy_forward(engine, monkeypatch)
    pair = engine.generate_pair("Hi")
    clean_calls = calls[:len(pair["clean"]["token_ids"])]
    fly_calls = calls[len(pair["clean"]["token_ids"]):]
    assert all(call["hooks"] == 0 for call in clean_calls)
    assert all(call["hooks"] == 2 for call in fly_calls)
    # A stray hook (e.g. a leaked Fly attachment) must stop Clean inference outright.
    handle = engine.llm.model.model.layers[0].register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(RuntimeError, match="hooks are present"):
            engine.generate_pair("Again")
    finally:
        handle.remove()
    assert_no_hooks(engine)


def test_truncated_reply_is_flagged_stored_and_closed_before_next_user_turn(qwen_engine, tmp_path):
    pair = qwen_engine.generate_pair("Расскажи длинную историю.", max_new_tokens=2)
    for branch in ("clean", "fly"):
        assert pair[branch]["stop_reason"] in {"max_new_tokens", "eos"}
    # Random weights rarely emit <|im_end|> in two steps; force the scenario if it did.
    truncated = [branch for branch in ("clean", "fly") if pair[branch]["truncated"]]
    assert truncated, "Expected at least one branch to hit max_new_tokens"
    for branch in truncated:
        stored = qwen_engine.sessions[branch].messages[-1]
        assert stored["stop_reason"] == "max_new_tokens" and stored["generated_tokens"] == 2
        assert stored["content"] == pair[branch]["text"]
    next_pair = qwen_engine.generate_pair("Новый вопрос.")
    for branch in truncated:
        rendered = qwen_engine.llm.tokenizer.decode(next_pair[branch]["prompt_ids"])
        text = pair[branch]["text"]
        assert f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>user\nНовый вопрос.<|im_end|>" in rendered


def test_qwen3_template_boundaries_and_end_of_turn_markers(qwen_engine):
    tokenizer = qwen_engine.llm.tokenizer
    for prompt in DIVERSE_PROMPTS[:3]:
        qwen_engine.generate_pair(prompt)
    plan = qwen_engine._build_context(qwen_engine.sessions["fly"], DIVERSE_PROMPTS[3], reserve=4)
    text = plan.text
    assert text.startswith("<|im_start|>user\n")
    assert text.endswith(f"<|im_start|>user\n{DIVERSE_PROMPTS[3]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert text.count("<|im_start|>") == 8 and text.count("<|im_end|>") == 7
    # Historical assistant turns carry no empty think block in the Qwen3 template.
    assert text.count("<think>") == 1
    ids = plan.ids[0].tolist()
    im_start, im_end = tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    assert ids.count(im_start) == 8 and ids.count(im_end) == 7
    from liquidaion_fly.llm import stop_token_ids
    assert im_end in stop_token_ids(tokenizer, qwen_engine.llm.model)


def test_both_qwen_end_of_turn_tokens_stop_generation(qwen_engine, monkeypatch):
    tokenizer = qwen_engine.llm.tokenizer
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    qwen_engine.llm.model.generation_config.eos_token_id = [tokenizer.eos_token_id, endoftext]
    original = qwen_engine.llm.model.forward

    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        result.logits[..., endoftext] = 1e4
        return result
    monkeypatch.setattr(qwen_engine.llm.model, "forward", forward)
    pair = qwen_engine.generate_pair("Hi", max_new_tokens=4)
    for branch in ("clean", "fly"):
        assert pair[branch]["token_ids"] == [endoftext]
        assert pair[branch]["stop_reason"] == "eos" and not pair[branch]["truncated"]
        assert pair[branch]["text"] == ""


def test_overflow_trims_whole_oldest_exchanges_without_losing_history(qwen_engine):
    prompts = [f"Вопрос номер {index}: " + "слово " * 6 for index in range(6)]
    for prompt in prompts:
        pair = qwen_engine.generate_pair(prompt)
    session = qwen_engine.sessions["clean"]
    assert len(session.messages) == 12, "Session history must be kept intact"
    plan = qwen_engine._build_context(session, "Последний вопрос.", reserve=4)
    assert plan.dropped_messages > 0 and plan.dropped_messages % 2 == 0
    assert plan.tokens <= plan.budget and plan.untrimmed_tokens > plan.budget
    assert plan.text.startswith("<|im_start|>user\n") and plan.messages[0]["role"] == "user"
    assert plan.messages[-1] == {"role": "user", "content": "Последний вопрос."}
    assert plan.messages[:-1] == template_messages(session.messages[plan.dropped_messages:])
    assert pair["clean"]["history_dropped_messages"] > 0 and pair["clean"]["context_near_limit"]
    assert any(event["type"] == "history_trimmed" for event in qwen_engine.recording.events)


def test_message_too_long_for_context_is_rejected_without_side_effects(qwen_engine):
    qwen_engine.generate_pair("Короткий вопрос.")
    before = {branch: list(session.messages) for branch, session in qwen_engine.sessions.items()}
    caches = {branch: session.cache for branch, session in qwen_engine.sessions.items()}
    turn = qwen_engine.turn_index
    with pytest.raises(ContextOverflowError, match="latest message"):
        qwen_engine.generate_pair("слово " * 400)
    assert {branch: session.messages for branch, session in qwen_engine.sessions.items()} == before
    assert all(caches[branch] is session.cache for branch, session in qwen_engine.sessions.items())
    assert qwen_engine.turn_index == turn
    assert_no_hooks(qwen_engine)


def test_kv_caches_are_per_branch_and_reused_only_on_exact_prefix(engine, monkeypatch):
    engine.generate_pair("first")
    clean, fly = engine.sessions["clean"], engine.sessions["fly"]
    assert clean.cache is not fly.cache
    fly_cache, fly_length = fly.cache, fly.cache.get_seq_length()
    calls = spy_forward(engine, monkeypatch)
    pair = engine.generate_pair("second")
    assert pair["clean"]["cache_event"] == pair["fly"]["cache_event"] == "reuse"
    # Clean ran first; it must not have extended or replaced the Fly cache meanwhile.
    clean_steps = len(pair["clean"]["token_ids"])
    assert fly.cache is fly_cache
    assert calls[clean_steps]["cache_length"] == fly_length == len(pair["fly"]["prompt_ids"]) - len(calls[clean_steps]["ids"])
    # Reuse must be numerically equivalent to recomputing the whole prompt.
    fresh = ChatSession(list(clean.messages[:-2]))
    rerun = engine._generate("clean", "second", 3, Event(), None, None, session=fresh)
    assert rerun["cache_event"] == "new" and rerun["token_ids"] == pair["clean"]["token_ids"]


def test_qwen_template_forces_cache_discard_after_each_turn(qwen_engine):
    # Qwen3 renders historical assistant turns without the empty think block, so the
    # previous cache never matches the next prompt prefix and is recomputed, not reused.
    qwen_engine.generate_pair("Первый.")
    pair = qwen_engine.generate_pair("Второй.")
    assert pair["clean"]["cache_event"] == pair["fly"]["cache_event"] == "discard_prefix_mismatch"
    assert not pair["clean"]["cache_reused"] and not pair["fly"]["cache_reused"]


def test_cancellation_before_fly_keeps_user_turns_aligned(engine):
    cancel = Event()
    pair = engine.generate_pair("Hi", cancel_event=cancel,
                                on_token=lambda branch, token, text: cancel.set())
    assert pair["clean"]["stop_reason"] == "cancelled" or len(pair["clean"]["token_ids"]) == 1
    assert pair["fly"]["stop_reason"] == "cancelled_before_start"
    for session in engine.sessions.values():
        assert [m["role"] for m in session.messages] == ["user", "assistant"]
        assert session.messages[0]["content"] == "Hi"
    assert engine.sessions["fly"].messages[1]["content"] == ""
    assert engine.sessions["fly"].messages[1]["stop_reason"] == "cancelled_before_start"
    nxt = engine.generate_pair("Next")
    assert nxt["clean"]["stop_reason"] != "cancelled_before_start"


def test_reset_and_fresh_session_semantics(engine):
    engine.generate_pair("Hi")
    engine.generate_pair("Again")
    old_id = engine.session_id
    messages = {branch: list(session.messages) for branch, session in engine.sessions.items()}
    clean_cache = engine.sessions["clean"].cache
    engine.reset_state()
    assert np.all(engine.reservoir.state == 0)
    assert {branch: session.messages for branch, session in engine.sessions.items()} == messages
    assert engine.sessions["fly"].cache is None and engine.sessions["clean"].cache is clean_cache
    engine.generate_pair("Warm")
    engine.clear_chats()
    assert np.linalg.norm(engine.reservoir.state) > 0, "Clear chats keeps the reservoir"
    assert all(not s.messages and s.cache is None for s in engine.sessions.values())
    assert engine.turn_index == 0
    engine.generate_pair("Warm")
    engine.new_session()
    assert engine.session_id != old_id and engine.turn_index == 0
    assert np.all(engine.reservoir.state == 0)
    assert all(not s.messages and s.cache is None and s.cached_ids is None for s in engine.sessions.values())
    fresh = engine.generate_pair("Hi")
    assert fresh["clean"]["cache_event"] == fresh["fly"]["cache_event"] == "new"
    assert fresh["clean"]["prompt_ids"] == fresh["fly"]["prompt_ids"]


def test_summary_debug_log_contains_no_conversation_text(engine, tmp_path):
    path = engine.enable_debug_log(tmp_path / "summary.jsonl")
    engine.generate_pair("secret words")
    raw = path.read_text()
    assert "secret" not in raw
    records = read_log(path)
    assert len(records) == 2
    for record in records:
        assert "rendered_text" not in record and "input_ids" not in record
        assert record["input_token_count"] > 0 and record["stop_reason"] in {"eos", "max_new_tokens"}
        assert record["kv_cache"]["event"] == "new"
        assert record["session_id"] == engine.session_id
    fly = records[1]
    assert fly["steering_enabled"] and fly["feedback_enabled"]
    assert fly["steering_strength"] == engine.config.strength
    assert fly["reservoir_rms_after"] != fly["reservoir_rms_before"]
    assert records[0]["reservoir_rms_before"] is None
    engine.disable_debug_log()
    engine.generate_pair("more")
    assert len(read_log(path)) == 2


def test_generation_limit_defaults_and_bounds(engine):
    from liquidaion_fly.observatory_engine import MAX_NEW_TOKENS_LIMIT, ObservatoryConfig
    assert ObservatoryConfig().max_new_tokens == 512
    assert ObservatoryConfig().max_context_tokens > 1024
    ObservatoryConfig(max_new_tokens=1024, max_context_tokens=4096).validate()
    with pytest.raises(ValueError):
        ObservatoryConfig(max_new_tokens=4096, max_context_tokens=4096).validate()
    with pytest.raises(ValueError):
        ObservatoryConfig(max_new_tokens=MAX_NEW_TOKENS_LIMIT + 1, max_context_tokens=10**6).validate()
    with pytest.raises(ValueError, match="positions"):
        engine.configure(max_context_tokens=engine.model_context_limit + 1)
    pair = engine.generate_pair("Hi", max_new_tokens=5)
    # Both branches obey the same limit.
    assert pair["clean"]["max_new_tokens"] == pair["fly"]["max_new_tokens"] == 5
    assert all(len(pair[b]["token_ids"]) <= 5 for b in ("clean", "fly"))


def test_failed_fly_branch_rolls_back_the_whole_pair(engine, monkeypatch):
    engine.generate_pair("first")
    before = {branch: list(session.messages) for branch, session in engine.sessions.items()}
    original = engine.llm.model.forward

    def fail(*args, **kwargs):
        if engine.coupling.handles:
            raise torch.cuda.OutOfMemoryError("Injected software-test OOM")
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.llm.model, "forward", fail)
    with pytest.raises(RuntimeError, match="Lower the Context"):
        engine.generate_pair("second")
    # Clean had already finished; its turn must be withdrawn so histories stay aligned.
    assert {branch: session.messages for branch, session in engine.sessions.items()} == before
    assert engine.sessions["clean"].cache is None and engine.sessions["fly"].cache is None
    assert engine.turn_index == 1
    assert_no_hooks(engine)
