"""Sliding-window continuous generation: software tests on random tiny weights.

These check bookkeeping (window bounds, reservoir updates, probes, files), not language.
"""
from threading import Event

import numpy as np
import pytest

from conftest import assert_no_hooks
from liquidaion_fly.continuous import (ContinuousConfig, ContinuousRun, load_run, parse_conditions,
                                       repetition_metrics, run_experiment, seed_token_ids)

SEED = "Once upon a winter"


def spy_inputs(engine, monkeypatch):
    calls = []
    original = engine.llm.model.forward

    def forward(*args, **kwargs):
        calls.append({"ids": kwargs["input_ids"][0].tolist(), "use_cache": kwargs.get("use_cache"),
                      "cache": kwargs.get("past_key_values"), "hooks": engine._hook_count()})
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.llm.model, "forward", forward)
    return calls


def config(**values):
    settings = dict(seed_text=SEED, window=8, total_tokens=40, condition="fly", checkpoints=(), probe_tokens=4,
                    snapshot_interval=8)
    settings.update(values)
    return ContinuousConfig(**settings)


def test_window_is_bounded_and_drops_old_tokens_exactly(engine, monkeypatch):
    calls = spy_inputs(engine, monkeypatch)
    run = ContinuousRun(engine, config(window=8, total_tokens=40))
    summary = run.run()
    seed = seed_token_ids(engine.llm.tokenizer, SEED)
    history = seed + run.generated
    assert summary["generated_tokens"] == 40 and summary["stop_reason"] == "total_tokens"
    assert len(calls) == 40 and summary["max_window_seen"] == 8
    for step, call in enumerate(calls):
        expected = history[:len(seed) + step][-8:]
        assert call["ids"] == expected, f"step {step} model input is not the latest window"
        assert len(call["ids"]) <= 8
        assert call["use_cache"] is False and call["cache"] is None
    assert summary["window_shifts"] == len(history) - 8
    assert_no_hooks(engine)


def test_reservoir_persists_and_advances_once_per_capture_and_feedback(engine):
    run = ContinuousRun(engine, config(total_tokens=30, feedback=0.25))
    reservoir = run.reservoir
    times = []
    original = run.step_once

    def step():
        row = original()
        times.append(reservoir.time)
        return row
    run.step_once = step
    summary = run.run()
    assert run.reservoir is reservoir and not run.resets
    assert np.linalg.norm(reservoir.state) > 0
    assert all(b > a for a, b in zip(times, times[1:])), "Reservoir time must advance every step"
    # Exactly one capture update plus one feedback update per generated token.
    assert summary["reservoir_advances"] == 2 * summary["forward_calls"] == 60
    assert reservoir.time == pytest.approx(60 * reservoir.dynamics.dt)
    no_feedback = ContinuousRun(engine, config(total_tokens=10, feedback=0.0)).run()
    assert no_feedback["reservoir_advances"] == no_feedback["forward_calls"] == 10


def test_periodic_reset_happens_exactly_at_interval_steps(engine, tmp_path):
    run = ContinuousRun(engine, config(condition="fly_reset", reset_interval=7, total_tokens=30), tmp_path / "r")
    summary = run.run()
    assert summary["resets"] == [7, 14, 21, 28]
    rows = load_run(tmp_path / "r")["telemetry"]
    assert [row["step"] for row in rows if row["reset"]] == [7, 14, 21, 28]
    assert json_metadata(tmp_path / "r")["config"]["reset_interval"] == 7
    with pytest.raises(ValueError):
        ContinuousRun(engine, config(condition="fly_reset", reset_interval=None))
    with pytest.raises(ValueError):
        ContinuousRun(engine, config(condition="fly", reset_interval=5))


def json_metadata(path):
    import json
    return json.loads((path / "metadata.json").read_text())


def test_clean_runs_without_hooks_and_leaves_reservoir_untouched(engine, monkeypatch):
    calls = spy_inputs(engine, monkeypatch)
    live = engine.reservoir.state.copy()
    run = ContinuousRun(engine, config(condition="clean", total_tokens=12))
    run.run()
    assert run.reservoir is None and all(call["hooks"] == 0 for call in calls)
    np.testing.assert_array_equal(engine.reservoir.state, live)
    handle = engine.llm.model.model.layers[0].register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(RuntimeError, match="hooks are present"):
            ContinuousRun(engine, config(condition="clean", total_tokens=2)).run()
    finally:
        handle.remove()


def test_checkpoint_probes_do_not_change_the_main_run(engine):
    plain = ContinuousRun(engine, config(total_tokens=40))
    plain.run()
    probed = ContinuousRun(engine, config(total_tokens=40, checkpoints=(8, 16, 32), probe_tokens=4))
    summary = probed.run()
    assert probed.generated == plain.generated
    np.testing.assert_array_equal(probed.reservoir.state, plain.reservoir.state)
    assert [p["step"] for p in summary["probes"]] == [8, 16, 32]
    for probe in summary["probes"]:
        assert probe["probe_tokens"] == 4 and probe["retained_first_logits_equal_main"] is True
        assert probe["retained_vs_reset"]["mean_kl_baseline_to_condition"] >= 0
    # A probe opened near the end is still evaluated on the tokens that exist.
    late = ContinuousRun(engine, config(total_tokens=10, checkpoints=(8,), probe_tokens=6)).run()
    assert late["probes"][0]["probe_tokens"] == 2
    assert_no_hooks(engine)


def test_identical_conditions_reproduce_tokens_and_states(engine):
    first = ContinuousRun(engine, config(total_tokens=25, condition="random"))
    first.run()
    second = ContinuousRun(engine, config(total_tokens=25, condition="random"))
    second.run()
    assert first.generated == second.generated
    np.testing.assert_array_equal(first.reservoir.state, second.reservoir.state)
    fly = ContinuousRun(engine, config(total_tokens=25))
    fly.run()
    assert fly.reservoir.fingerprint != first.reservoir.fingerprint


def test_saved_run_replays_tokens_text_and_telemetry(engine, tmp_path):
    ContinuousRun(engine, config(total_tokens=30, checkpoints=(8,)), tmp_path / "fly").run()
    saved = load_run(tmp_path / "fly")
    tokenizer = engine.llm.tokenizer
    generated = saved["tokens"]["generated"]
    assert [row["token_id"] for row in saved["telemetry"]] == generated
    assert [row["step"] for row in saved["telemetry"]] == list(range(30))
    assert saved["text"] == SEED + tokenizer.decode(generated)
    assert "".join(row["fragment"] for row in saved["telemetry"]) == tokenizer.decode(generated)
    assert saved["states"]["steps"][0] == 0 and saved["states"]["steps"][-1] == 30
    assert saved["states"]["states"].shape == (len(saved["states"]["steps"]), 3)
    assert saved["summary"]["probes"][0]["step"] == 8
    assert saved["metadata"]["tokenization"].startswith("raw continuation")


def test_cancellation_leaves_consistent_files(engine, tmp_path):
    cancel = Event()

    def progress(row, run):
        if row["step"] == 11:
            cancel.set()
    summary = ContinuousRun(engine, config(total_tokens=100, checkpoints=(4,)), tmp_path / "c").run(cancel, progress)
    saved = load_run(tmp_path / "c")
    assert summary["stop_reason"] == "cancelled" and summary["generated_tokens"] == 12
    assert len(saved["telemetry"]) == len(saved["tokens"]["generated"]) == 12
    assert saved["summary"]["stop_reason"] == "cancelled"
    assert saved["text"] == SEED + engine.llm.tokenizer.decode(saved["tokens"]["generated"])
    assert_no_hooks(engine)


def test_eos_policies_stop_or_suppress_and_record(engine, tmp_path):
    probe = ContinuousRun(engine, config(total_tokens=5))
    probe.run()
    engine.llm.tokenizer.eos_token_id = probe.generated[2]
    stopped = ContinuousRun(engine, config(total_tokens=20)).run()
    assert stopped["stop_reason"] == "eos" and stopped["generated_tokens"] == 2
    forced = ContinuousRun(engine, config(total_tokens=20, eos_policy="suppress"), tmp_path / "s")
    summary = forced.run()
    assert summary["generated_tokens"] == 20 and summary["stop_reason"] == "total_tokens"
    assert summary["eos_suppressed"] and summary["eos_suppressed"][0]["step"] == 2
    assert probe.generated[2] not in forced.generated
    rows = load_run(tmp_path / "s")["telemetry"]
    assert [row["step"] for row in rows if row["eos_suppressed"] is not None] == \
        [event["step"] for event in summary["eos_suppressed"]]


def test_experiment_runner_compares_conditions(engine, tmp_path):
    base = config(total_tokens=24, checkpoints=(8,), probe_tokens=4)
    result = run_experiment(engine, base, parse_conditions("clean,fly,fly_reset,random", [8]), tmp_path)
    names = [row["condition"] for row in result["conditions"]]
    assert names == ["clean", "fly", "fly_reset_8", "random"]
    assert set(result["first_divergence_from_clean"]) == {"fly", "fly_reset_8", "random"}
    assert (tmp_path / "comparison.json").exists() and (tmp_path / "reservoir_rms.png").exists()
    advances = {row["condition"]: row["reservoir_advances_per_token"] for row in result["conditions"]}
    assert advances == {"clean": 0.0, "fly": 2.0, "fly_reset_8": 2.0, "random": 2.0}


def test_repetition_metrics_detect_exact_loops():
    looping = [1, 2, 3] * 100
    metrics = repetition_metrics(looping, chunk=100, max_period=32)
    assert metrics["chunks"][0]["best_period"] == 3 and metrics["chunks"][0]["self_repeat_rate"] == 1.0
    assert metrics["exact_loop_onset_chunk"] == 0
    varied = list(range(300))
    assert repetition_metrics(varied, chunk=100, max_period=32)["exact_loop_onset_chunk"] is None


def test_qwen_seed_is_raw_text_without_special_tokens(qwen_engine):
    tokenizer = qwen_engine.llm.tokenizer
    ids = seed_token_ids(tokenizer, "Однажды в студёную зимнюю пору")
    assert tokenizer.decode(ids) == "Однажды в студёную зимнюю пору"
    special = set(tokenizer.all_special_ids)
    assert not special & set(ids)
    run = ContinuousRun(qwen_engine, config(seed_text="Однажды в студёную зимнюю пору", window=6, total_tokens=8))
    assert run.window_ids == ids[-6:]
    run.run()
    assert run.max_window_seen == 6


def test_chat_prefix_is_pinned_and_only_the_body_slides(qwen_engine, monkeypatch):
    calls = spy_inputs(qwen_engine, monkeypatch)
    run = ContinuousRun(qwen_engine, config(seed_text="Однажды в студёную зимнюю пору", window=6, total_tokens=12,
                                            prompt_policy="chat_prefix"))
    run.run()
    prefix = run.prefix_ids
    tokenizer = qwen_engine.llm.tokenizer
    assert tokenizer.decode(prefix).startswith("<|im_start|>user\nПродолжи текст.<|im_end|>")
    assert tokenizer.decode(prefix).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    history = run.seed_ids + run.generated
    for step, call in enumerate(calls):
        assert call["ids"][:len(prefix)] == prefix
        assert call["ids"][len(prefix):] == history[:len(run.seed_ids) + step][-6:]
    assert ContinuousRun(qwen_engine, config()).prefix_ids == []


def test_sampling_config_is_validated():
    for bad in ({"decoding": "beam"}, {"temperature": 0}, {"top_p": 0}, {"top_p": 1.5}):
        with pytest.raises(ValueError):
            config(**bad).validate()


def test_sampling_is_reproducible_and_independent_of_condition_order(engine):
    def tokens(condition, **values):
        run = ContinuousRun(engine, config(condition=condition, decoding="sample", total_tokens=30, **values))
        run.run()
        return run.generated
    clean_first = tokens("clean")
    fly_second = tokens("fly")
    fly_first = tokens("fly")
    clean_second = tokens("clean")
    assert clean_first == clean_second and fly_first == fly_second
    assert tokens("clean", sampling_seed=7) != clean_first, "A different seed must change the sample path"


def test_common_random_numbers_make_zero_coupling_identical_to_clean(engine):
    clean = ContinuousRun(engine, config(condition="clean", decoding="sample", total_tokens=30))
    clean.run()
    silent = ContinuousRun(engine, config(condition="fly", decoding="sample", total_tokens=30, strength=0.0, feedback=0.0))
    silent.run()
    assert silent.generated == clean.generated
    greedy = ContinuousRun(engine, config(condition="clean", total_tokens=30))
    greedy.run()
    narrow = ContinuousRun(engine, config(condition="clean", decoding="sample", top_p=1e-9, total_tokens=30))
    narrow.run()
    assert narrow.generated == greedy.generated, "top_p -> 0 must reduce to greedy"


def test_sampled_probes_check_logits_not_argmax(engine):
    summary = ContinuousRun(engine, config(decoding="sample", total_tokens=24, checkpoints=(8,), probe_tokens=4)).run()
    probe = summary["probes"][0]
    assert probe["retained_first_logits_equal_main"] is True and probe["retained_first_logits_max_abs_diff"] == 0.0


def test_sampled_eos_suppression_uses_the_same_draw(engine):
    run = ContinuousRun(engine, config(condition="clean", decoding="sample", total_tokens=20, temperature=5.0))
    run.run()
    engine.llm.tokenizer.eos_token_id = run.generated[3]
    forced = ContinuousRun(engine, config(condition="clean", decoding="sample", total_tokens=20, temperature=5.0,
                                          eos_policy="suppress"))
    summary = forced.run()
    assert summary["generated_tokens"] == 20 and summary["eos_suppressed"]
    assert run.generated[3] not in forced.generated
    assert forced.generated[:3] == run.generated[:3]


def test_repetition_metrics_detect_paragraph_length_cycles():
    paragraph = list(range(91))
    metrics = repetition_metrics(paragraph * 12, chunk=512)
    assert metrics["chunks"][1]["best_period"] == 91 and metrics["chunks"][1]["self_repeat_rate"] == 1.0


def test_archived_gzip_telemetry_loads_identically(engine, tmp_path):
    import gzip
    ContinuousRun(engine, config(total_tokens=12), tmp_path / "g").run()
    plain = load_run(tmp_path / "g")
    path = tmp_path / "g" / "telemetry.jsonl"
    (tmp_path / "g" / "telemetry.jsonl.gz").write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    assert load_run(tmp_path / "g")["telemetry"] == plain["telemetry"]
