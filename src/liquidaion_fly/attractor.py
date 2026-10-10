"""Attractor diagnostic: does recurrent Fly coupling drive Qwen into a repetitive regime?

Runs real greedy inference through the observatory engine under matched controls and
writes structured JSON/CSV results plus plots. Nothing here applies repetition
penalties or other decoding changes; decoding is the same greedy argmax as the GUI.

Protocols
  conversation  - one multi-turn chat of unrelated prompts per condition, identical
                  zero initial reservoir state, persistent history (repetition, accuracy,
                  activation/reservoir similarity across turns).
  sensitivity   - first-token distributions for different prompts after each
                  condition's own conversation (does the latest input still matter?)
                  and on a fresh context (pure intervention effect versus Clean).
  repeated      - one identical prompt asked repeatedly with fresh history while the
                  reservoir persists (state dependence, convergence of end states).
  turn_map      - the turn-to-turn map state -> answer -> state under a repeated prompt,
                  swept over coupling strength, with detection of fixed points and cycles.
  recovery      - probe prompts after the strongest coupled conversation with steering
                  off, feedback off, reservoir reset, fresh history, or Clean.
  determinism   - an identical rerun to bound numerical noise.

Usage:
  python -m liquidaion_fly.attractor --output results/attractor [--quick]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from .debuglog import digest
from .observatory_engine import ChatSession, ObservatoryConfig, ObservatoryEngine
from .recording import SessionRecording
from .reservoir import Reservoir

LOG = logging.getLogger("liquidaion.attractor")
ROOT = Path(__file__).resolve().parents[2]

# Unrelated prompts in the order of the reported failing chat; None means no fixed answer.
CONVERSATION = [
    ("Привет! Расскажи, что такое коннектом плодовой мушки.", None),
    ("Как такой коннектом можно связать с языковой моделью?", None),
    ("Ты можешь работать в разных режимах — как в чате и как в цикле?", None),
    ("Напиши только слово КВАРЦ.", ["кварц"]),
    ("Сколько будет 7 умножить на 8?", ["56"]),
    ("Назови столицу Франции.", ["париж", "paris"]),
    ("Тебе не хватает токенов на ответ.", None),
    ("Переведи на английский: «доброе утро».", ["good morning"]),
    ("Сколько ног у насекомого?", ["6", "шест"]),
    ("Напиши только число: 2 плюс 2.", ["4", "четыре"]),
]
PROBES = [item for item in CONVERSATION if item[1] is not None]
REPEATED_PROMPT = "Опиши одним абзацем, что ты видишь за окном."
TURN_MAP_PROMPTS = [REPEATED_PROMPT, "Придумай короткое название для нового цветка."]


@dataclass(frozen=True)
class Condition:
    name: str
    coupled: bool = True
    strength: float = 0.0
    feedback: float = 0.0
    mode: str = "fly"
    reset_between_turns: bool = False


def default_conditions(strengths=(0.05, 0.10, 0.25, 0.50), feedback=0.25,
                       controls=(0.25, 0.50)) -> list[Condition]:
    conditions = [Condition("clean", coupled=False), Condition("fly_off")]
    for s in strengths:
        conditions.append(Condition(f"steer_{s:.2f}", strength=s))
    for s in strengths:
        conditions.append(Condition(f"steer_fb_{s:.2f}", strength=s, feedback=feedback))
    for s in controls:
        conditions.append(Condition(f"steer_fb_reset_{s:.2f}", strength=s, feedback=feedback,
                                    reset_between_turns=True))
        conditions.append(Condition(f"random_fb_{s:.2f}", strength=s, feedback=feedback, mode="random"))
    return conditions


# ---------------------------------------------------------------- text metrics

def words(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def opening(text: str, n: int = 5) -> str:
    return " ".join(words(text)[:n])


def char_ngrams(text: str, n: int = 3) -> Counter:
    text = " ".join(words(text))
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def cosine_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    return float(dot / (np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values()))))


def text_similarity(a: str, b: str) -> float:
    return cosine_counts(char_ngrams(a), char_ngrams(b))


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    """Fraction of word n-grams that already occurred earlier in the same response."""
    tokens = words(text)
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def distinct_n(text: str, n: int = 2) -> float:
    tokens = words(text)
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def answered(text: str, expected: list[str] | None) -> bool | None:
    if expected is None:
        return None
    lowered = text.lower()
    return any(item in lowered for item in expected)


# ------------------------------------------------------------ vector metrics

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def mean_offdiagonal(vectors: list[np.ndarray]) -> float | None:
    pairs = [cosine(a, b) for a, b in itertools.combinations(vectors, 2)]
    return float(np.mean(pairs)) if pairs else None


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = np.exp(log_softmax(a)), np.exp(log_softmax(b))
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
    la, lb = log_softmax(a), log_softmax(b)
    return float(np.sum(np.exp(la) * (la - lb)))


# ------------------------------------------------------------------ runner

class Diagnostic:
    def __init__(self, engine: ObservatoryEngine, limit: int, output: Path):
        self.engine = engine
        self.limit = limit
        self.output = output
        self.token_rows: list[dict[str, Any]] = []
        self.random_graph = engine.control_connectome()

    def reservoir(self, condition: Condition, state: np.ndarray | None = None) -> Reservoir:
        graph = self.random_graph if condition.mode == "random" else self.engine.connectome
        # This diagnostic has always used the baseline leak 0.2 (Dynamics defaults), not config.leak_rate.
        reservoir = self.engine.create_reservoir(graph, leak=0.2)
        if state is not None:
            reservoir.state = state.copy()
        return reservoir

    def coupling(self, condition: Condition, reservoir: Reservoir):
        if not condition.coupled:
            return None
        return self.engine._coupling(reservoir, condition.mode, condition.strength, condition.feedback)

    def generate(self, label: str, prompt: str, session: ChatSession, coupling, limit: int | None = None,
                 logits: bool = False) -> dict:
        # Recording snapshots are irrelevant here; keep memory flat across hundreds of runs.
        self.engine.recording = SessionRecording(self.engine.metadata)
        return self.engine._generate(label, prompt, limit or self.limit, Event(), None, None,
                                     session=session, coupling=coupling, scientific=logits)

    def conversation(self, condition: Condition) -> dict:
        reservoir = self.reservoir(condition)
        coupling = self.coupling(condition, reservoir)
        session = ChatSession([])
        turns = []
        for index, (prompt, expected) in enumerate(CONVERSATION):
            if condition.reset_between_turns:
                reservoir.reset()
            before = reservoir.state.copy()
            result = self.generate(condition.name, prompt, session, coupling)
            trace = result["trace"]
            features = np.array([row["features"] for row in trace]) if trace else np.empty((0, 0))
            for step, row in enumerate(trace):
                self.token_rows.append({"condition": condition.name, "turn": index, "step": step,
                                        "state_rms": row.get("live_reservoir_rms"),
                                        "delta_relative_rms": row.get("delta_relative_rms"),
                                        "steering_vector_norm": row.get("steering_vector_norm"),
                                        "feedback_input_norm": row.get("feedback_input_norm"),
                                        "capture_rms": row.get("capture_rms")})
            turns.append({
                "turn": index, "prompt": prompt, "text": result["text"], "token_ids": result["token_ids"],
                "generated_tokens": len(result["token_ids"]), "stop_reason": result["stop_reason"],
                "prompt_tokens": result["prompt_tokens"], "history_dropped_messages": result["history_dropped_messages"],
                "opening": opening(result["text"]), "repeated_4gram_fraction": repeated_ngram_fraction(result["text"]),
                "distinct_2": distinct_n(result["text"]), "answered": answered(result["text"], expected),
                "reservoir_rms_before": float(np.sqrt(np.mean(before ** 2))),
                "reservoir_rms_after": float(np.sqrt(np.mean(reservoir.state ** 2))),
                "mean_delta_relative_rms": float(np.mean([r["delta_relative_rms"] for r in trace])) if trace else 0.0,
                "mean_steering_vector_norm": float(np.mean([r["steering_vector_norm"] for r in trace])) if trace else 0.0,
                "mean_feedback_input_norm": float(np.mean([r.get("feedback_input_norm", 0.0) for r in trace])) if trace else 0.0,
                "_features_mean": features.mean(axis=0) if features.size else None,
                "_state_end": reservoir.state.copy() if condition.coupled else None,
            })
            LOG.info("%s turn %d [%s, %d tok]: %s", condition.name, index, result["stop_reason"],
                     len(result["token_ids"]), result["text"][:90].replace("\n", " "))
        return {"condition": condition, "turns": turns, "history": session.messages,
                "final_state": reservoir.state.copy()}

    def summarize_conversation(self, run: dict, clean: dict | None) -> dict:
        turns = run["turns"]
        texts = [t["text"] for t in turns]
        pairs = [text_similarity(a, b) for a, b in itertools.combinations(texts, 2)]
        openings = [t["opening"] for t in turns]
        accuracy = [t["answered"] for t in turns if t["answered"] is not None]
        features = [t["_features_mean"] for t in turns if t["_features_mean"] is not None]
        states = [t["_state_end"] for t in turns if t["_state_end"] is not None]
        best_earlier = [max((text_similarity(texts[i], texts[j]) for j in range(i)), default=0.0)
                        for i in range(len(texts))]
        summary = {
            "condition": run["condition"].name, **{k: v for k, v in vars(run["condition"]).items() if k != "name"},
            "turns": len(turns),
            "mean_cross_turn_text_similarity": float(np.mean(pairs)),
            "max_similarity_to_an_earlier_reply": float(max(best_earlier[1:])),
            "turns_copying_an_earlier_reply": int(sum(s > 0.8 for s in best_earlier[1:])),
            "repeated_openings": len(openings) - len(set(openings)),
            "mean_repeated_4gram_fraction": float(np.mean([t["repeated_4gram_fraction"] for t in turns])),
            "probe_accuracy": float(np.mean(accuracy)) if accuracy else None,
            "truncated_turns": sum(t["stop_reason"] == "max_new_tokens" for t in turns),
            "mean_generated_tokens": float(np.mean([t["generated_tokens"] for t in turns])),
            "mean_activation_similarity_across_turns": mean_offdiagonal(features),
            "mean_reservoir_end_state_similarity": mean_offdiagonal(states),
            "final_reservoir_rms": float(np.sqrt(np.mean(run["final_state"] ** 2))),
            "mean_delta_relative_rms": float(np.mean([t["mean_delta_relative_rms"] for t in turns])),
            "mean_feedback_input_norm": float(np.mean([t["mean_feedback_input_norm"] for t in turns])),
        }
        if clean is not None:
            summary["mean_similarity_to_clean_same_turn"] = float(np.mean(
                [text_similarity(a["text"], b["text"]) for a, b in zip(turns, clean["turns"])]))
            summary["token_identical_turns_vs_clean"] = int(sum(
                a["token_ids"] == b["token_ids"] for a, b in zip(turns, clean["turns"])))
        return summary

    def first_token_logits(self, condition: Condition, prompt: str, history: list[dict],
                           state: np.ndarray | None) -> np.ndarray:
        reservoir = self.reservoir(condition, state)
        result = self.generate(condition.name + "_probe", prompt, ChatSession(list(history)),
                               self.coupling(condition, reservoir), limit=1, logits=True)
        return result["logits"][0]

    def sensitivity(self, conditions: list[Condition], runs: dict[str, dict]) -> list[dict]:
        rows = []
        prompts = [prompt for prompt, _ in PROBES]
        fresh_clean = {p: self.first_token_logits(Condition("clean", coupled=False), p, [], None) for p in prompts}
        for condition in conditions:
            run = runs[condition.name]
            conditioned = {p: self.first_token_logits(condition, p, run["history"], run["final_state"])
                           for p in prompts}
            fresh = {p: self.first_token_logits(condition, p, [], None) for p in prompts}
            rows.append({
                "condition": condition.name,
                "mean_pairwise_js_after_conversation": float(np.mean(
                    [js_divergence(conditioned[a], conditioned[b]) for a, b in itertools.combinations(prompts, 2)])),
                "mean_pairwise_js_fresh_context": float(np.mean(
                    [js_divergence(fresh[a], fresh[b]) for a, b in itertools.combinations(prompts, 2)])),
                "mean_kl_clean_to_condition_fresh_context": float(np.mean(
                    [kl_divergence(fresh_clean[p], fresh[p]) for p in prompts])),
                "argmax_differs_from_clean_fresh_context": int(sum(
                    int(fresh_clean[p].argmax()) != int(fresh[p].argmax()) for p in prompts)),
            })
            LOG.info("sensitivity %s: %s", condition.name, rows[-1])
        return rows

    def repeated(self, conditions: list[Condition], repeats: int) -> list[dict]:
        rows = []
        for condition in conditions:
            reservoir = self.reservoir(condition)
            coupling = self.coupling(condition, reservoir)
            texts, states = [], []
            for index in range(repeats):
                result = self.generate(condition.name + "_repeat", REPEATED_PROMPT, ChatSession([]), coupling)
                texts.append(result["text"])
                states.append(reservoir.state.copy())
                rows.append({"condition": condition.name, "repeat": index, "text": result["text"],
                             "generated_tokens": len(result["token_ids"]), "stop_reason": result["stop_reason"],
                             "token_ids_sha": hash(tuple(result["token_ids"])),
                             "similarity_to_first": text_similarity(texts[0], result["text"]),
                             "reservoir_rms": float(np.sqrt(np.mean(reservoir.state ** 2))),
                             "end_state_step_distance": float(np.linalg.norm(states[-1] - states[-2]))
                             if len(states) > 1 else None})
            LOG.info("repeated %s: %d distinct outputs of %d", condition.name, len(set(texts)), repeats)
        return rows

    def turn_map(self, strengths: list[float], feedbacks: list[float], repeats: int) -> list[dict]:
        """Iterate one prompt with fresh history while the reservoir persists; classify the orbit."""
        rows = []
        for prompt, feedback, strength in itertools.product(TURN_MAP_PROMPTS, feedbacks, strengths):
            condition = Condition(f"map_s{strength:.2f}_fb{feedback:.2f}", strength=strength, feedback=feedback)
            reservoir = self.reservoir(condition)
            coupling = self.coupling(condition, reservoir)
            outputs, states, texts, stops = [], [], {}, []
            for _ in range(repeats):
                result = self.generate(condition.name, prompt, ChatSession([]), coupling)
                outputs.append(tuple(result["token_ids"]))
                states.append(reservoir.state.copy())
                stops.append(result["stop_reason"])
                texts.setdefault(digest(list(outputs[-1]))[:12], result["text"][:400])
            period = detect_period(outputs, tail=max(4, repeats - 2))
            rows.append({"prompt": prompt, "strength": strength, "feedback": feedback, "repeats": repeats,
                         "distinct_outputs": len(set(outputs)), "distinct_outputs_in_tail": len(set(outputs[repeats // 2:])),
                         "period": period, "regime": {None: "no period detected", 1: "fixed point"}.get(
                             period, f"cycle of period {period}"),
                         "transient_turns": transient(outputs, period),
                         "last_state_step_distance": float(np.linalg.norm(states[-1] - states[-2])),
                         "final_state_rms": float(np.sqrt(np.mean(states[-1] ** 2))),
                         "output_lengths": [len(o) for o in outputs],
                         "output_hashes": [digest(list(o))[:12] for o in outputs],
                         "truncated_outputs": stops.count("max_new_tokens"), "distinct_texts": texts})
            LOG.info("turn map s=%.2f fb=%.2f %s: %s", strength, feedback, prompt[:20], rows[-1]["regime"])
        return rows

    def recovery(self, base: Condition, run: dict, clean_run: dict) -> list[dict]:
        variants = [
            ("as_is", base, run["history"], run["final_state"]),
            ("steering_off", Condition("steering_off", strength=0.0, feedback=base.feedback, mode=base.mode),
             run["history"], run["final_state"]),
            ("feedback_off", Condition("feedback_off", strength=base.strength, feedback=0.0, mode=base.mode),
             run["history"], run["final_state"]),
            ("reservoir_reset", base, run["history"], None),
            ("fresh_conversation", base, [], run["final_state"]),
            ("fresh_and_reset", base, [], None),
            ("clean_same_history", Condition("clean", coupled=False), run["history"], None),
            ("clean_own_history", Condition("clean", coupled=False), clean_run["history"], None),
        ]
        earlier = [t["text"] for t in run["turns"]]
        rows = []
        for label, condition, history, state in variants:
            for prompt, expected in PROBES:
                reservoir = self.reservoir(condition, state)
                result = self.generate(f"recovery_{label}", prompt, ChatSession(list(history)),
                                       self.coupling(condition, reservoir))
                rows.append({"variant": label, "prompt": prompt, "text": result["text"],
                             "stop_reason": result["stop_reason"], "answered": answered(result["text"], expected),
                             "max_similarity_to_conversation_reply": max(
                                 (text_similarity(result["text"], text) for text in earlier), default=0.0)})
            subset = [row for row in rows if row["variant"] == label]
            LOG.info("recovery %s: accuracy %.2f", label, np.mean([r["answered"] for r in subset]))
        return rows

    def determinism(self, condition: Condition, run: dict, turns: int = 3) -> dict:
        reservoir = self.reservoir(condition)
        coupling = self.coupling(condition, reservoir)
        session = ChatSession([])
        identical = []
        for index, (prompt, _) in enumerate(CONVERSATION[:turns]):
            result = self.generate(condition.name + "_rerun", prompt, session, coupling)
            identical.append(result["token_ids"] == run["turns"][index]["token_ids"])
        return {"condition": condition.name, "turns_checked": turns, "token_identical": identical}


def detect_period(outputs: list[tuple], tail: int) -> int | None:
    """Smallest p such that the last `tail` outputs repeat with period p (p <= tail // 2)."""
    last = outputs[-tail:]
    for p in range(1, len(last) // 2 + 1):
        if all(last[i] == last[i - p] for i in range(p, len(last))):
            return p
    return None


def transient(outputs: list[tuple], period: int | None) -> int | None:
    if period is None:
        return None
    start = len(outputs) - 1
    while start - period >= 0 and outputs[start - period] == outputs[start]:
        start -= 1
    return max(0, start - period + 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(output: Path, summaries: list[dict], sensitivity: list[dict], token_rows: list[dict],
         turn_map: list[dict] | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = ["#2a78d6", "#eb6834", "#1baf7a"]
    plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.titlecolor": ink, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = [s["condition"] for s in summaries]
    files = []

    def bars(ax, values, title, xlabel):
        y = np.arange(len(names))
        ax.barh(y, [0 if v is None else v for v in values], color=series[0], height=0.6)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    bars(axes[0], [s["mean_cross_turn_text_similarity"] for s in summaries],
         "Similarity between replies\nto unrelated prompts", "mean char-3-gram cosine")
    bars(axes[1], [s["probe_accuracy"] for s in summaries], "Probe questions answered", "fraction correct")
    bars(axes[2], [s["mean_activation_similarity_across_turns"] for s in summaries],
         "Layer-6 activation similarity\nacross turns", "mean cosine (turn-mean features)")
    fig.tight_layout()
    fig.savefig(output / "conversation_metrics.png", dpi=130)
    plt.close(fig)
    files.append("conversation_metrics.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    by_name = {row["condition"]: row for row in sensitivity}
    bars(ax, [by_name.get(n, {}).get("mean_pairwise_js_after_conversation") for n in names],
         "Input sensitivity after each conversation\n(first-token JS divergence between different prompts)",
         "mean pairwise JS divergence (nats)")
    fig.tight_layout()
    fig.savefig(output / "input_sensitivity.png", dpi=130)
    plt.close(fig)
    files.append("input_sensitivity.png")

    selected = [n for n in ("fly_off", "steer_fb_0.25", "steer_fb_0.50") if n in names]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for color, name in zip(series, selected):
        rows = [r for r in token_rows if r["condition"] == name]
        axes[0].plot([r["state_rms"] for r in rows], color=color, linewidth=1.2, label=name)
        axes[1].plot([r["delta_relative_rms"] for r in rows], color=color, linewidth=1.2, label=name)
    axes[0].set_title("Reservoir state RMS over all generated tokens of the conversation", loc="left")
    axes[1].set_title("Relative hidden-state intervention at layer 14", loc="left")
    axes[1].set_xlabel("generated token (turns concatenated)")
    for ax in axes:
        ax.grid(color=grid, linewidth=0.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "dynamics.png", dpi=130)
    plt.close(fig)
    files.append("dynamics.png")

    if turn_map:
        prompts = list(dict.fromkeys(row["prompt"] for row in turn_map))
        fig, axes = plt.subplots(1, len(prompts), figsize=(6 * len(prompts), 4), sharey=True, squeeze=False)
        for ax, prompt in zip(axes[0], prompts):
            for color, feedback in zip(series, sorted({row["feedback"] for row in turn_map})):
                rows = [row for row in turn_map if row["prompt"] == prompt and row["feedback"] == feedback]
                ax.plot([row["strength"] for row in rows], [row["distinct_outputs_in_tail"] for row in rows],
                        color=color, linewidth=2, marker="o", markersize=8, label=f"feedback {feedback:.2f}")
            ax.set_title(f"Turn map orbit: «{prompt[:34]}…»", loc="left")
            ax.set_xlabel("steering strength")
            ax.set_ylabel("distinct answers in the last half (1 = fixed point)")
            ax.grid(color=grid, linewidth=0.6)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / "turn_map.png", dpi=130)
        plt.close(fig)
        files.append("turn_map.png")
    return files


def serial(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Condition):
        return vars(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run(model: str, graph: Path, output: Path, device: str, limit: int, context: int,
        quick: bool = False, repeats: int = 5, revision: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    config = ObservatoryConfig(max_context_tokens=context, max_new_tokens=limit, revision=revision)
    started = time.perf_counter()
    engine = ObservatoryEngine(model, graph, device=device, config=config)
    conditions = default_conditions(strengths=(0.25, 0.50), controls=(0.50,)) if quick else default_conditions()
    diagnostic = Diagnostic(engine, limit, output)
    runs: dict[str, dict] = {}
    for condition in conditions:
        runs[condition.name] = diagnostic.conversation(condition)
    clean = runs["clean"]
    summaries = [diagnostic.summarize_conversation(runs[c.name], clean if c.name != "clean" else None)
                 for c in conditions]
    sensitivity = diagnostic.sensitivity(conditions, runs)
    repeated_conditions = [c for c in conditions if c.name in {"clean", "fly_off", "steer_fb_0.25", "steer_fb_0.50"}]
    repeated = diagnostic.repeated(repeated_conditions, 3 if quick else repeats)
    sweep = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    turn_map = diagnostic.turn_map(sweep[::2] if quick else sweep, [0.0, 0.25], 6 if quick else 8)
    strongest = next(c for c in reversed(conditions) if c.name.startswith("steer_fb_") and "reset" not in c.name)
    recovery = diagnostic.recovery(strongest, runs[strongest.name], clean)
    determinism = [diagnostic.determinism(c, runs[c.name]) for c in conditions
                   if c.name in {"clean", strongest.name}]
    fly_off_matches = all(a["token_ids"] == b["token_ids"]
                          for a, b in zip(runs["fly_off"]["turns"], clean["turns"]))
    conversations = {name: [{k: v for k, v in turn.items() if not k.startswith("_")} for turn in run["turns"]]
                     for name, run in runs.items()}
    result = {
        "metadata": {**engine.metadata, "max_new_tokens": limit, "max_context_tokens": context,
                     "decoding": "greedy argmax, no repetition penalty", "seed": engine.config.seed,
                     "capture_layer": engine.config.capture_layer, "steer_layer": engine.config.steer_layer,
                     "quick": quick, "seconds": None},
        "implementation_checks": {"fly_off_token_identical_to_clean": fly_off_matches,
                                  "determinism": determinism},
        "conversation_summary": summaries, "sensitivity": sensitivity,
        "repeated_prompt": repeated, "turn_map": turn_map, "recovery": recovery, "conversations": conversations,
    }
    write_csv(output / "conversation_summary.csv", summaries)
    write_csv(output / "conversation_turns.csv", [{"condition": name, **{k: v for k, v in t.items() if k != "token_ids"}}
                                                  for name, turns in conversations.items() for t in turns])
    write_csv(output / "sensitivity.csv", sensitivity)
    write_csv(output / "repeated_prompt.csv", repeated)
    write_csv(output / "recovery.csv", recovery)
    write_csv(output / "turn_map.csv", turn_map)
    write_csv(output / "token_dynamics.csv", diagnostic.token_rows)
    result["figures"] = plot(output, summaries, sensitivity, diagnostic.token_rows, turn_map)
    result["metadata"]["seconds"] = time.perf_counter() - started
    (output / "attractor_diagnostic.json").write_text(
        json.dumps(result, default=serial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Attractor diagnostic for Fly-to-LLM coupling")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/observatory.json")
    parser.add_argument("--model-path")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/cx512")
    parser.add_argument("--output", type=Path, default=ROOT / "results/attractor")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Fewer conditions for a fast smoke run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = json.loads(args.config.read_text())
    device = "cpu" if args.cpu else config.get("device", "cuda")
    result = run(args.model_path or config["model"], args.graph, args.output, device, args.max_new_tokens,
                 args.context, quick=args.quick, repeats=args.repeats, revision=config.get("revision"))
    print(json.dumps({"output": str(args.output), "seconds": result["metadata"]["seconds"],
                      "fly_off_token_identical_to_clean": result["implementation_checks"]["fly_off_token_identical_to_clean"]}))


if __name__ == "__main__":
    main()
