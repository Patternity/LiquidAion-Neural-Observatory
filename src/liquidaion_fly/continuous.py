"""Continuous generation with a sliding token window and a persistent reservoir.

Working name: "Infinite Nekrasov". The language model sees at most `window` recent
tokens; the reservoir keeps evolving across the whole run.

Design decisions (see docs/CONTINUOUS_GENERATION.md):

- Prompt policy. "raw" (default): the seed is tokenized with `add_special_tokens=False`;
  no chat template, BOS or instruction is added, before or during the run, and every
  model input is exactly the most recent `window` token ids. "chat_prefix": a fixed,
  documented prefix (one user turn holding `instruction`, then the assistant header)
  is pinned in front of the window and never evicted; only the open assistant text
  slides, so eviction cannot break template structure. Nothing is ever inserted later.
- Exact sliding window. Each step recomputes the whole window with `use_cache=False`
  and positions 0..len-1. RoPE attention depends only on relative positions, so this
  equals attending over the last `window` tokens. Truncating a KV cache instead is not
  equivalent: retained keys/values were computed while evicted tokens were visible,
  which would silently reintroduce discarded text.
- One forward pass per generated token. The capture hook reads only the last position
  and the steering hook modifies only the last position. Recomputing the window
  therefore advances the reservoir exactly once from capture and once from feedback
  per step, independent of window length. Earlier positions in the window are not
  steered on recomputation, unlike the cached chat mode where steered activations of
  earlier tokens persist in the KV cache.
- Probes at checkpoints run on cloned reservoirs and never touch the main run.
- Decoding. "greedy" (argmax) is the deterministic baseline. "sample" uses temperature and
  nucleus (top-p) sampling by inverse CDF with exactly one uniform draw per step from a
  generator seeded per condition. Every condition with the same `sampling_seed` therefore
  consumes the same random numbers (common random numbers): Clean and Fly diverge only
  where the reservoir actually moved the distribution, and results do not depend on the
  order in which conditions run. No repetition penalty is applied.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import torch

from .llm import compare_logits, stop_token_ids
from .reservoir import Reservoir, random_control

LOG = logging.getLogger("liquidaion.continuous")
ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = ("clean", "fly", "fly_reset", "random")
DEFAULT_SEED_TEXT = "Однажды в студёную зимнюю пору"
DEFAULT_INSTRUCTION = "Продолжи текст."
PROMPT_POLICIES = ("raw", "chat_prefix")
DEFAULT_CHECKPOINTS = (128, 256, 512, 1024, 2048, 4096)
MAX_WINDOW = 4096


@dataclass
class ContinuousConfig:
    seed_text: str = DEFAULT_SEED_TEXT
    window: int = 128
    total_tokens: int = 1000
    condition: str = "fly"
    strength: float = 0.25
    feedback: float = 0.25
    reset_interval: int | None = None
    eos_policy: str = "stop"
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS
    probe_tokens: int = 16
    snapshot_interval: int = 64
    prompt_policy: str = "raw"
    instruction: str = DEFAULT_INSTRUCTION
    decoding: str = "greedy"
    temperature: float = 0.8
    top_p: float = 0.9
    sampling_seed: int = 42

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"Condition must be one of {CONDITIONS}")
        if not 1 <= self.window <= MAX_WINDOW:
            raise ValueError(f"Window must lie in [1, {MAX_WINDOW}] tokens")
        if self.total_tokens < 1:
            raise ValueError("Total tokens must be positive")
        if self.eos_policy not in {"stop", "suppress"}:
            raise ValueError("EOS policy must be 'stop' or 'suppress'")
        if self.condition == "fly_reset" and not (self.reset_interval and self.reset_interval > 0):
            raise ValueError("fly_reset requires a positive reset_interval")
        if self.condition != "fly_reset" and self.reset_interval:
            raise ValueError("reset_interval applies only to fly_reset")
        if any(not np.isfinite(x) or not 0 <= x <= 1 for x in (self.strength, self.feedback)):
            raise ValueError("Strength and feedback must lie in [0, 1]")
        if self.probe_tokens < 1 or self.snapshot_interval < 1:
            raise ValueError("Probe tokens and snapshot interval must be positive")
        if not self.seed_text:
            raise ValueError("Seed text must not be empty")
        if self.prompt_policy not in PROMPT_POLICIES:
            raise ValueError(f"Prompt policy must be one of {PROMPT_POLICIES}")
        if self.decoding not in {"greedy", "sample"}:
            raise ValueError("Decoding must be 'greedy' or 'sample'")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not np.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")

    @property
    def coupled(self) -> bool:
        return self.condition != "clean"


def prefix_token_ids(tokenizer: Any, config: "ContinuousConfig") -> list[int]:
    """Pinned tokens in front of every window; empty for raw continuation."""
    if config.prompt_policy == "raw":
        return []
    text = tokenizer.apply_chat_template([{"role": "user", "content": config.instruction}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    return seed_token_ids(tokenizer, text)


def nucleus_probabilities(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Temperature-scaled probabilities restricted to the smallest set with mass >= top_p."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    ordered, order = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(ordered, dim=-1)
    keep = int(torch.searchsorted(cumulative, torch.tensor(top_p, dtype=cumulative.dtype)).item()) + 1
    nucleus = torch.zeros_like(probs)
    nucleus[order[:keep]] = ordered[:keep]
    return nucleus / nucleus.sum()


def inverse_cdf(probs: torch.Tensor, u: float) -> int:
    """Token whose cumulative probability (in token-id order) first exceeds u."""
    cumulative = torch.cumsum(probs, dim=-1)
    index = int(torch.searchsorted(cumulative, torch.tensor(u * float(cumulative[-1]), dtype=cumulative.dtype)).item())
    return min(index, len(probs) - 1)


def seed_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize the seed as raw text, without special tokens or a chat template."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


class IncrementalDecoder:
    """Emit text only once the pending token bytes form complete characters."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.pending: list[int] = []

    def push(self, token: int) -> str:
        self.pending.append(token)
        text = self.tokenizer.decode(self.pending, skip_special_tokens=False)
        if text.endswith("�") and len(self.pending) < 8:
            return ""
        self.pending = []
        return text

    def flush(self) -> str:
        text = self.tokenizer.decode(self.pending, skip_special_tokens=False) if self.pending else ""
        self.pending = []
        return text


class ContinuousRun:
    """One condition: sliding-window greedy generation coupled to a persistent reservoir."""

    def __init__(self, engine: Any, config: ContinuousConfig, output: Path | None = None):
        config.validate()
        self.engine = engine
        self.config = config
        self.llm = engine.llm
        self.output = Path(output) if output is not None else None
        graph = random_control(engine.graph, engine.config.seed) if config.condition == "random" else engine.graph
        self.graph = graph
        self.reservoir = Reservoir(graph, features=engine.config.features, seed=engine.config.seed) \
            if config.coupled else None
        self.coupling = self._coupling(self.reservoir) if config.coupled else None
        seed_ids = seed_token_ids(self.llm.tokenizer, config.seed_text)
        if not seed_ids:
            raise ValueError("Seed text produced no tokens")
        self.seed_ids = seed_ids
        self.prefix_ids = prefix_token_ids(self.llm.tokenizer, config)
        # The only token state inside the engine is this bounded window (plus a fixed prefix).
        self.window_ids = seed_ids[-config.window:]
        self.generated: list[int] = []
        self.step = 0
        self.stops = stop_token_ids(self.llm.tokenizer, self.llm.model)
        self.decoder = IncrementalDecoder(self.llm.tokenizer)
        self.forward_calls = 0
        self.reservoir_advances = 0
        self.max_window_seen = len(self.window_ids)
        self.resets: list[int] = []
        self.eos_suppressed: list[dict[str, int]] = []
        self.snapshots: dict[int, np.ndarray] = {}
        self.pending_probes: list[dict[str, Any]] = []
        self.probes: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self.seconds = 0.0
        self.tail_text = ""
        # Independent, explicitly seeded stream per condition: one uniform draw per step.
        self.rng = torch.Generator(device="cpu").manual_seed(config.sampling_seed)
        self._files: dict[str, Any] = {}

    def _coupling(self, reservoir: Reservoir):
        mode = "random" if self.config.condition == "random" else "fly"
        return self.engine._coupling(reservoir, mode, self.config.strength, self.config.feedback)

    # ------------------------------------------------------------ inference

    def _forward_logits(self, window: list[int], coupling: Any) -> torch.Tensor:
        """One forward pass over the pinned prefix plus exactly `window`; last-position logits on CPU."""
        if len(window) > self.config.window:
            raise RuntimeError("Model input exceeds the configured window")
        ids = torch.tensor([self.prefix_ids + window], device=self.llm.device)
        if coupling is not None:
            before = coupling.reservoir.time
            coupling.attach(self.llm.model)
        elif self.engine._hook_count():
            raise RuntimeError("Forward hooks are present during uncoupled inference")
        try:
            with torch.inference_mode():
                result = self.llm.model(input_ids=ids, use_cache=False)
        finally:
            if coupling is not None:
                coupling.detach()
        if coupling is not None:
            dt = coupling.reservoir.dynamics.dt
            self.last_advances = int(round((coupling.reservoir.time - before) / dt))
        if getattr(result, "past_key_values", None) is not None:
            raise RuntimeError("A KV cache was created in sliding-window mode")
        return result.logits[0, -1].float().cpu()

    def step_once(self) -> dict[str, Any]:
        """Generate one token; returns the telemetry row for this step."""
        config = self.config
        reset = False
        if (config.condition == "fly_reset" and self.step > 0 and self.step % config.reset_interval == 0):
            self.reservoir.reset()
            self.resets.append(self.step)
            reset = True
        if self.step in config.checkpoints and self.reservoir is not None:
            self._open_probe()
        if self.reservoir is not None:
            self.snapshot_if_due()
        self.last_advances = 0
        logits = self._forward_logits(self.window_ids, self.coupling)
        self.forward_calls += 1
        self.reservoir_advances += self.last_advances
        for probe in self.pending_probes:
            if probe["step"] == self.step and "main_logits" not in probe:
                probe["main_logits"] = logits.numpy().copy()
        token, suppressed = self.select(logits)
        if suppressed is not None:
            self.eos_suppressed.append({"step": self.step, "eos_token_id": suppressed})
        elif token in self.stops:
            self.stop_reason = "eos"
        row: dict[str, Any] = {"step": self.step, "token_id": token, "window_len": len(self.window_ids),
                               "window_start": len(self.seed_ids) + self.step - len(self.window_ids),
                               "reset": reset, "eos_suppressed": suppressed, "reservoir_advances": self.last_advances}
        if self.coupling is not None and self.coupling.records:
            record = self.coupling.records[-1]
            state = self.reservoir.state
            row.update({"state_rms": float(np.sqrt(np.mean(state ** 2))), "state_norm": float(np.linalg.norm(state)),
                        "delta_relative_rms": record["delta_relative_rms"],
                        "steering_vector_norm": record["steering_vector_norm"],
                        "feedback_input_norm": record.get("feedback_input_norm", 0.0),
                        "input_norm": record["input_norm"], "capture_rms": record["capture_rms"]})
            # Per-step traces would grow without bound over thousands of tokens.
            self.coupling.clear_trace()
        if self.stop_reason == "eos":
            row["fragment"] = ""
            return row
        self.generated.append(token)
        self.window_ids = (self.window_ids + [token])[-config.window:]
        self.max_window_seen = max(self.max_window_seen, len(self.window_ids))
        row["fragment"] = self.decoder.push(token)
        self.step += 1
        self._run_due_probes()
        return row

    def select(self, logits: torch.Tensor) -> tuple[int, int | None]:
        """Pick the next token; returns (token, suppressed EOS id or None)."""
        config = self.config
        stops = sorted(self.stops)
        if config.decoding == "greedy":
            token = int(logits.argmax())
            if token in self.stops and config.eos_policy == "suppress":
                masked = logits.clone()
                masked[stops] = float("-inf")
                return int(masked.argmax()), token
            return token, None
        u = float(torch.rand((), generator=self.rng))
        probs = nucleus_probabilities(logits, config.temperature, config.top_p)
        token = inverse_cdf(probs, u)
        if token in self.stops and config.eos_policy == "suppress":
            # Same draw on the distribution without EOS, so every step still uses one number.
            probs = probs.clone()
            probs[stops] = 0.0
            if float(probs.sum()) <= 0:
                probs = nucleus_probabilities(logits.index_fill(0, torch.tensor(stops), float("-inf")),
                                              config.temperature, config.top_p)
            return inverse_cdf(probs, u), token
        return token, None

    def snapshot_if_due(self) -> None:
        if self.step % self.config.snapshot_interval == 0 or self.step in self.config.checkpoints:
            self.snapshots[self.step] = self.reservoir.state.copy()

    # --------------------------------------------------------------- probes

    def _open_probe(self) -> None:
        self.pending_probes.append({"step": self.step, "context": list(self.window_ids),
                                    "state": self.reservoir.state.copy(), "time": self.reservoir.time})

    def _run_due_probes(self, force: bool = False) -> None:
        due = [p for p in self.pending_probes
               if force or self.step >= p["step"] + self.config.probe_tokens]
        for probe in due:
            self.pending_probes.remove(probe)
            forced = self.generated[probe["step"]:probe["step"] + self.config.probe_tokens]
            if forced:
                self.probes.append(self.probe(probe, forced))

    def probe(self, checkpoint: dict[str, Any], forced: list[int]) -> dict[str, Any]:
        """Score identical tokens under retained, zero and no reservoir state on cloned reservoirs."""
        state_before = self.reservoir.state.copy()
        time_before = self.reservoir.time

        def run(state: np.ndarray | None) -> np.ndarray:
            if state is None:
                coupling = None
            else:
                reservoir = Reservoir(self.graph, features=self.engine.config.features, seed=self.engine.config.seed)
                reservoir.state = state.copy()
                reservoir.time = checkpoint["time"]
                coupling = self._coupling(reservoir)
            window = list(checkpoint["context"])
            rows = []
            for token in forced:
                rows.append(self._forward_logits(window, coupling).numpy())
                if coupling is not None:
                    coupling.clear_trace()
                window = (window + [token])[-self.config.window:]
            return np.stack(rows)

        retained = run(checkpoint["state"])
        reset = run(np.zeros_like(checkpoint["state"]))
        clean = run(None)
        if not np.array_equal(self.reservoir.state, state_before) or self.reservoir.time != time_before:
            raise RuntimeError("Checkpoint probe mutated the main reservoir")
        return {"step": checkpoint["step"], "probe_tokens": len(forced),
                "context_tokens": len(checkpoint["context"]),
                "seed_tokens_in_context": max(0, len(self.seed_ids) - checkpoint["step"]),
                "retained_state_rms": float(np.sqrt(np.mean(checkpoint["state"] ** 2))),
                "retained_vs_reset": compare_logits(reset, retained),
                "clean_vs_retained": compare_logits(clean, retained),
                "clean_vs_reset": compare_logits(clean, reset),
                "retained_first_logits_equal_main": bool(np.array_equal(retained[0], checkpoint["main_logits"]))
                if "main_logits" in checkpoint else None,
                "retained_first_logits_max_abs_diff": float(np.abs(retained[0] - checkpoint["main_logits"]).max())
                if "main_logits" in checkpoint else None}

    # ------------------------------------------------------------- run loop

    def metadata(self) -> dict[str, Any]:
        return {"experiment": "continuous_sliding_window", "config": asdict(self.config),
                "engine": {key: self.engine.metadata[key] for key in
                           ("model", "model_revision", "connectome_hash", "node_hash")},
                "reservoir_fingerprint": self.reservoir.fingerprint if self.reservoir else None,
                "capture_layer": self.engine.config.capture_layer, "steer_layer": self.engine.config.steer_layer,
                "features": self.engine.config.features, "seed": self.engine.config.seed,
                "decoding": {"method": self.config.decoding, "temperature": self.config.temperature,
                             "top_p": self.config.top_p, "sampling_seed": self.config.sampling_seed,
                             "rng": "torch.Generator(cpu) per condition, one uniform per step, inverse CDF",
                             "repetition_penalty": None} if self.config.decoding == "sample"
                else {"method": "greedy", "repetition_penalty": None},
                "tokenization": "raw continuation: tokenizer(seed, add_special_tokens=False); no chat template"
                if self.config.prompt_policy == "raw" else
                "chat_prefix: pinned chat-template user turn + assistant header, then the sliding body",
                "prefix_token_ids": self.prefix_ids,
                "inference": "full recomputation of the bounded window each step, use_cache=False, positions 0..len-1",
                "seed_token_ids": self.seed_ids, "stop_token_ids": sorted(self.stops),
                "device": str(self.llm.device), "started_utc": datetime.now(timezone.utc).isoformat()}

    def _open_files(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "metadata.json").write_text(json.dumps(self.metadata(), ensure_ascii=False, indent=2) + "\n")
        self._files = {"telemetry": (self.output / "telemetry.jsonl").open("w", encoding="utf-8"),
                       "text": (self.output / "text.txt").open("w", encoding="utf-8")}
        self._files["text"].write(self.config.seed_text)

    def _write(self, row: dict[str, Any]) -> None:
        self._files["telemetry"].write(json.dumps(row, ensure_ascii=False) + "\n")
        self._files["text"].write(row.get("fragment", ""))
        if row["step"] % 64 == 0:
            for stream in self._files.values():
                stream.flush()

    def run(self, cancel: Event | None = None, on_progress: Callable[[dict, "ContinuousRun"], None] | None = None) -> dict:
        cancel = cancel or Event()
        device = self.llm.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if self.output is not None:
            self._open_files()
        started = time.perf_counter()
        try:
            while self.step < self.config.total_tokens:
                if cancel.is_set():
                    self.stop_reason = "cancelled"
                    break
                row = self.step_once()
                row["elapsed"] = time.perf_counter() - started
                if self.output is not None:
                    self._write(row)
                if on_progress:
                    on_progress(row, self)
                if self.stop_reason == "eos":
                    break
            else:
                self.stop_reason = "total_tokens"
            if self.reservoir is not None:
                self.snapshots[self.step] = self.reservoir.state.copy()
            self._run_due_probes(force=True)
        except BaseException:
            self.stop_reason = self.stop_reason or "error"
            raise
        finally:
            self.seconds = time.perf_counter() - started
            tail = self.decoder.flush()
            self.tail_text = tail
            if self._files:
                self._files["text"].write(tail)
                for stream in self._files.values():
                    stream.close()
                self._files = {}
            if self.output is not None:
                self._save()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        device = self.llm.device
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        return {"condition": self.config.condition, "reset_interval": self.config.reset_interval,
                "stop_reason": self.stop_reason, "generated_tokens": len(self.generated),
                "seconds": self.seconds, "tokens_per_second": len(self.generated) / max(self.seconds, 1e-9),
                "window": self.config.window, "max_window_seen": self.max_window_seen,
                "window_shifts": max(0, len(self.seed_ids) + len(self.generated) - self.config.window),
                "forward_calls": self.forward_calls, "reservoir_advances": self.reservoir_advances,
                "resets": self.resets, "eos_suppressed": self.eos_suppressed,
                "peak_gpu_allocated_bytes": peak, "probes": self.probes,
                "final_state_rms": float(np.sqrt(np.mean(self.reservoir.state ** 2))) if self.reservoir is not None else None}

    def _save(self) -> None:
        steps = sorted(self.snapshots)
        np.savez_compressed(self.output / "states.npz", steps=np.array(steps, dtype=np.int64),
                            states=np.stack([self.snapshots[s] for s in steps]) if steps else np.empty((0, 0)))
        (self.output / "token_ids.json").write_text(json.dumps({"seed": self.seed_ids, "generated": self.generated}) + "\n")
        (self.output / "summary.json").write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n")


def load_run(path: str | Path) -> dict[str, Any]:
    """Read a saved condition directory back: metadata, tokens, telemetry, text and states."""
    path = Path(path)
    # Archived runs keep their telemetry gzip-compressed; live runs stream plain JSON lines.
    if (path / "telemetry.jsonl").exists():
        raw = (path / "telemetry.jsonl").read_text(encoding="utf-8")
    else:
        import gzip
        raw = gzip.decompress((path / "telemetry.jsonl.gz").read_bytes()).decode("utf-8")
    telemetry = [json.loads(line) for line in raw.splitlines() if line]
    with np.load(path / "states.npz") as data:
        states = {"steps": data["steps"].tolist(), "states": data["states"]}
    return {"metadata": json.loads((path / "metadata.json").read_text()),
            "summary": json.loads((path / "summary.json").read_text()),
            "tokens": json.loads((path / "token_ids.json").read_text()),
            "telemetry": telemetry, "text": (path / "text.txt").read_text(encoding="utf-8"), "states": states}


# ---------------------------------------------------------------- analysis

def repetition_metrics(tokens: list[int], chunk: int = 512, max_period: int = 256) -> dict[str, Any]:
    """Token-level repetition: distinct n-grams and exact periodic loops, overall and per chunk."""
    def distinct(seq: list[int], n: int) -> float:
        grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
        return len(set(grams)) / len(grams) if grams else 1.0

    def loop_fraction(seq: list[int]) -> tuple[float, int | None]:
        # Fraction of positions equal to the token p steps earlier, maximized over periods p.
        array = np.asarray(seq)
        best, period = 0.0, None
        for p in range(1, min(max_period, len(array) - 1) + 1):
            rate = float(np.mean(array[p:] == array[:-p]))
            if rate > best + 1e-12:
                best, period = rate, p
        return best, period

    chunks = []
    for start in range(0, len(tokens), chunk):
        part = tokens[start:start + chunk]
        rate, period = loop_fraction(part)
        chunks.append({"start": start, "distinct_1": distinct(part, 1), "distinct_2": distinct(part, 2),
                       "self_repeat_rate": rate, "best_period": period})
    onset = next((c["start"] for c in chunks if c["self_repeat_rate"] > 0.9), None)
    # Repetition-dominated intervals: consecutive chunks whose tokens mostly repeat a fixed period.
    intervals, current = [], None
    for c in chunks:
        dominated = c["self_repeat_rate"] > 0.5
        if dominated and current is None:
            current = {"start": c["start"], "end": c["start"] + chunk}
        elif dominated:
            current["end"] = c["start"] + chunk
        elif current is not None:
            intervals.append({**current, "recovered": True})
            current = None
    if current is not None:
        intervals.append({**current, "end": min(current["end"], len(tokens)), "recovered": False})
    return {"distinct_1": distinct(tokens, 1), "distinct_2": distinct(tokens, 2), "chunks": chunks,
            "exact_loop_onset_chunk": onset,
            "intervals": {"count": len(intervals), "tokens": sum(i["end"] - i["start"] for i in intervals),
                          "recovered": sum(i["recovered"] for i in intervals), "list": intervals}}


def state_similarity(states: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    unit = states / np.maximum(norms, 1e-12)
    return unit @ unit.T


def rms_autocorrelation_peak(rms: np.ndarray, max_lag: int = 512) -> dict[str, Any]:
    x = rms - rms.mean()
    if len(x) < 4 or not np.any(x):
        return {"lag": None, "value": None}
    full = np.correlate(x, x, mode="full")[len(x) - 1:]
    full = full / full[0]
    lags = np.arange(len(full))
    window = (lags >= 2) & (lags <= min(max_lag, len(x) - 1))
    if not window.any():
        return {"lag": None, "value": None}
    best = int(lags[window][np.argmax(full[window])])
    return {"lag": best, "value": float(full[best])}


def plot_experiment(output: Path, runs: dict[str, dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.titlecolor": ink, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    files = []
    coupled = [name for name in runs if runs[name]["telemetry"] and "state_rms" in runs[name]["telemetry"][0]]
    if coupled:
        fig, axes = plt.subplots(len(coupled), 1, figsize=(12, 2.2 * len(coupled) + 0.6), sharex=True, squeeze=False)
        for ax, color, name in zip(axes[:, 0], series, coupled):
            rows = runs[name]["telemetry"]
            ax.plot([r["step"] for r in rows], [r["state_rms"] for r in rows], color=color, linewidth=0.8)
            for step in runs[name]["summary"]["resets"]:
                ax.axvline(step, color=muted, linewidth=0.5, alpha=0.4)
            ax.set_title(f"{name}: reservoir state RMS", loc="left")
            ax.grid(color=grid, linewidth=0.5)
        axes[-1, 0].set_xlabel("generated token")
        fig.tight_layout()
        fig.savefig(output / "reservoir_rms.png", dpi=130)
        plt.close(fig)
        files.append("reservoir_rms.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        for color, name in zip(series, coupled):
            probes = runs[name]["summary"]["probes"]
            if probes:
                ax.plot([p["step"] for p in probes], [p["retained_vs_reset"]["mean_kl_baseline_to_condition"] for p in probes],
                        color=color, linewidth=2, marker="o", markersize=8, label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("checkpoint (generated tokens)")
        ax.set_ylabel("mean KL(reset ‖ retained), identical tokens")
        ax.set_title("Effect of retained reservoir state at checkpoints", loc="left")
        ax.grid(color=grid, linewidth=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / "checkpoint_kl.png", dpi=130)
        plt.close(fig)
        files.append("checkpoint_kl.png")

        name = "fly" if "fly" in runs else coupled[0]
        states = runs[name]["states"]["states"]
        if len(states) > 1:
            fig, ax = plt.subplots(figsize=(6, 5))
            steps = runs[name]["states"]["steps"]
            image = ax.imshow(state_similarity(states), cmap="Blues", vmin=0, vmax=1,
                              extent=[steps[0], steps[-1], steps[-1], steps[0]])
            fig.colorbar(image, ax=ax, label="cosine similarity")
            ax.set_title(f"{name}: similarity of reservoir snapshots", loc="left")
            ax.set_xlabel("generated token")
            ax.set_ylabel("generated token")
            fig.tight_layout()
            fig.savefig(output / "state_similarity.png", dpi=130)
            plt.close(fig)
            files.append("state_similarity.png")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    for color, (name, run) in zip(series, runs.items()):
        chunks = repetition_metrics(run["tokens"]["generated"])["chunks"]
        ax.plot([c["start"] for c in chunks], [c["distinct_2"] for c in chunks], color=color, linewidth=2,
                marker="o", markersize=5, label=name)
    ax.set_title("Token diversity per 512-token chunk (distinct bigrams / bigrams)", loc="left")
    ax.set_xlabel("generated token")
    ax.set_ylim(0, 1.02)
    ax.grid(color=grid, linewidth=0.5)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(output / "diversity.png", dpi=130)
    plt.close(fig)
    files.append("diversity.png")
    return files


def compare_runs(output: Path, names: list[str]) -> dict[str, Any]:
    runs = {name: load_run(output / name) for name in names}
    table = []
    for name, run in runs.items():
        summary = run["summary"]
        rms = np.array([r["state_rms"] for r in run["telemetry"] if "state_rms" in r])
        repetition = repetition_metrics(run["tokens"]["generated"])
        probes = summary["probes"]
        table.append({"condition": name, "reset_interval": summary["reset_interval"],
                      "generated_tokens": summary["generated_tokens"], "stop_reason": summary["stop_reason"],
                      "seconds": summary["seconds"], "tokens_per_second": summary["tokens_per_second"],
                      "peak_gpu_allocated_bytes": summary["peak_gpu_allocated_bytes"],
                      "window_shifts": summary["window_shifts"], "max_window_seen": summary["max_window_seen"],
                      "reservoir_advances_per_token": summary["reservoir_advances"] / max(summary["forward_calls"], 1),
                      "eos_suppressed": len(summary["eos_suppressed"]),
                      "state_rms_mean": float(rms.mean()) if rms.size else None,
                      "state_rms_last_1000_mean": float(rms[-1000:].mean()) if rms.size else None,
                      "rms_autocorrelation_peak": rms_autocorrelation_peak(rms) if rms.size else None,
                      "distinct_1": repetition["distinct_1"], "distinct_2": repetition["distinct_2"],
                      "exact_loop_onset_chunk": repetition["exact_loop_onset_chunk"],
                      "checkpoint_kl_retained_vs_reset": {p["step"]: p["retained_vs_reset"]["mean_kl_baseline_to_condition"]
                                                          for p in probes},
                      "checkpoint_argmax_disagreement": {p["step"]: p["retained_vs_reset"]["argmax_disagreement_fraction"]
                                                         for p in probes},
                      # Runs recorded before sampling support checked the argmax instead of the logits.
                      "probe_integrity": all(p.get("retained_first_logits_equal_main",
                                                   p.get("retained_first_argmax_matches_main")) is not False
                                             for p in probes),
                      "repetition_intervals": repetition["intervals"],
                      "excerpts": excerpts(run)})
    comparison = {"conditions": table, "figures": plot_experiment(output, runs),
                  "first_divergence_from_clean": first_divergence(runs)}
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    return comparison


def excerpts(run: dict, length: int = 120) -> dict[str, dict]:
    """Beginning, middle and end of the generated text, with token offsets."""
    tokens = run["tokens"]["generated"]
    fragments = [row.get("fragment", "") for row in run["telemetry"]]
    result = {}
    for name, start in (("beginning", 0), ("middle", max(0, len(tokens) // 2 - length // 2)),
                        ("end", max(0, len(tokens) - length))):
        result[name] = {"token_offset": start, "tokens": min(length, len(tokens) - start),
                        "text": "".join(fragments[start:start + length])}
    return result


def first_divergence(runs: dict[str, dict]) -> dict[str, int | None]:
    if "clean" not in runs:
        return {}
    base = runs["clean"]["tokens"]["generated"]
    result = {}
    for name, run in runs.items():
        if name == "clean":
            continue
        other = run["tokens"]["generated"]
        result[name] = next((i for i, (a, b) in enumerate(zip(base, other)) if a != b), None)
    return result


def run_experiment(engine: Any, base: ContinuousConfig, conditions: list[tuple[str, int | None]], output: Path,
                   cancel: Event | None = None, on_progress: Callable | None = None) -> dict[str, Any]:
    """Run each (condition, reset_interval) with identical seed, window and decoding; then compare."""
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for condition, interval in conditions:
        config = ContinuousConfig(**{**asdict(base), "condition": condition, "reset_interval": interval})
        name = condition if interval is None else f"{condition}_{interval}"
        LOG.info("condition %s: %d tokens, window %d", name, config.total_tokens, config.window)
        summary = ContinuousRun(engine, config, output / name).run(cancel, on_progress)
        LOG.info("condition %s finished: %s, %d tokens, %.1f tok/s", name, summary["stop_reason"],
                 summary["generated_tokens"], summary["tokens_per_second"])
        names.append(name)
        if cancel is not None and cancel.is_set():
            break
    return compare_runs(output, names)


def parse_conditions(text: str, reset_intervals: list[int]) -> list[tuple[str, int | None]]:
    result = []
    for name in (part.strip() for part in text.split(",") if part.strip()):
        if name == "fly_reset":
            result.extend(("fly_reset", interval) for interval in reset_intervals)
        else:
            result.append((name, None))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous sliding-window generation (Infinite Nekrasov)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/observatory.json")
    parser.add_argument("--model-path")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/cx512")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-text", default=DEFAULT_SEED_TEXT)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--conditions", default="clean,fly,fly_reset")
    parser.add_argument("--reset-intervals", default="128,512")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--feedback", type=float, default=0.25)
    parser.add_argument("--eos", choices=("stop", "suppress"), default="stop")
    parser.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    parser.add_argument("--probe-tokens", type=int, default=16)
    parser.add_argument("--snapshot-interval", type=int, default=64)
    parser.add_argument("--prompt-policy", choices=PROMPT_POLICIES, default="raw")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--decoding", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .observatory import engine_config
    from .observatory_engine import ObservatoryEngine
    settings = json.loads(args.config.read_text())
    device = "cpu" if args.cpu else settings.get("device", "cuda")
    engine = ObservatoryEngine(args.model_path or settings["model"], args.graph, device=device,
                               config=engine_config(settings))
    base = ContinuousConfig(seed_text=args.seed_text, window=args.window, total_tokens=args.tokens,
                            strength=args.strength, feedback=args.feedback, eos_policy=args.eos,
                            checkpoints=tuple(int(x) for x in args.checkpoints.split(",") if x),
                            probe_tokens=args.probe_tokens, snapshot_interval=args.snapshot_interval,
                            prompt_policy=args.prompt_policy, instruction=args.instruction,
                            decoding=args.decoding, temperature=args.temperature, top_p=args.top_p,
                            sampling_seed=args.sampling_seed)
    conditions = parse_conditions(args.conditions, [int(x) for x in args.reset_intervals.split(",") if x])
    result = run_experiment(engine, base, conditions, args.output)
    print(json.dumps([{key: row[key] for key in ("condition", "generated_tokens", "stop_reason", "tokens_per_second")}
                      for row in result["conditions"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
