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
- External injection. Operator text is tokenized raw (no role markers) and teacher-forced
  into the same stream: each injected token is appended in place of a sampled token, and the
  next forward pass processes it (one capture + one feedback update per token, as for any
  token). Forced steps draw no random number. Provenance of every window position is
  tracked, so model tokens are AUTONOMOUS or POST-INJECTION depending on whether an injected
  token is inside the window that produced them.
- External neural cue. Optionally, while the last window token is an injected one, a fixed
  positive input is added to the forcing of the capture update for the 16 DA-annotated
  neurons (or a size-matched random control group). It is an input to the dynamics, not a
  state overwrite, and it stops as soon as autonomous tokens are processed again.
- Leak rate. The reservoir leak dt/tau can be set per run and changed at token boundaries;
  the state is kept. Injections and leak changes are recorded with their stream position,
  so a run can be reproduced exactly from its schedule.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import torch

from .llm import compare_logits, stop_token_ids
from .reservoir import Reservoir  # noqa: F401 (re-export for older scripts)

LOG = logging.getLogger("liquidaion.continuous")
ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = ("clean", "fly", "fly_reset", "random")
DEFAULT_SEED_TEXT = "Однажды в студёную зимнюю пору"
DEFAULT_INSTRUCTION = "Продолжи текст."
PROMPT_POLICIES = ("raw", "chat_prefix")
DEFAULT_CHECKPOINTS = (128, 256, 512, 1024, 2048, 4096)
MAX_WINDOW = 4096
CUE_TARGETS = ("off", "da", "random")
BASELINE_LEAK = 0.2


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
    leak_rate: float = BASELINE_LEAK
    cue_target: str = "off"
    cue_strength: float = 0.5
    # Reproducible intervention schedules: (stream position, text) and (stream position, leak).
    injections: tuple = ()
    leak_schedule: tuple = ()
    # Record-only: the reservoir state after EVERY forward pass (model and injected tokens) in
    # state_trace.npz. states.npz snapshots are taken per generation step, before its forward pass.
    state_trace: bool = False

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
        if not np.isfinite(self.leak_rate) or not 0 < self.leak_rate <= 1:
            raise ValueError("Leak rate must lie in (0, 1]")
        if self.cue_target not in CUE_TARGETS:
            raise ValueError(f"Cue target must be one of {CUE_TARGETS}")
        if not np.isfinite(self.cue_strength) or self.cue_strength < 0:
            raise ValueError("Cue strength must be finite and non-negative")
        for position, text in self.injections:
            if int(position) < 0 or not str(text):
                raise ValueError("Injections need a non-negative position and non-empty text")
        for position, leak in self.leak_schedule:
            if int(position) < 0 or not np.isfinite(leak) or not 0 < leak <= 1:
                raise ValueError("Leak schedule needs non-negative positions and leaks in (0, 1]")

    @property
    def coupled(self) -> bool:
        return self.condition != "clean"


def cue_neuron_indices(graph_dir: str | Path, target: str, seed: int = 42) -> tuple[list[int], list[str]]:
    """Reservoir indices and root IDs of the cue group, read from nodes.csv (graph order).

    "da": every node annotated nt_type == DA. "random": the same number of nodes drawn with a
    fixed seed from the non-DA nodes. Indices equal row positions because load_graph builds
    the graph in nodes.csv order.
    """
    with (Path(graph_dir) / "nodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    da = [i for i, row in enumerate(rows) if row.get("nt_type") == "DA"]
    if target == "da":
        chosen = da
    elif target == "random":
        others = [i for i in range(len(rows)) if i not in set(da)]
        chosen = sorted(np.random.default_rng(seed).choice(others, size=len(da), replace=False).tolist())
    else:
        chosen = []
    return chosen, [rows[i]["root_id"] for i in chosen]


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
        # The selected connectome, or its random-topology control; reservoirs come from the selected provider.
        graph = engine.control_connectome() if config.condition == "random" else engine.connectome
        self.graph = graph
        self.reservoir = engine.create_reservoir(graph, leak=config.leak_rate) if config.coupled else None
        self.coupling = self._coupling(self.reservoir) if config.coupled else None
        seed_ids = seed_token_ids(self.llm.tokenizer, config.seed_text)
        if not seed_ids:
            raise ValueError("Seed text produced no tokens")
        self.seed_ids = seed_ids
        profile = getattr(engine, "profile", None)
        if config.prompt_policy == "chat_prefix" and profile is not None and profile.interaction != "chat":
            raise ValueError(f"chat_prefix uses a chat template; {profile.display} is a Base model, use raw continuation")
        self.prefix_ids = prefix_token_ids(self.llm.tokenizer, config)
        # The only token state inside the engine is this bounded window (plus a fixed prefix).
        self.window_ids = seed_ids[-config.window:]
        # Provenance per window position: None for seed/model tokens, else the injection id.
        self.window_sources: list[int | None] = [None] * len(self.window_ids)
        self.stream: list[tuple[int, int | None]] = []
        self.forced: deque[tuple[int, int]] = deque()
        self._injection_queue: queue.Queue = queue.Queue()
        self._control_queue: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self._ids_lock = threading.Lock()
        self.injections: dict[int, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.event_states: dict[str, np.ndarray] = {}
        self._schedule_injections = sorted((int(p), str(t)) for p, t in config.injections)
        self._schedule_leak = sorted((int(p), float(v)) for p, v in config.leak_schedule)
        has_labels = engine.connectome.nt_types is not None
        self.da_indices, self.da_root_ids = engine.cue_group("da", require_input=False) if has_labels else ([], [])
        target, roots = ([], []) if config.cue_target == "off" or not config.coupled else engine.cue_group(config.cue_target)
        self.cue_indices = target
        self.cue_root_ids = roots if target else []
        self.cue_vector = None
        if config.coupled and target and config.cue_strength > 0:
            self.cue_vector = np.zeros(len(self.reservoir.state))
            self.cue_vector[target] = config.cue_strength
        self.generated: list[int] = []
        self.step = 0
        self.stops = stop_token_ids(self.llm.tokenizer, self.llm.model)
        self.decoder = IncrementalDecoder(self.llm.tokenizer)
        self.forward_calls = 0
        self.reservoir_advances = 0
        self.trace: list[tuple[int, int, int, bool, bool, np.ndarray]] = []
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

    # ------------------------------------------------- operator interventions

    def inject(self, text: str) -> int:
        """Queue operator text from any thread; it is inserted at the next safe token boundary."""
        if not text:
            raise ValueError("Injected text must not be empty")
        with self._ids_lock:
            injection_id = next(self._ids)
        self._injection_queue.put((injection_id, text, "live"))
        return injection_id

    def request_leak(self, leak: float) -> None:
        """Queue a leak-rate change from any thread; applied at the next token boundary."""
        if not np.isfinite(leak) or not 0 < leak <= 1:
            raise ValueError("Leak rate must lie in (0, 1]")
        self._control_queue.put(float(leak))

    @property
    def position(self) -> int:
        return len(self.stream)

    def _event(self, kind: str, **values: Any) -> dict[str, Any]:
        event = {"type": kind, "position": self.position, "step": self.step,
                 "simulation_time": self.reservoir.time if self.reservoir is not None else None, **values}
        self.events.append(event)
        return event

    def _set_leak(self, leak: float, origin: str) -> None:
        if self.reservoir is None:
            self._event("leak_ignored", requested=leak, origin=origin, reason="no reservoir (clean)")
            return
        old = self.reservoir.leak
        self.reservoir.set_leak(leak)
        self._event("leak_changed", old=old, new=self.reservoir.leak, origin=origin)

    def _start_injection(self, injection_id: int, text: str, origin: str) -> None:
        ids = seed_token_ids(self.llm.tokenizer, text)
        record = {"id": injection_id, "text": text, "token_ids": ids, "origin": origin,
                  "start_position": self.position, "start_step": self.step, "inserted": 0,
                  "end_position": None, "left_window_position": None, "status": "inserting" if ids else "empty"}
        self.injections[injection_id] = record
        self._event("injection_started", injection_id=injection_id, tokens=len(ids), origin=origin)
        if self.reservoir is not None:
            self.event_states[f"injection{injection_id}_start"] = self.reservoir.state.copy()
        self.forced.extend((token, injection_id) for token in ids)

    def _apply_controls(self) -> None:
        position = self.position
        while self._schedule_leak and self._schedule_leak[0][0] <= position:
            self._set_leak(self._schedule_leak.pop(0)[1], "schedule")
        while True:
            try:
                leak = self._control_queue.get_nowait()
            except queue.Empty:
                break
            self._set_leak(leak, "live")
        # A new injection starts only when the previous one has been fully inserted.
        if self.forced:
            return
        if self._schedule_injections and self._schedule_injections[0][0] <= position:
            _, text = self._schedule_injections.pop(0)
            with self._ids_lock:
                injection_id = next(self._ids)
            self._start_injection(injection_id, text, "schedule")
            return
        try:
            injection_id, text, origin = self._injection_queue.get_nowait()
        except queue.Empty:
            return
        self._start_injection(injection_id, text, origin)

    def _cancel_pending(self, reason: str) -> None:
        """Record every intervention that was not (fully) inserted; nothing is dropped silently."""
        if self.forced:
            injection_id = self.forced[0][1]
            self.injections[injection_id]["status"] = f"partial_{reason}"
            self._event("injection_interrupted", injection_id=injection_id, reason=reason,
                        inserted=self.injections[injection_id]["inserted"], remaining=len(self.forced))
            self.forced.clear()
        while True:
            try:
                injection_id, text, origin = self._injection_queue.get_nowait()
            except queue.Empty:
                break
            self.injections[injection_id] = {"id": injection_id, "text": text, "origin": origin, "token_ids": [],
                                             "status": f"not_inserted_{reason}", "start_position": None,
                                             "start_step": None, "inserted": 0, "end_position": None,
                                             "left_window_position": None}
            self._event("injection_not_inserted", injection_id=injection_id, reason=reason)
        for position, text in self._schedule_injections:
            self._event("scheduled_injection_not_reached", scheduled_position=position, text=text, reason=reason)
        self._schedule_injections = []

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
        """Process one stream token: sample a model token, or teacher-force one injected token."""
        config = self.config
        self._apply_controls()
        autonomous = not self.forced
        reset = False
        if autonomous:
            if (config.condition == "fly_reset" and self.step > 0 and self.step % config.reset_interval == 0
                    and (not self.resets or self.resets[-1] != self.step)):
                self.reservoir.reset()
                self.resets.append(self.step)
                reset = True
            if self.step in config.checkpoints and self.reservoir is not None \
                    and not any(p["step"] == self.step for p in self.pending_probes):
                self._open_probe()
            if self.reservoir is not None:
                self.snapshot_if_due()
        window_external = sum(source is not None for source in self.window_sources)
        # The cue accompanies the processing of an injected token: the newest window token is external.
        cue_applied = self.cue_vector is not None and self.window_sources[-1] is not None
        self.last_advances = 0
        if self.coupling is not None:
            self.coupling.cue = self.cue_vector if cue_applied else None
        try:
            logits = self._forward_logits(self.window_ids, self.coupling)
        finally:
            if self.coupling is not None:
                self.coupling.cue = None
        self.forward_calls += 1
        self.reservoir_advances += self.last_advances
        if config.state_trace and self.reservoir is not None:
            self.trace.append((self.forward_calls - 1, self.step, self.position, bool(cue_applied), autonomous,
                               self.reservoir.state.copy()))
        suppressed = None
        if autonomous:
            for probe in self.pending_probes:
                if probe["step"] == self.step and "main_logits" not in probe:
                    probe["main_logits"] = logits.numpy().copy()
            token, suppressed = self.select(logits)
            injection_id = None
            if suppressed is not None:
                self.eos_suppressed.append({"step": self.step, "eos_token_id": suppressed})
            elif token in self.stops:
                self.stop_reason = "eos"
            category = "post_injection" if window_external else "autonomous"
        else:
            token, injection_id = self.forced.popleft()
            category = "external"
        row: dict[str, Any] = {"step": self.step, "position": self.position, "token_id": token,
                               "source": "model" if autonomous else "external", "injection_id": injection_id,
                               "category": category, "window_len": len(self.window_ids),
                               "window_start": len(self.seed_ids) + self.position - len(self.window_ids),
                               "window_external": window_external, "cue_applied": cue_applied,
                               "reset": reset, "eos_suppressed": suppressed, "reservoir_advances": self.last_advances}
        if self.coupling is not None and self.coupling.records:
            record = self.coupling.records[-1]
            state = self.reservoir.state
            row.update({"state_rms": float(np.sqrt(np.mean(state ** 2))), "state_norm": float(np.linalg.norm(state)),
                        "delta_relative_rms": record["delta_relative_rms"],
                        "steering_vector_norm": record["steering_vector_norm"],
                        "feedback_input_norm": record.get("feedback_input_norm", 0.0),
                        "input_norm": record["input_norm"], "capture_rms": record["capture_rms"],
                        "leak": self.reservoir.leak,
                        # Layers whose hooks actually fired in this forward pass.
                        "capture_layer": record["capture_layer"], "steer_layer": record["steer_layer"]})
            if self.da_indices:
                da = state[self.da_indices]
                row.update({"da_mean": float(da.mean()), "da_abs_mean": float(np.abs(da).mean())})
            # Per-step traces would grow without bound over thousands of tokens.
            self.coupling.clear_trace()
        if self.stop_reason == "eos":
            row["fragment"] = ""
            return row
        before = {source for source in self.window_sources if source is not None}
        self.stream.append((token, injection_id))
        self.window_ids = (self.window_ids + [token])[-config.window:]
        self.window_sources = (self.window_sources + [injection_id])[-config.window:]
        self.max_window_seen = max(self.max_window_seen, len(self.window_ids))
        row["fragment"] = self.decoder.push(token)
        if autonomous:
            self.generated.append(token)
            self.step += 1
        else:
            record = self.injections[injection_id]
            record["inserted"] += 1
            if not any(source == injection_id for _, source in self.forced):
                record["status"] = "inserted"
                record["end_position"] = self.position
                self._event("injection_completed", injection_id=injection_id, tokens=record["inserted"])
                if self.reservoir is not None:
                    self.event_states[f"injection{injection_id}_end"] = self.reservoir.state.copy()
        after = {source for source in self.window_sources if source is not None}
        for injection_id_left in sorted(before - after):
            # The last token of this injection has just been evicted from the model's window.
            self.injections[injection_id_left]["left_window_position"] = self.position
            self._event("injection_left_window", injection_id=injection_id_left)
            if self.reservoir is not None:
                self.event_states[f"injection{injection_id_left}_left_window"] = self.reservoir.state.copy()
        row["window_external_after"] = len([s for s in self.window_sources if s is not None])
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
        self.pending_probes.append({"step": self.step, "position": self.position, "context": list(self.window_ids),
                                    "sources": list(self.window_sources), "state": self.reservoir.state.copy(),
                                    "time": self.reservoir.time, "leak": self.reservoir.leak})

    def _run_due_probes(self, force: bool = False) -> None:
        due = [p for p in self.pending_probes
               if force or self.position >= p["position"] + self.config.probe_tokens]
        for probe in due:
            self.pending_probes.remove(probe)
            forced = self.stream[probe["position"]:probe["position"] + self.config.probe_tokens]
            if forced:
                self.probes.append(self.probe(probe, forced))

    def probe(self, checkpoint: dict[str, Any], forced: list[tuple[int, int | None]]) -> dict[str, Any]:
        """Score identical stream tokens under retained, zero and no reservoir state on cloned reservoirs.

        Clones use the leak in force at the checkpoint and the same cue rule as the main run.
        """
        state_before = self.reservoir.state.copy()
        time_before = self.reservoir.time

        def run(state: np.ndarray | None) -> np.ndarray:
            if state is None:
                coupling = None
            else:
                reservoir = self.engine.create_reservoir(self.graph, leak=checkpoint["leak"])
                reservoir.state = state.copy()
                reservoir.time = checkpoint["time"]
                coupling = self._coupling(reservoir)
            window, sources = list(checkpoint["context"]), list(checkpoint.get("sources", [None] * len(checkpoint["context"])))
            rows = []
            for token, source in forced:
                if coupling is not None:
                    coupling.cue = self.cue_vector if (self.cue_vector is not None and sources[-1] is not None) else None
                rows.append(self._forward_logits(window, coupling).numpy())
                if coupling is not None:
                    coupling.cue = None
                    coupling.clear_trace()
                window = (window + [token])[-self.config.window:]
                sources = (sources + [source])[-self.config.window:]
            return np.stack(rows)

        retained = run(checkpoint["state"])
        reset = run(np.zeros_like(checkpoint["state"]))
        clean = run(None)
        if not np.array_equal(self.reservoir.state, state_before) or self.reservoir.time != time_before:
            raise RuntimeError("Checkpoint probe mutated the main reservoir")
        return {"step": checkpoint["step"], "position": checkpoint["position"], "probe_tokens": len(forced),
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
        language_model = self.engine.model_info() if hasattr(self.engine, "model_info") else None
        if language_model is not None and self.coupling is not None:
            language_model["projection_fingerprint"] = self.coupling.projection_fingerprint
        return {"experiment": "continuous_sliding_window", "config": asdict(self.config),
                "language_model": language_model,
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
                "reservoir_dynamics": {"dt": self.reservoir.dynamics.dt, "tau": self.reservoir.dynamics.tau,
                                       "leak": self.reservoir.leak, "gain": self.reservoir.dynamics.gain,
                                       "input_strength": self.reservoir.dynamics.input_strength}
                if self.reservoir is not None else None,
                "cue": {"target": self.config.cue_target, "strength": self.config.cue_strength,
                        "indices": self.cue_indices, "root_ids": self.cue_root_ids,
                        "applied_to": ("capture update forcing" if len(self.engine.config.active_bypasses) == 1
                                       else "the single synchronous update forcing")
                                      + ", while the newest window token is injected",
                        "active": self.cue_vector is not None},
                "da_neurons": {"indices": self.da_indices, "root_ids": self.da_root_ids},
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
            self._cancel_pending(self.stop_reason or "error")
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
                "stream_tokens": len(self.stream), "external_tokens": sum(s is not None for _, s in self.stream),
                "injections": [self.injections[k] for k in sorted(self.injections)], "events": self.events,
                "final_leak": self.reservoir.leak if self.reservoir is not None else None,
                "final_state_rms": float(np.sqrt(np.mean(self.reservoir.state ** 2))) if self.reservoir is not None else None}

    def _save(self) -> None:
        steps = sorted(self.snapshots)
        np.savez_compressed(self.output / "states.npz", steps=np.array(steps, dtype=np.int64),
                            states=np.stack([self.snapshots[s] for s in steps]) if steps else np.empty((0, 0)),
                            **{f"event_{name}": state for name, state in self.event_states.items()})
        if self.trace:
            forward, step, position, cue, autonomous, states = zip(*self.trace)
            np.savez_compressed(self.output / "state_trace.npz", forward=np.array(forward), step=np.array(step),
                                position=np.array(position), cue_applied=np.array(cue),
                                autonomous=np.array(autonomous), states=np.stack(states))
        (self.output / "token_ids.json").write_text(json.dumps({"seed": self.seed_ids, "generated": self.generated,
                                                                "stream": [[t, s] for t, s in self.stream]}) + "\n")
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
        states = {"steps": data["steps"].tolist(), "states": data["states"],
                  "events": {key[len("event_"):]: data[key] for key in data.files if key.startswith("event_")}}
    trace = None
    if (path / "state_trace.npz").exists():
        with np.load(path / "state_trace.npz") as data:
            trace = {key: data[key] for key in data.files}
    return {"state_trace": trace, "metadata": json.loads((path / "metadata.json").read_text()),
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
    parser.add_argument("--leak", type=float, default=BASELINE_LEAK, help="Reservoir leak dt/tau (baseline 0.2)")
    parser.add_argument("--cue", choices=CUE_TARGETS, default="off", help="External neural cue target group")
    parser.add_argument("--cue-strength", type=float, default=0.5)
    parser.add_argument("--inject", action="append", default=[], metavar="POSITION:TEXT",
                        help="Insert TEXT when the stream (after the seed) reaches POSITION tokens; repeatable")
    parser.add_argument("--leak-at", action="append", default=[], metavar="POSITION:LEAK")
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
                            sampling_seed=args.sampling_seed, leak_rate=args.leak, cue_target=args.cue,
                            cue_strength=args.cue_strength,
                            injections=tuple((int(item.split(":", 1)[0]), item.split(":", 1)[1]) for item in args.inject),
                            leak_schedule=tuple((int(item.split(":", 1)[0]), float(item.split(":", 1)[1]))
                                                for item in args.leak_at))
    conditions = parse_conditions(args.conditions, [int(x) for x in args.reset_intervals.split(",") if x])
    result = run_experiment(engine, base, conditions, args.output)
    print(json.dumps([{key: row[key] for key in ("condition", "generated_tokens", "stop_reason", "tokens_per_second")}
                      for row in result["conditions"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
