"""Single frozen transformer, two independent chats, and measured reservoir activity.

The engine is owned by one worker. Its lock also serializes direct API calls; only
Telemetry.snapshot and a cancellation Event need to be accessed from the UI.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from typing import Any, Callable

import numpy as np
import torch

from .connectome import load_graph, sha256
from .debuglog import InferenceDebugLog, digest
from .environment import inspect_environment
from .llm import Coupling, LocalLLM, compare_logits, stop_token_ids
from .recording import OfflineReplay, SessionRecording
from .reservoir import Reservoir, random_control
from .telemetry import Telemetry

# Upper bound offered for max_new_tokens; the effective bound is the context budget.
MAX_NEW_TOKENS_LIMIT = 4096
# Prompt plus reserved generation above this fraction of the context is reported as near exhaustion.
CONTEXT_WARNING_FRACTION = 0.9
# Stop reasons: "eos" is a normal end of turn; "max_new_tokens" is a truncated answer.
STOP_REASONS = ("eos", "max_new_tokens", "cancelled", "cancelled_before_start", "forced_complete")


class ContextOverflowError(ValueError):
    """The latest user message cannot fit even after dropping all earlier turns."""


@dataclass
class ObservatoryConfig:
    max_context_tokens: int = 4096
    max_new_tokens: int = 512
    strength: float = 0.25
    feedback: float = 0.25
    steering_enabled: bool = True
    feedback_enabled: bool = True
    capture_layer: int = 6
    steer_layer: int = 14
    features: int = 64
    seed: int = 42
    threads: int = 4
    revision: str | None = None

    def validate(self) -> None:
        if self.max_context_tokens < 16 or not 1 <= self.max_new_tokens < self.max_context_tokens:
            raise ValueError("Context must have at least 16 tokens and reserve generation space")
        if self.max_new_tokens > MAX_NEW_TOKENS_LIMIT:
            raise ValueError(f"max_new_tokens may not exceed {MAX_NEW_TOKENS_LIMIT}")
        if not 0 <= self.capture_layer < self.steer_layer:
            raise ValueError("Capture layer must precede the steering layer")
        if self.features < 1 or self.threads < 1:
            raise ValueError("Features and threads must be positive")
        if any(not np.isfinite(x) or not 0 <= x <= 1 for x in (self.strength, self.feedback)):
            raise ValueError("Strength and feedback must lie in [0, 1]")


@dataclass
class ChatSession:
    # Full history. Assistant entries also carry stop_reason/generated_tokens metadata,
    # which is stripped before the chat template is applied.
    messages: list[dict[str, Any]]
    cache: Any = None
    cached_ids: list[int] | None = None


@dataclass
class ContextPlan:
    """The exact model input for one generation and how it was derived from history."""
    messages: list[dict[str, str]]
    text: str
    ids: torch.Tensor
    budget: int
    untrimmed_tokens: int
    dropped_messages: int = 0
    trim_steps: list[dict[str, int]] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return int(self.ids.shape[1])


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


class MeasuredCoupling(Coupling):
    """Expose actual input and feedback without changing the established pathway."""
    def capture(self, module: Any, args: tuple, output: Any) -> None:
        super().capture(module, args, output)
        self.current["input_norm"] = float(np.linalg.norm(self.current["features"]))
        mapped = self.reservoir.dynamics.input_strength * (self.reservoir.input @ np.asarray(self.current["features"]))
        self.current["mapped_input_norm"] = float(np.linalg.norm(mapped))

    def steer(self, module: Any, args: tuple, output: Any) -> Any:
        original = output[0] if isinstance(output, tuple) else output
        result = super().steer(module, args, output)
        modified = result[0] if isinstance(result, tuple) else result
        self.records[-1]["steering_vector_norm"] = float((modified[:, -1].float() - original[:, -1].float()).norm())
        self.records[-1]["feedback_enabled"] = bool(self.feedback)
        self.records[-1]["feedback_strength"] = self.feedback
        self.records[-1]["feedback_state_rms"] = float(np.sqrt(np.mean(self.reservoir.state ** 2)))
        return result


class ObservatoryEngine:
    def __init__(self, model_path: str | Path, graph_dir: str | Path,
                 device: str = "cpu", config: ObservatoryConfig | dict | None = None):
        if isinstance(config, dict):
            config = ObservatoryConfig(**config)
        self.config = config or ObservatoryConfig()
        self.config.validate()
        resolved = torch.device(device)
        if resolved.type == "cpu":
            resolved = torch.device("cpu")
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("Observatory supports CPU and CUDA devices only")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable; no CPU fallback")
        self._lock = RLock()
        self.graph_dir = Path(graph_dir)
        self.graph, self.node_ids = load_graph(self.graph_dir)
        self.reservoir = Reservoir(self.graph, features=self.config.features, seed=self.config.seed)
        # This is the sole LocalLLM and model instance, shared across all conditions.
        self.llm = LocalLLM(str(model_path), device=str(resolved), revision=self.config.revision,
                            threads=self.config.threads)
        if self.config.steer_layer >= len(self.llm.model.model.layers):
            raise ValueError("Configured layer indices exceed the loaded transformer")
        self.model_context_limit = getattr(self.llm.model.config, "max_position_embeddings", None)
        self._check_model_context(self.config)
        self.coupling = self._coupling(self.reservoir)
        self.sessions = {name: ChatSession([]) for name in ("clean", "fly")}
        self.session_id = uuid.uuid4().hex
        self.turn_index = 0
        self.debug_log: InferenceDebugLog | None = None
        revision_path = Path(model_path) / "revision.json"
        revision_info = json.loads(revision_path.read_text()) if revision_path.exists() else {}
        revision = revision_info.get("revision", self.llm.revision)
        if self.config.revision and revision and self.config.revision != revision:
            raise ValueError("Local model revision differs from the requested immutable revision")
        self.metadata = {"model": str(model_path), "model_revision": revision,
                         "connectome_hash": sha256(self.graph_dir / "edges.csv"),
                         "node_hash": sha256(self.graph_dir / "nodes.csv"),
                         "reservoir_fingerprint": self.reservoir.fingerprint,
                         "nodes": len(self.node_ids), "edges": int(self.graph.nnz),
                         "hardware": {**inspect_environment(), "device": str(resolved)},
                         "decoding": "greedy", "weights": "pretrained"}
        self.recording = SessionRecording(self.metadata)
        self._feedback_events = 0
        self.telemetry = Telemetry(**self.metadata, device=str(resolved), status="ready", branch=None,
                                   generated_tokens=0, tokens_per_second=0.0,
                                   input_norm=0.0, mapped_input_norm=0.0, steering_vector_norm=0.0,
                                   delta_relative_rms=0.0, token_index=None)
        self._publish("initialized")

    def _coupling(self, reservoir: Reservoir, mode: str = "fly", strength: float | None = None,
                  feedback: float | None = None) -> MeasuredCoupling:
        if strength is None:
            strength = self.config.strength if self.config.steering_enabled else 0.0
        if feedback is None:
            feedback = self.config.feedback if self.config.feedback_enabled else 0.0
        return MeasuredCoupling(reservoir, self.llm.model.config.hidden_size,
                                self.config.capture_layer, self.config.steer_layer,
                                strength, feedback, mode=mode, seed=self.config.seed)

    def _check_model_context(self, config: ObservatoryConfig) -> None:
        if self.model_context_limit and config.max_context_tokens > self.model_context_limit:
            raise ValueError(f"Context of {config.max_context_tokens} tokens exceeds the model's "
                             f"{self.model_context_limit} positions")

    def enable_debug_log(self, path: str | Path, level: str = "summary") -> Path:
        with self._lock:
            self.debug_log = InferenceDebugLog(path, level)
            return self.debug_log.path

    def disable_debug_log(self) -> None:
        with self._lock:
            self.debug_log = None

    def _hook_count(self) -> int:
        return sum(len(layer._forward_hooks) for layer in self.llm.model.model.layers)

    def _publish(self, event: str, callback: Callable | None = None, **values: Any) -> dict:
        actual_reservoir = values.pop("actual_reservoir", self.reservoir)
        state = actual_reservoir.state
        memory = {}
        if self.llm.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.llm.device)
            memory = {"gpu_allocated_bytes": torch.cuda.memory_allocated(self.llm.device),
                      "gpu_reserved_bytes": torch.cuda.memory_reserved(self.llm.device),
                      "gpu_free_bytes": free, "gpu_total_bytes": total}
        snapshot = self.telemetry.update(state=state.tolist(), activity=np.abs(state).tolist(),
                                         state_rms=float(np.sqrt(np.mean(state ** 2))),
                                         simulation_time=actual_reservoir.time,
                                         settings=asdict(self.config),
                                         timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                         simulation_step=int(round(actual_reservoir.time / actual_reservoir.dynamics.dt)),
                                         feedback_event_count=self._feedback_events, **memory, **values)
        self.recording.append(event, snapshot=snapshot)
        if callback:
            callback(snapshot)
        return snapshot

    def configure(self, **settings: Any) -> dict:
        with self._lock:
            mutable = {"strength", "feedback", "steering_enabled", "feedback_enabled", "max_new_tokens",
                       "max_context_tokens"}
            if set(settings) - mutable:
                raise ValueError("Only runtime steering, feedback, and context settings may be configured")
            candidate = ObservatoryConfig(**{**asdict(self.config), **settings})
            candidate.validate()
            self._check_model_context(candidate)
            context_changed = candidate.max_context_tokens != self.config.max_context_tokens
            self.config = candidate
            self.coupling.strength = candidate.strength if candidate.steering_enabled else 0.0
            self.coupling.feedback = candidate.feedback if candidate.feedback_enabled else 0.0
            if set(settings) & {"strength", "feedback", "steering_enabled", "feedback_enabled"}:
                self.sessions["fly"].cache = self.sessions["fly"].cached_ids = None
            if context_changed:
                for session in self.sessions.values():
                    session.cache = None
                    session.cached_ids = None
            self._publish("settings_changed")
            return asdict(candidate)

    def _render(self, messages: list[dict[str, str]]) -> str:
        return self.llm.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)

    def _build_context(self, session: ChatSession, prompt: str, reserve: int) -> ContextPlan:
        """Render the latest user turn after as much complete history as the budget allows.

        Whole oldest exchanges are dropped from the model input only; the session history is
        left intact. The latest user message is never truncated or replaced: if it cannot fit
        on its own, ContextOverflowError is raised instead of feeding a malformed template.
        """
        budget = self.config.max_context_tokens - reserve
        if budget < 1:
            raise ContextOverflowError("Generation reserve leaves no room for the prompt")
        history = session.messages
        latest = {"role": "user", "content": prompt}
        start = 0
        plan: ContextPlan | None = None
        while True:
            messages = template_messages(history[start:]) + [latest]
            text = self._render(messages)
            ids = self.llm.tokenizer(text, return_tensors="pt")["input_ids"]
            if plan is None:
                plan = ContextPlan(messages, text, ids, budget, int(ids.shape[1]))
            else:
                plan.trim_steps.append({"dropped_messages": start, "tokens": int(ids.shape[1])})
                plan.messages, plan.text, plan.ids, plan.dropped_messages = messages, text, ids, start
            if ids.shape[1] <= budget:
                break
            if start >= len(history):
                raise ContextOverflowError(
                    f"The latest message needs {ids.shape[1]} prompt tokens but only {budget} remain "
                    f"after reserving {reserve} new tokens in a {self.config.max_context_tokens}-token "
                    "context. Shorten the message, lower the new-token limit or raise the context.")
            # Drop one complete exchange: advance to the next user message after `start`.
            start += 1
            while start < len(history) and history[start]["role"] != "user":
                start += 1
        self._check_latest_message(plan, prompt)
        return plan

    def _check_latest_message(self, plan: ContextPlan, prompt: str) -> None:
        # The template must end with exactly the latest user message and the generation prompt.
        marker = "\x00LIQUIDAION\x00"
        tail = self._render([{"role": "user", "content": marker}]).split(marker, 1)[1]
        if not plan.text.endswith(prompt + tail):
            raise RuntimeError("Rendered model input does not end with the latest user message")

    def _forward(self, ids: torch.Tensor, cache: Any, coupling: Coupling | None,
                 hooks: dict[str, int] | None = None):
        if coupling:
            coupling.attach(self.llm.model)
            if hooks is not None:
                hooks["attached"] += 1
        elif self._hook_count():
            # Uncoupled inference must never run through Fly hooks left from another stream.
            raise RuntimeError("Forward hooks are present during uncoupled inference")
        try:
            with torch.inference_mode():
                return self.llm.model(input_ids=ids, past_key_values=cache, use_cache=True)
        finally:
            if coupling:
                coupling.detach()
                if hooks is not None:
                    hooks["detached"] += 1

    def _generate(self, branch: str, prompt: str, limit: int, cancel: Event,
                  on_token: Callable | None, on_telemetry: Callable | None,
                  session: ChatSession | None = None, coupling: Coupling | None = None,
                  forced_tokens: list[int] | None = None, scientific: bool = False,
                  prompt_reserve: int | None = None, turn: int | None = None) -> dict:
        session = session or self.sessions[branch]
        reserve = limit if prompt_reserve is None else prompt_reserve
        plan = self._build_context(session, prompt, reserve)
        ids = plan.ids.to(self.llm.device)
        prompt_ids = ids[0].tolist()
        cached_ids = session.cached_ids or []
        reuse = bool(session.cache is not None and cached_ids and
                     len(cached_ids) < len(prompt_ids) and prompt_ids[:len(cached_ids)] == cached_ids)
        if reuse:
            cache_event = "reuse"
        elif session.cache is not None:
            cache_event = "discard_prefix_mismatch"
        else:
            cache_event = "new"
        cache = session.cache if reuse else None
        current = ids[:, len(cached_ids):] if reuse else ids
        processed = prompt_ids.copy()
        generated: list[int] = []
        logits: list[np.ndarray] = []
        stops = stop_token_ids(self.llm.tokenizer, self.llm.model)
        hooks = {"attached": 0, "detached": 0, "present_before": self._hook_count()}
        reservoir = coupling.reservoir if coupling else None
        rms_before = float(np.sqrt(np.mean(reservoir.state ** 2))) if reservoir is not None else None
        stop_reason = "forced_complete" if forced_tokens is not None else "max_new_tokens"
        context_near_limit = bool(plan.dropped_messages or
                                  plan.tokens + limit > CONTEXT_WARNING_FRACTION * self.config.max_context_tokens)
        start = time.perf_counter()
        if coupling:
            coupling.clear_trace()
        self._publish("generation_started", on_telemetry, status="generating", branch=branch,
                      generated_tokens=0, tokens_per_second=0.0, input_norm=0.0,
                      delta_relative_rms=0.0, mapped_input_norm=0.0, steering_vector_norm=0.0, token_index=None,
                      prompt_tokens=plan.tokens, context_limit=self.config.max_context_tokens,
                      max_new_tokens=limit, history_dropped_messages=plan.dropped_messages,
                      context_near_limit=context_near_limit, stop_reason=None,
                      actual_reservoir=coupling.reservoir if coupling else self.reservoir)
        if plan.dropped_messages:
            self.recording.append("history_trimmed", branch=branch, dropped_messages=plan.dropped_messages,
                                  untrimmed_tokens=plan.untrimmed_tokens, prompt_tokens=plan.tokens,
                                  budget=plan.budget)
        try:
            for step in range(limit):
                if cancel.is_set():
                    stop_reason = "cancelled"
                    break
                if coupling:
                    coupling.token_position = len(processed) - 1
                result = self._forward(current, cache, coupling, hooks)
                distribution = result.logits[0, -1].float().detach().cpu().numpy()
                if scientific:
                    logits.append(distribution)
                token = int(distribution.argmax()) if forced_tokens is None else int(forced_tokens[step])
                generated.append(token)
                cache = result.past_key_values
                session.cache, session.cached_ids = cache, processed.copy()
                text = self.llm.tokenizer.decode(generated, skip_special_tokens=True)
                elapsed = time.perf_counter() - start
                measured = {}
                if coupling and coupling.records:
                    record = coupling.records[-1]
                    if record["feedback_enabled"]:
                        self._feedback_events += 1
                        self.recording.append("feedback", branch=branch, token_index=step,
                                              feedback_strength=record["feedback_strength"],
                                              state_rms=record["feedback_state_rms"])
                    measured = {key: record[key] for key in ("input_norm", "mapped_input_norm", "steering_vector_norm", "delta_relative_rms", "capture_rms",
                               "before_rms", "after_rms", "feedback_enabled", "feedback_strength",
                               "feedback_state_rms")}
                self._publish("token", on_telemetry, branch=branch, generated_tokens=len(generated),
                              tokens_per_second=len(generated) / max(elapsed, 1e-9), token_index=step,
                              actual_reservoir=coupling.reservoir if coupling else self.reservoir, **measured)
                self.recording.append("generated_token", branch=branch, token_id=token, text=text,
                                      scientific=scientific)
                if on_token:
                    on_token(branch, token, text)
                if forced_tokens is None and token in stops:
                    stop_reason = "eos"
                    break
                current = torch.tensor([[token]], device=self.llm.device)
                processed.append(token)
        except BaseException:
            # A failed forward may mutate a DynamicCache; never reuse it. History is not committed.
            session.cache = session.cached_ids = None
            self._publish("generation_error", on_telemetry, status="error", branch=branch)
            raise
        finally:
            hooks["present_after"] = self._hook_count()
        elapsed = time.perf_counter() - start
        text = self.llm.tokenizer.decode(generated, skip_special_tokens=True)
        session.messages.extend([{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": text, "stop_reason": stop_reason,
                                  "generated_tokens": len(generated)}])
        rms_after = float(np.sqrt(np.mean(reservoir.state ** 2))) if reservoir is not None else None
        report = {"stop_reason": stop_reason, "truncated": stop_reason == "max_new_tokens",
                  "prompt_tokens": plan.tokens, "context_limit": self.config.max_context_tokens,
                  "max_new_tokens": limit, "history_dropped_messages": plan.dropped_messages,
                  "context_near_limit": context_near_limit, "cache_event": cache_event}
        self._debug_record(branch, turn, plan, prompt, prompt_ids, generated, text, limit, report,
                           cached_ids, reuse, hooks, coupling, rms_before, rms_after,
                           len(session.messages) - 2)
        self._publish("generation_finished", on_telemetry,
                      status="cancelled" if stop_reason == "cancelled" else "ready",
                      branch=branch, generated_tokens=len(generated),
                      tokens_per_second=len(generated) / max(elapsed, 1e-9),
                      actual_reservoir=coupling.reservoir if coupling else self.reservoir,
                      **{key: report[key] for key in ("stop_reason", "prompt_tokens", "context_limit",
                         "max_new_tokens", "history_dropped_messages", "context_near_limit")})
        return {"prompt": prompt, "text": text, "token_ids": generated, "prompt_ids": prompt_ids,
                "seconds": elapsed, "tokens_per_second": len(generated) / max(elapsed, 1e-9),
                "cancelled": stop_reason == "cancelled", "cache_reused": reuse, **report,
                "logits": np.stack(logits) if logits else np.empty((0, 0)),
                "trace": [dict(record) for record in coupling.records] if coupling else []}

    def _debug_record(self, branch: str, turn: int | None, plan: ContextPlan, prompt: str,
                      prompt_ids: list[int], generated: list[int], text: str, limit: int,
                      report: dict, cached_ids: list[int], reuse: bool, hooks: dict,
                      coupling: Coupling | None, rms_before: float | None, rms_after: float | None,
                      history_before: int) -> None:
        if self.debug_log is None:
            return
        tokenizer = self.llm.tokenizer
        record = {
            "session_id": self.session_id, "branch": branch, "turn_index": turn,
            "input_token_count": plan.tokens, "max_new_tokens": limit,
            "generated_token_count": len(generated), **report,
            "history_messages_total": history_before,
            "history_messages_in_input": len(plan.messages) - 1,
            "history_trimming": {"dropped_messages": plan.dropped_messages, "untrimmed_tokens": plan.untrimmed_tokens,
                                 "budget": plan.budget, "steps": plan.trim_steps},
            "kv_cache": {"event": report["cache_event"], "cached_tokens_before": len(cached_ids),
                         "reused_tokens": len(cached_ids) if reuse else 0},
            "hooks": hooks,
            "coupled": coupling is not None,
            "steering_enabled": bool(coupling and coupling.strength > 0 and coupling.mode != "none"),
            "steering_strength": coupling.strength if coupling else 0.0,
            "feedback_enabled": bool(coupling and coupling.feedback > 0),
            "feedback_strength": coupling.feedback if coupling else 0.0,
            "coupling_mode": coupling.mode if coupling else None,
            "reservoir_rms_before": rms_before, "reservoir_rms_after": rms_after,
            "rendered_sha256": digest(plan.text), "input_ids_sha256": digest(prompt_ids),
            "latest_user_sha256": digest(prompt),
            # Full-level fields; dropped by the logger at summary level.
            "rendered_text": plan.text, "input_ids": prompt_ids,
            "input_tokens": tokenizer.convert_ids_to_tokens(prompt_ids)
                if hasattr(tokenizer, "convert_ids_to_tokens") else None,
            "generated_ids": generated, "generated_text": text, "latest_user_message": prompt,
        }
        self.debug_log.write(record)

    def generate_pair(self, prompt: str, max_new_tokens: int | None = None,
                      cancel_event: Event | None = None, on_token: Callable | None = None,
                      on_telemetry: Callable | None = None) -> dict:
        with self._lock:
            limit = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
            if not isinstance(limit, int) or not 1 <= limit < self.config.max_context_tokens:
                raise ValueError("Generation limit must be positive and smaller than context")
            if limit > MAX_NEW_TOKENS_LIMIT:
                raise ValueError(f"Generation limit may not exceed {MAX_NEW_TOKENS_LIMIT}")
            # Fail before any branch runs if either history cannot carry the new message.
            for session in self.sessions.values():
                self._build_context(session, prompt, limit)
            cancel = cancel_event if cancel_event is not None else Event()
            turn = self.turn_index
            # A pair is atomic: if any branch fails, neither history keeps this turn.
            committed = {branch: len(session.messages) for branch, session in self.sessions.items()}
            try:
                result = self._generate_branches(prompt, limit, cancel, on_token, on_telemetry, turn)
            except torch.cuda.OutOfMemoryError as exc:
                self._rollback(committed)
                torch.cuda.empty_cache()
                raise RuntimeError("GPU memory was exhausted while processing this turn. Lower the Context "
                                   "setting (eager attention needs about 5 GB for a 5k-token prompt).") from exc
            except BaseException:
                self._rollback(committed)
                raise
            self.turn_index += 1
            self.recording.append("message_pair", prompt=prompt, turn_index=turn,
                                  responses={name: result[name]["text"] for name in result},
                                  stop_reasons={name: result[name]["stop_reason"] for name in result})
            return result

    def _rollback(self, committed: dict[str, int]) -> None:
        for branch, length in committed.items():
            session = self.sessions[branch]
            if len(session.messages) > length:
                del session.messages[length:]
                session.cache = session.cached_ids = None

    def _generate_branches(self, prompt: str, limit: int, cancel: Event, on_token: Callable | None,
                           on_telemetry: Callable | None, turn: int) -> dict:
        result = {}
        for branch in ("clean", "fly"):
            if cancel.is_set():
                # Keep both histories aligned on user turns even if a branch never ran.
                self.sessions[branch].messages.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "", "stop_reason": "cancelled_before_start",
                     "generated_tokens": 0}])
                result[branch] = {"text": "", "token_ids": [], "cancelled": True, "seconds": 0.0,
                                  "stop_reason": "cancelled_before_start", "truncated": False}
                continue
            result[branch] = self._generate(branch, prompt, limit, cancel, on_token, on_telemetry,
                                 coupling=self.coupling if branch == "fly" else None, turn=turn)
        return result

    def idle_tick(self, steps: int = 1) -> dict:
        with self._lock:
            self.reservoir.advance(steps=steps)
            return self._publish("idle_tick", status="idle", branch=None, input_norm=0.0,
                                 delta_relative_rms=0.0, mapped_input_norm=0.0, steering_vector_norm=0.0, token_index=None)

    def reset_state(self) -> dict:
        with self._lock:
            self.reservoir.reset()
            self.coupling.clear_trace()
            # Old Fly cache carries interventions from its former reservoir state.
            self.sessions["fly"].cache = self.sessions["fly"].cached_ids = None
            return self._publish("state_reset", status="ready", input_norm=0.0, mapped_input_norm=0.0,
                                 steering_vector_norm=0.0, delta_relative_rms=0.0, token_index=None)

    def clear_chats(self) -> dict:
        with self._lock:
            self.sessions = {name: ChatSession([]) for name in ("clean", "fly")}
            self.turn_index = 0
            return self._publish("chats_cleared", status="ready", branch=None, generated_tokens=0)

    def new_session(self) -> dict:
        with self._lock:
            self.reservoir.reset()
            self.coupling.clear_trace()
            self.sessions = {name: ChatSession([]) for name in ("clean", "fly")}
            self.recording = SessionRecording(self.metadata)
            self._feedback_events = 0
            self.session_id = uuid.uuid4().hex
            self.turn_index = 0
            return self._publish("new_session", status="ready", branch=None, generated_tokens=0,
                                 input_norm=0.0, mapped_input_norm=0.0, steering_vector_norm=0.0,
                                 delta_relative_rms=0.0, token_index=None)

    def scientific_compare(self, prompt: str, comparison: str = "clean_vs_fly", max_tokens: int = 8,
                           cancel_event: Event | None = None, on_telemetry: Callable | None = None) -> dict:
        with self._lock:
            status = "error"
            try:
                result = self._scientific_compare(prompt, comparison, max_tokens, cancel_event, on_telemetry)
                status = "cancelled" if result.get("cancelled") else "ready"
                return result
            finally:
                self._publish("scientific_finished", on_telemetry, status=status, branch=None,
                              input_norm=0.0, mapped_input_norm=0.0, steering_vector_norm=0.0,
                              delta_relative_rms=0.0, token_index=None)

    def _scientific_compare(self, prompt: str, comparison: str, max_tokens: int,
                            cancel_event: Event | None, on_telemetry: Callable | None) -> dict:
        with self._lock:
            if comparison not in {"clean_vs_fly", "fly_vs_random"}:
                raise ValueError("Unknown scientific comparison")
            if not 1 <= max_tokens < self.config.max_context_tokens:
                raise ValueError("Scientific token limit is outside the context budget")
            cancel = cancel_event if cancel_event is not None else Event()
            initial = self.reservoir.state.copy()
            initial_time = self.reservoir.time
            def condition(random: bool = False):
                graph = random_control(self.graph, self.config.seed) if random else self.graph
                reservoir = Reservoir(graph, features=self.config.features, seed=self.config.seed)
                reservoir.state = initial.copy()
                reservoir.time = initial_time
                return self._coupling(reservoir, "random" if random else "fly")
            first_coupling = None if comparison == "clean_vs_fly" else condition()
            first = self._generate("clean" if first_coupling is None else "fly", prompt, max_tokens,
                         cancel, None, on_telemetry, session=ChatSession([]), coupling=first_coupling, scientific=True)
            if cancel.is_set() or not first["token_ids"]:
                return {"comparison": comparison, "cancelled": True, "matched_prefix": True}
            second_coupling = condition(random=comparison == "fly_vs_random")
            second = self._generate("fly" if comparison == "clean_vs_fly" else "random", prompt,
                          len(first["token_ids"]), cancel, None, on_telemetry, session=ChatSession([]),
                          coupling=second_coupling, forced_tokens=first["token_ids"], scientific=True,
                          prompt_reserve=max_tokens)
            if first["prompt_ids"] != second["prompt_ids"]:
                raise RuntimeError("Scientific comparison prompt prefixes differ")
            matched = len(second["token_ids"])
            metrics = compare_logits(first["logits"][:matched], second["logits"]) if matched else {}
            result = {"comparison": comparison, "matched_prefix": True, "matched_steps": matched,
                      "prompt_ids": first["prompt_ids"], "forced_token_ids": first["token_ids"][:matched],
                      "metrics": metrics, "cancelled": cancel.is_set(),
                      "reference_text": first["text"], "initial_state_rms": float(np.sqrt(np.mean(initial ** 2))),
                      "reference_trace": first["trace"], "condition_trace": second["trace"],
                      "interpretation": "Same-prefix logit effect; no semantic or topology advantage is implied"}
            self.recording.append("scientific_comparison", result=result)
            return result

    def demo_clean_vs_fly(self, prompt: str = "Describe a calm morning.", **kwargs: Any) -> dict:
        return self._demo_generated_pair(prompt, "clean_vs_fly", **kwargs)

    def demo_random_topology(self, prompt: str = "Describe a calm morning.", **kwargs: Any) -> dict:
        return self._demo_generated_pair(prompt, "fly_vs_random", **kwargs)

    def _demo_generated_pair(self, prompt: str, comparison: str, max_tokens: int = 8,
                             cancel_event: Event | None = None,
                             on_telemetry: Callable | None = None) -> dict:
        """Keep descriptive free continuations separate from controlled logit metrics."""
        with self._lock:
            cancel = cancel_event if cancel_event is not None else Event()
            initial = self.reservoir.state.copy()
            initial_time = self.reservoir.time
            status = "error"
            started = time.perf_counter()
            try:
                result = self.scientific_compare(prompt, comparison, max_tokens, cancel, on_telemetry)
                result["scientific_seconds"] = time.perf_counter() - started
                free = {"reference": None, "condition": None, "cancelled": cancel.is_set(),
                        "interpretation": "Descriptive independent greedy generation; metrics use the separate matched-prefix run"}
                result["free_generation"] = free
                if cancel.is_set():
                    status = "cancelled"
                    return result
                labels = ("clean", "fly") if comparison == "clean_vs_fly" else ("fly", "random")
                for role, label in zip(("reference", "condition"), labels):
                    if cancel.is_set():
                        break
                    coupling = None
                    if label != "clean":
                        graph = random_control(self.graph, self.config.seed) if label == "random" else self.graph
                        reservoir = Reservoir(graph, features=self.config.features, seed=self.config.seed)
                        reservoir.state = initial.copy()
                        reservoir.time = initial_time
                        coupling = self._coupling(reservoir, label)
                    output = self._generate(label + "_free", prompt, max_tokens, cancel, None,
                                            on_telemetry, session=ChatSession([]), coupling=coupling)
                    free[role] = {key: output[key] for key in ("prompt", "prompt_ids", "text", "token_ids",
                                  "seconds", "tokens_per_second", "cancelled")}
                    free[role]["condition"] = label
                    free[role]["initial_state_rms"] = float(np.sqrt(np.mean(initial ** 2)))
                free["cancelled"] = cancel.is_set()
                result["cancelled"] = cancel.is_set()
                result["free_generation_seconds"] = sum(output["seconds"] for output in
                                                        (free["reference"], free["condition"]) if output)
                self.recording.append("demo_free_generation", comparison=comparison, result=free)
                status = "cancelled" if cancel.is_set() else "ready"
                return result
            finally:
                self._publish("demo_finished", on_telemetry, status=status, branch=None,
                              input_norm=0.0, mapped_input_norm=0.0, steering_vector_norm=0.0,
                              delta_relative_rms=0.0, token_index=None)

    def demo_persistence(self, prompt_a: str = "Remember the word ocean.",
                         prompt_b: str = "Describe a calm morning.", max_tokens: int = 8, idle_steps: int = 10,
                         cancel_event: Event | None = None, on_telemetry: Callable | None = None) -> dict:
        with self._lock:
            status = "error"
            try:
                result = self._demo_persistence(prompt_a, prompt_b, max_tokens, idle_steps,
                                                cancel_event, on_telemetry)
                status = "cancelled" if result.get("cancelled") else "ready"
                return result
            finally:
                self._publish("scientific_finished", on_telemetry, status=status, branch=None,
                              input_norm=0.0, mapped_input_norm=0.0, steering_vector_norm=0.0,
                              delta_relative_rms=0.0, token_index=None)

    def _demo_persistence(self, prompt_a: str, prompt_b: str, max_tokens: int, idle_steps: int,
                          cancel_event: Event | None, on_telemetry: Callable | None) -> dict:
        with self._lock:
            if not 1 <= max_tokens < self.config.max_context_tokens:
                raise ValueError("Scientific token limit is outside the context budget")
            # Drive a temporary reservoir with a first prompt, then compare retained and
            # reset states on an identical second-prompt token path. Live chats stay intact.
            temp = Reservoir(self.graph, features=self.config.features, seed=self.config.seed)
            coupling = self._coupling(temp)
            cancel = cancel_event if cancel_event is not None else Event()
            self._generate("fly_warmup", prompt_a, max_tokens, cancel, None, on_telemetry,
                           session=ChatSession([]), coupling=coupling, scientific=True)
            before_idle_rms = float(np.sqrt(np.mean(temp.state ** 2)))
            temp.advance(steps=idle_steps)
            retained = temp.state.copy()
            self._publish("persistence_idle", on_telemetry, actual_reservoir=temp, status="experiment",
                          input_norm=0.0, mapped_input_norm=0.0, delta_relative_rms=0.0,
                          steering_vector_norm=0.0, token_index=None, branch="fly_idle", idle_steps=idle_steps)
            first = self._generate("fly_retained", prompt_b, max_tokens, cancel, None, on_telemetry,
                           session=ChatSession([]), coupling=coupling, scientific=True)
            temp.reset()
            if cancel.is_set() or not first["token_ids"]:
                return {"comparison": "retained_vs_reset", "cancelled": True, "matched_prefix": True}
            second = self._generate("fly_reset", prompt_b, len(first["token_ids"]), cancel, None, on_telemetry,
                           session=ChatSession([]), coupling=coupling,
                           forced_tokens=first["token_ids"], scientific=True, prompt_reserve=max_tokens)
            if first["prompt_ids"] != second["prompt_ids"]:
                raise RuntimeError("Persistence comparison prompt prefixes differ")
            n = len(second["token_ids"])
            result = {"comparison": "retained_vs_reset", "matched_prefix": True, "matched_steps": n,
                      "retained_state_rms": float(np.sqrt(np.mean(retained ** 2))),
                      "before_idle_state_rms": before_idle_rms, "idle_steps": idle_steps,
                      "prompt_a": prompt_a, "prompt_b": prompt_b,
                      "metrics": compare_logits(first["logits"][:n], second["logits"]) if n else {},
                      "forced_token_ids": first["token_ids"][:n], "cancelled": cancel.is_set()}
            self.recording.append("persistence_comparison", result=result)
            return result

    def save_recording(self, path: str | Path) -> Path:
        with self._lock:
            return self.recording.save(path, {key: session.messages for key, session in self.sessions.items()},
                        {"values": self.reservoir.state.tolist(), "simulation_time": self.reservoir.time,
                         "fingerprint": self.reservoir.fingerprint,
                         "feedback_event_count": self._feedback_events}, asdict(self.config))

    @staticmethod
    def load_recording(path: str | Path) -> OfflineReplay:
        return OfflineReplay.load(path)

    def restore_session(self, path: str | Path) -> dict:
        with self._lock:
            replay = OfflineReplay.load(path)
            if replay.metadata.get("model_revision") != self.metadata["model_revision"]:
                raise ValueError("Recording model revision does not match this engine")
            for key in ("capture_layer", "steer_layer", "features", "seed"):
                if replay.settings.get(key) != getattr(self.config, key):
                    raise ValueError(f"Recording adapter setting {key} does not match this engine")
            state = np.asarray(replay.state["values"], dtype=float)
            if replay.state["fingerprint"] != self.reservoir.fingerprint or state.shape != self.reservoir.state.shape:
                raise ValueError("Recording reservoir does not match this engine")
            if not np.isfinite(state).all() or np.max(np.abs(state)) > 1 + 1e-9:
                raise ValueError("Recording contains invalid reservoir state")
            simulation_time = float(replay.state["simulation_time"])
            if not np.isfinite(simulation_time) or simulation_time < 0:
                raise ValueError("Recording contains invalid simulation time")
            mutable = {"strength", "feedback", "steering_enabled", "feedback_enabled",
                       "max_new_tokens", "max_context_tokens"}
            self.configure(**{key: replay.settings[key] for key in mutable})
            self.reservoir.state = state.copy()
            self.reservoir.time = simulation_time
            self.sessions = {name: ChatSession(replay.messages[name]) for name in ("clean", "fly")}
            self.turn_index = sum(m.get("role") == "user" for m in replay.messages["clean"])
            self.recording.events = replay.events
            self.recording.metadata = replay.metadata
            self._feedback_events = replay.state.get("feedback_event_count",
                                        sum(event["type"] == "feedback" for event in replay.events))
            return self._publish("session_restored", status="ready", branch=None)
