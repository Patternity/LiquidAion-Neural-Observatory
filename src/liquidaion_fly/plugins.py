"""Trusted local Python experiment scripts run against the live Observatory engine.

A plugin is an ordinary ``.py`` file defining ``run(api)``. It is executed in the engine
worker thread with the user's full operating-system permissions: there is no sandbox. The
``ExperimentAPI`` is a thin wrapper over the same engine methods the GUI uses.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "results" / "plugins"
COUPLING_SETTINGS = {"strength", "feedback", "steering_enabled", "feedback_enabled", "leak_rate"}
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PluginCancelled(Exception):
    """Raised by ``api.check_cancelled()`` after Stop was requested."""


@dataclass
class GenerationResult:
    """One generated turn on one branch, as produced by the engine."""
    branch: str
    prompt: str
    text: str
    token_ids: list[int]
    prompt_ids: list[int]
    stop_reason: str
    seconds: float
    tokens_per_second: float
    cancelled: bool
    decoding: dict[str, Any]
    coupling: dict[str, Any]
    model: dict[str, Any]
    # Per-token coupling records (Fly only): norms, RMS values, layers, update counts.
    telemetry: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Cannot save {type(value).__name__} in a result")


class ExperimentAPI:
    """What a plugin's ``run(api)`` receives. All calls run on the engine worker thread."""

    def __init__(self, engine: Any, script: Path, cancel_event: Event | None = None,
                 log: Callable[[str], None] | None = None, on_telemetry: Callable | None = None,
                 output_root: Path = DEFAULT_OUTPUT_ROOT):
        self._engine = engine
        self._cancel = cancel_event if cancel_event is not None else Event()
        self._log = log or print
        self._on_telemetry = on_telemetry
        self.script = Path(script)
        self._output_root = Path(output_root)
        self._output_dir: Path | None = None
        self.sessions_touched = False
        self.saved: list[str] = []

    # ------------------------------------------------------------ status

    def log(self, message: Any) -> None:
        """Show a line in the Observatory's plugin log."""
        self._log(str(message))

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        """Raise PluginCancelled if Stop was requested; call it between experiment steps."""
        if self._cancel.is_set():
            raise PluginCancelled("Stop requested")

    # ------------------------------------------------------------ configuration

    def set_bypasses(self, bypasses: list) -> dict[str, Any]:
        """Replace the bypass list: [(capture, steer), ...] or [(capture, steer, enabled), ...].

        Same validation and application path as the MODEL COUPLING panel. Returns the
        executed configuration (active pairs, coupling mode, update count per token).
        """
        self._engine.configure(bypasses=[list(row) for row in bypasses])
        return self._engine.coupling_info()

    def set_coupling(self, **settings: Any) -> dict[str, Any]:
        """Set strength, feedback (0..1), steering_enabled, feedback_enabled or leak_rate."""
        unknown = set(settings) - COUPLING_SETTINGS
        if unknown:
            raise ValueError(f"Unknown coupling settings {sorted(unknown)}; allowed: {sorted(COUPLING_SETTINGS)}")
        self._engine.configure(**settings)
        return self.get_config()

    # ------------------------------------------------------------ reset

    def new_session(self) -> None:
        """Clean start without reloading weights: empty histories and KV caches, zero reservoir,
        cleared coupling traces and counters, a new session recording."""
        self.sessions_touched = True
        self._engine.new_session()

    def reset_state(self) -> None:
        """Zero the reservoir and drop the Fly KV cache; both histories are kept."""
        self._engine.reset_state()

    # ------------------------------------------------------------ generation

    def generate(self, prompt: str, mode: str = "fly", max_tokens: int | None = None,
                 decoding: str = "greedy") -> GenerationResult | dict[str, GenerationResult]:
        """Generate one turn with the Observatory's chat path (greedy, as in the GUI).

        mode: "fly", "clean" or "both" (both branches, like Send to both; returns a dict).
        max_tokens: None uses the current Max new setting. The turn is appended to that
        branch's history; call new_session() for independent runs. For sampling use
        run_continuous(decoding="sample", ...).
        """
        if decoding != "greedy":
            raise ValueError("The chat path decodes greedily; use run_continuous(decoding='sample', ...) for sampling")
        self.check_cancelled()
        self.sessions_touched = True
        if mode == "both":
            pair = self._engine.generate_pair(prompt, max_new_tokens=max_tokens, cancel_event=self._cancel,
                                              on_telemetry=self._on_telemetry)
            return {branch: self._result(branch, pair[branch]) for branch in ("clean", "fly")}
        if mode not in {"clean", "fly"}:
            raise ValueError("mode must be 'fly', 'clean' or 'both'")
        output = self._engine.generate_branch(mode, prompt, max_new_tokens=max_tokens, cancel_event=self._cancel,
                                              on_telemetry=self._on_telemetry)
        return self._result(mode, output)

    def _result(self, branch: str, output: dict[str, Any]) -> GenerationResult:
        return GenerationResult(
            branch=branch, prompt=output.get("prompt", ""), text=output.get("text", ""),
            token_ids=list(output.get("token_ids", [])), prompt_ids=list(output.get("prompt_ids", [])),
            stop_reason=output.get("stop_reason", ""), seconds=float(output.get("seconds", 0.0)),
            tokens_per_second=float(output.get("tokens_per_second", 0.0)), cancelled=bool(output.get("cancelled")),
            decoding={"method": "greedy", "max_new_tokens": output.get("max_new_tokens"),
                      "repetition_penalty": None},
            coupling=self._engine.coupling_info() if branch == "fly" else {"coupling_mode": "none (Clean)"},
            model=self._model(), telemetry=[dict(record) for record in output.get("trace", [])])

    def run_continuous(self, name: str | None = None, **config: Any) -> dict[str, Any]:
        """One Continuous Generation condition (sliding window; greedy or seeded sampling).

        Keyword arguments are ContinuousConfig fields, e.g. seed_text, total_tokens, window,
        condition ("clean", "fly", "fly_reset", "random"), decoding ("greedy" or "sample"),
        temperature, top_p, sampling_seed. Output goes to this plugin run's directory.
        Returns the run summary with "output" set to that directory.
        """
        from .continuous import ContinuousConfig, ContinuousRun
        self.check_cancelled()
        directory = self._unique(self.output_dir / self._name(name or f"continuous-{config.get('condition', 'fly')}"))
        run = ContinuousRun(self._engine, ContinuousConfig(**config), directory)
        summary = run.run(self._cancel)
        return {**summary, "output": str(directory)}

    # ------------------------------------------------------------ inspection

    def get_config(self) -> dict[str, Any]:
        """Current engine settings, model and executed coupling (a copy)."""
        return json.loads(json.dumps({"settings": asdict(self._engine.config), "model": self._model(),
                                      "coupling": self._engine.coupling_info()}, default=_json_default))

    def get_reservoir_state(self) -> dict[str, Any]:
        """A copy of the shared reservoir: state vector, simulation time, leak and fingerprint."""
        reservoir = self._engine.reservoir
        state = reservoir.state.copy()
        state.setflags(write=False)
        return {"state": state, "rms": float(np.sqrt(np.mean(state ** 2))), "simulation_time": reservoir.time,
                "leak": reservoir.leak, "fingerprint": reservoir.fingerprint, "neurons": len(state)}

    def _model(self) -> dict[str, Any]:
        info = self._engine.model_info()
        return {key: value for key, value in info.items() if not key.startswith("gpu_")}

    # ------------------------------------------------------------ results

    @property
    def output_dir(self) -> Path:
        """results/plugins/<script name>/<start time>/, created on first use, never reused."""
        if self._output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._output_dir = self._unique(self._output_root / self._name(self.script.stem) / stamp)
            self._output_dir.mkdir(parents=True)
        return self._output_dir

    def save_result(self, name: str, result: Any, extra: dict[str, Any] | None = None,
                    overwrite: bool = False) -> Path:
        """Write <output_dir>/<name>.json with the result and the full scientific context.

        result: a GenerationResult, a dict of them, or any JSON-serializable data.
        Existing files are never replaced unless overwrite=True.
        """
        path = self.output_dir / f"{self._name(name)}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        reservoir = self.get_reservoir_state()
        document = {"name": name, "saved_utc": datetime.now(timezone.utc).isoformat(), "script": str(self.script),
                    "result": result, "model": self._model(), "coupling": self._engine.coupling_info(),
                    "settings": asdict(self._engine.config),
                    "reservoir": {key: reservoir[key] for key in ("rms", "simulation_time", "leak", "fingerprint")},
                    "session_id": self._engine.session_id, "extra": extra or {}}
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2, default=_json_default) + "\n",
                        encoding="utf-8")
        self.saved.append(str(path))
        return path

    @staticmethod
    def _name(name: str) -> str:
        if not NAME.match(name):
            raise ValueError(f"Result name {name!r}: use letters, digits, '.', '_' or '-'")
        return name

    @staticmethod
    def _unique(path: Path) -> Path:
        candidate, index = path, 1
        while candidate.exists():
            candidate = path.with_name(f"{path.name}-{index}")
            index += 1
        return candidate


def load_plugin(path: str | Path) -> Callable[[ExperimentAPI], Any]:
    """Import a script file with importlib and return its run(api) function."""
    path = Path(path)
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"{path} is not a Python script")
    spec = importlib.util.spec_from_file_location(f"liquidaion_plugin_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError(f"{path.name} must define run(api)")
    return run


def run_plugin(engine: Any, path: str | Path, cancel_event: Event, log: Callable[[str], None],
               on_telemetry: Callable | None = None, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Execute one plugin and return its status; a script error is logged with its traceback and
    returned as status "error" (with "error" set), leaving the engine usable.

    Engine hooks are always detached by the engine itself; any forward hook still present
    afterwards was left by the script and is removed so later inference stays uncoupled.
    """
    path = Path(path).resolve()
    api = ExperimentAPI(engine, path, cancel_event, log, on_telemetry, output_root)
    started = datetime.now(timezone.utc).isoformat()
    status, error = "finished", None
    engine.recording.append("plugin_started", script=str(path),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    log(f"Running {path.name}")
    try:
        load_plugin(path)(api)
        if cancel_event.is_set():
            status = "cancelled"
    except PluginCancelled:
        status = "cancelled"
    except (Exception, SystemExit) as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        log(traceback.format_exc().rstrip())
    finally:
        leftover = engine._hook_count()
        if leftover:
            for layer in engine.llm.model.model.layers:
                layer._forward_hooks.clear()
            log(f"Removed {leftover} forward hook(s) left by the script")
        engine.recording.append("plugin_finished", script=str(path), status=status, error=error,
                                saved=api.saved)
        if api._output_dir is not None:
            (api._output_dir / "plugin_run.json").write_text(json.dumps(
                {"script": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "started_utc": started,
                 "finished_utc": datetime.now(timezone.utc).isoformat(), "status": status, "error": error,
                 "saved": api.saved}, ensure_ascii=False, indent=2) + "\n")
        log(f"{path.name}: {status}" + (f" ({error})" if error else ""))
    result = {"status": status, "script": str(path), "saved": api.saved,
            "output": str(api._output_dir) if api._output_dir else None, "sessions_touched": api.sessions_touched,
            "model": engine.model_info()}
    if error:
        result["error"] = f"{path.name} failed: {error} (traceback in the plugin log)"
    return result
