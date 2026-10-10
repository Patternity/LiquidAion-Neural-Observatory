"""Coupling policies: WHEN the neural interface runs inside a forward pass, and the hook plan.

A coupling policy declares a hook plan (layer index -> callable, in registration order) and
its update semantics; the kernel (install_hooks) is the only code that touches transformer
layers. The mathematics is delegated to a neural interface (interfaces.py).

Built-in policies, unchanged in behaviour:

  Coupling / MeasuredCoupling   legacy single bypass: hooks act in layer order within the
                                pass; a capture update and a feedback update (two reservoir
                                updates per token with feedback, one without)
  SynchronousCoupling           synchronous multi-bypass: one shared snapshot for all
                                steering layers, mean capture and feedback per unique layer,
                                one reservoir update after the pass
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from .interfaces import OriginalNeuralInterface, check_features, check_steered


def install_hooks(layers: Any, plan: list[tuple[int, Callable]]) -> list[Any]:
    """The kernel's single hook installer: bounds-checked, in plan order, all or nothing."""
    count = len(layers)
    for index, _ in plan:
        if not 0 <= index < count:
            raise ValueError(f"Coupling layer {index} is outside the model ({count} layers)")
    handles = []
    try:
        for index, hook in plan:
            handles.append(layers[index].register_forward_hook(hook))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles


def decoder_layers(model: Any) -> Any:
    """Transformer layers of a Hugging Face causal LM (or an explicit layer list)."""
    inner = getattr(model, "model", None)
    return inner.layers if inner is not None and hasattr(inner, "layers") else model


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


class Coupling:
    """Legacy single bypass. Builds the original interface when none is given (seeded as before)."""
    execution_mode = "legacy_single"

    def __init__(self, reservoir: Any, hidden_size: int, capture_layer: int,
                 steer_layer: int, strength: float = 0.25, feedback: float = 0.25,
                 mode: str = "fly", seed: int = 42, interface: Any = None):
        # Any pair is allowed: capture may sit below, on or above the steering layer (see _pass_record).
        if capture_layer < 0 or steer_layer < 0:
            raise ValueError("Layer indices must be non-negative")
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Strength must lie in [0, 1]")
        if not np.isfinite(feedback) or not 0 <= feedback <= 1:
            raise ValueError("Feedback must lie in [0, 1]")
        if mode not in {"none", "fly", "random", "static"}:
            raise ValueError("Unknown control mode")
        self.reservoir = reservoir
        self.capture_layer = capture_layer
        self.steer_layer = steer_layer
        self.strength = strength
        self.feedback = feedback
        self.mode = mode
        features = reservoir.input.shape[1] if hasattr(reservoir, "input") else getattr(reservoir, "input_features")
        self.interface = interface if interface is not None else \
            OriginalNeuralInterface(hidden_size, len(reservoir.state), features, seed)
        if (self.interface.reservoir_neurons, self.interface.features) != (len(reservoir.state), features):
            raise ValueError(f"Neural interface expects {self.interface.reservoir_neurons} neurons and "
                             f"{self.interface.features} input features; the reservoir has {len(reservoir.state)} "
                             f"and {features}")
        if self.interface.hidden_size != hidden_size:
            raise ValueError(f"Neural interface was built for hidden size {self.interface.hidden_size}, "
                             f"the model has {hidden_size}")
        self.features_dim = features
        self.records: list[dict[str, Any]] = []
        self.states: list[np.ndarray] = []
        self.handles: list[Any] = []
        self.static_state: np.ndarray | None = None
        # Optional per-neuron input added to the capture update only (external-origin cue).
        self.cue: np.ndarray | None = None
        self.current: dict[str, Any] = {}
        self._fired: set[str] = set()
        # Optional read-only diagnostics (steering_diagnostics.SteeringObserver); None changes nothing.
        self.observer: Any = None
        self.token_position = 0

    # Interface delegation (the original interface's matrices remain reachable for diagnostics).
    @property
    def projection_fingerprint(self) -> str:
        return self.interface.fingerprint

    @property
    def encoder(self) -> torch.Tensor:
        return self.interface.encoder

    @property
    def readout(self) -> np.ndarray:
        return self.interface.readout

    @property
    def decoder(self) -> torch.Tensor:
        return self.interface.decoder

    def features(self, hidden: torch.Tensor) -> np.ndarray:
        """Reservoir input observed from a captured hidden vector (validated interface output)."""
        return check_features(self.interface.observe(hidden), self.features_dim, "Neural interface observe()")

    def feedback_features(self, steered: torch.Tensor) -> np.ndarray:
        """Unscaled reservoir input derived from a steered hidden vector."""
        return check_features(self.interface.feedback(steered), self.features_dim, "Neural interface feedback()")

    def _pass_record(self, hook: str) -> dict[str, Any]:
        """The record of the current forward pass, opened by whichever hook fires first.

        Layers execute in index order, and hooks on one layer in registration order (capture
        first), so capture < steer, capture == steer and capture > steer all give one record and
        two reservoir updates per pass. A hook firing again means a new pass has begun.
        """
        if not self._fired or hook in self._fired:
            self._fired = set()
            self.current = {"step": len(self.records), "token_position": self.token_position,
                            "capture_layer": self.capture_layer, "steer_layer": self.steer_layer}
            self.records.append(self.current)
        self._fired.add(hook)
        return self.current

    def capture(self, module: Any, args: tuple, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        if h.ndim != 3 or h.shape[0] != 1:
            raise ValueError("This MVP supports one sequence at a time")
        observation = self.features(h[0, -1])
        if self.observer is not None:
            self.observer.capture(self.capture_layer, h[0, -1], observation)
        self.reservoir.advance(observation, extra=self.cue)
        self._pass_record("capture").update({
            "sequence_length": h.shape[1], "capture_rms": float(h[0, -1].float().square().mean().sqrt()),
            "features": observation.tolist()})

    def _steering_state(self) -> np.ndarray:
        state = self.reservoir.state.copy()
        if self.mode == "static":
            if self.static_state is None:
                self.static_state = state.copy()
            state = self.static_state
        return state

    def _intervene(self, h: torch.Tensor, state: np.ndarray,
                   layer: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the interface's steering rule at the last position; validate and measure it."""
        alpha = 0.0 if self.mode == "none" else self.strength
        modified, internals = self.interface.steer(h, state, alpha)
        modified = check_steered(h, modified, "Neural interface steer()")
        baseline = h[:, -1, :]
        baseline_rms = baseline.float().square().mean(dim=-1, keepdim=True).sqrt()
        actual_delta = modified[:, -1, :].float() - baseline.float()
        relative = float(actual_delta.square().mean().sqrt() / baseline_rms.mean().clamp_min(1e-8))
        if self.observer is not None:
            self.observer.steering(layer, state, internals or {}, baseline, modified[:, -1, :])
        measured = {"mode": self.mode, "strength": alpha, "before_rms": float(baseline_rms.mean()),
                    "after_rms": float(modified[:, -1, :].float().square().mean().sqrt()),
                    "delta_relative_rms": relative}
        metrics = (internals or {}).get("metrics")
        if metrics:
            measured["interface_metrics"] = {str(k): float(v) for k, v in metrics.items()}
        return modified, measured

    def steer(self, module: Any, args: tuple, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        state = self._steering_state()
        modified, measured = self._intervene(h, state, self.steer_layer)
        record = self._pass_record("steer")
        record.update({**measured,
                       "state_rms": float(np.sqrt(np.mean(state**2))),
                       "live_reservoir_rms": float(np.sqrt(np.mean(self.reservoir.state**2))),
                       "simulation_time": self.reservoir.time})
        self.states.append(self.reservoir.state.copy())
        if self.feedback:
            # Modified activations affect later reservoir reads, never their own past.
            feedback_input = self.feedback * self.feedback_features(modified[0, -1])
            if self.observer is not None:
                self.observer.feedback(self.steer_layer, feedback_input)
            record["feedback_input_norm"] = float(np.linalg.norm(feedback_input))
            self.reservoir.advance(feedback_input)
        else:
            record["feedback_input_norm"] = 0.0
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    def hook_plan(self) -> list[tuple[int, Callable]]:
        # Registration order matters when both hooks share a layer: capture reads the unsteered output.
        return [(self.capture_layer, self.capture), (self.steer_layer, self.steer)]

    def attach(self, model: Any) -> None:
        if self.handles:
            raise RuntimeError("Coupling is already attached")
        self._fired = set()
        self.handles = install_hooks(decoder_layers(model), self.hook_plan())

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear_trace(self) -> None:
        self.records.clear()
        self.states.clear()
        self.current = {}
        self._fired = set()
        self.static_state = None

    def reset_interface(self) -> None:
        self.interface.reset()


class MeasuredCoupling(Coupling):
    """Legacy single bypass that also records actual input and feedback (the engine's policy)."""
    def capture(self, module: Any, args: tuple, output: Any) -> None:
        super().capture(module, args, output)
        self.current["input_norm"] = float(np.linalg.norm(self.current["features"]))
        if hasattr(self.reservoir, "input"):
            mapped = self.reservoir.dynamics.input_strength * (self.reservoir.input @ np.asarray(self.current["features"]))
            self.current["mapped_input_norm"] = float(np.linalg.norm(mapped))
        else:
            self.current["mapped_input_norm"] = float("nan")

    def steer(self, module: Any, args: tuple, output: Any) -> Any:
        original = output[0] if isinstance(output, tuple) else output
        result = super().steer(module, args, output)
        modified = result[0] if isinstance(result, tuple) else result
        self.current["steering_vector_norm"] = float((modified[:, -1].float() - original[:, -1].float()).norm())
        self.current["feedback_enabled"] = bool(self.feedback)
        self.current["feedback_strength"] = self.feedback
        self.current["feedback_state_rms"] = float(np.sqrt(np.mean(self.reservoir.state ** 2)))
        return result


class SynchronousCoupling(Coupling):
    """Several bypasses on ONE shared reservoir, one aggregate reservoir update per forward pass.

    Per forward pass (one token):
      1. The first hooked layer to run opens the pass and snapshots the reservoir state x.
      2. Every steering layer adds the interface's residual computed from that same x.
      3. Every capture layer projects its (pre-steering) last-token activation to features.
      4. With feedback, every steering layer projects its steered activation to features.
      5. After the last hooked layer: u = mean(captures) + feedback * mean(feedback features),
         each mean taken over unique layers in ascending index order, and the reservoir advances
         once: x' = (1-leak) x + leak tanh(W x + input_strength U u + cue).
    Observations therefore act on steering from the next token on, for every pair direction.
    Only the sets of capture and steering layers matter: pairs with the same layer sets are
    mathematically equivalent, and row order or duplicate rows change nothing. One hook is
    registered per unique layer; a layer that both captures and steers reads its output
    before adding the residual.
    """
    execution_mode = "synchronous_multi"

    def __init__(self, reservoir: Any, hidden_size: int, bypasses: list[tuple[int, int]],
                 strength: float = 0.25, feedback: float = 0.25, mode: str = "fly", seed: int = 42,
                 interface: Any = None):
        active = sorted({(int(capture), int(steer)) for capture, steer in bypasses})
        if not active:
            raise ValueError("At least one bypass is required")
        super().__init__(reservoir, hidden_size, active[0][0], active[0][1], strength, feedback, mode, seed, interface)
        self.bypasses = active
        self.capture_layers = sorted({capture for capture, _ in active})
        self.steer_layers = sorted({steer for _, steer in active})
        self.hooked_layers = sorted(set(self.capture_layers) | set(self.steer_layers))
        self._pending: dict[str, Any] | None = None

    def _begin(self, h: torch.Tensor) -> dict[str, Any]:
        state = self._steering_state()
        record = {"step": len(self.records), "token_position": self.token_position,
                  "sequence_length": h.shape[1], "coupling_mode": self.execution_mode,
                  "bypasses": [list(pair) for pair in self.bypasses],
                  # Layers whose hooks fire in this pass (lists in synchronous mode).
                  "capture_layer": list(self.capture_layers), "steer_layer": list(self.steer_layers),
                  "capture": {}, "steering": {}}
        self.records.append(record)
        self.current = record
        self.states.append(self.reservoir.state.copy())
        self._pending = {"seen": set(), "state": state, "captures": {}, "feedback": {}, "record": record}
        return self._pending

    def _layer_hook(self, index: int, module: Any, args: tuple, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        if h.ndim != 3 or h.shape[0] != 1:
            raise ValueError("This MVP supports one sequence at a time")
        pending = self._pending
        if pending is None or index in pending["seen"]:
            pending = self._begin(h)
        pending["seen"].add(index)
        record = pending["record"]
        result = None
        if index in self.capture_layers:
            pending["captures"][index] = self.features(h[0, -1])
            if self.observer is not None:
                self.observer.capture(index, h[0, -1], pending["captures"][index])
            record["capture"][str(index)] = {"capture_rms": float(h[0, -1].float().square().mean().sqrt())}
        if index in self.steer_layers:
            modified, measured = self._intervene(h, pending["state"], index)
            measured["steering_vector_norm"] = float((modified[:, -1].float() - h[:, -1].float()).norm())
            record["steering"][str(index)] = measured
            if self.feedback:
                pending["feedback"][index] = self.feedback_features(modified[0, -1])
            result = (modified, *output[1:]) if isinstance(output, tuple) else modified
        if index == self.hooked_layers[-1]:
            self._commit(pending)
        return result

    def _commit(self, pending: dict[str, Any]) -> None:
        if set(pending["captures"]) != set(self.capture_layers):
            raise RuntimeError("A capture layer did not fire in this forward pass")
        record = pending["record"]
        captured = np.mean([pending["captures"][layer] for layer in self.capture_layers], axis=0)
        feedback_input = np.zeros_like(captured)
        if self.feedback:
            feedback_input = self.feedback * np.mean([pending["feedback"][layer] for layer in self.steer_layers], axis=0)
        mapped = self.reservoir.dynamics.input_strength * (self.reservoir.input @ captured) \
            if hasattr(self.reservoir, "input") else None
        before = self.reservoir.state.copy()
        if self.observer is not None:
            self.observer.feedback(None, feedback_input)
            self.observer.aggregate(captured, feedback_input)
        # The single update of this token: capture and feedback forcing share one tanh step.
        self.reservoir.advance(captured + feedback_input, extra=self.cue)
        steering = [record["steering"][str(layer)] for layer in self.steer_layers]
        record.update({
            "features": captured.tolist(), "input_norm": float(np.linalg.norm(captured)),
            "mapped_input_norm": float(np.linalg.norm(mapped)) if mapped is not None else float("nan"),
            "capture_rms": float(np.mean([record["capture"][str(layer)]["capture_rms"] for layer in self.capture_layers])),
            "mode": self.mode, "strength": steering[0]["strength"],
            **{key: float(np.mean([item[key] for item in steering]))
               for key in ("before_rms", "after_rms", "delta_relative_rms", "steering_vector_norm")},
            "state_rms": _rms(pending["state"]), "live_reservoir_rms": _rms(before),
            "simulation_time": self.reservoir.time,
            "feedback_enabled": bool(self.feedback), "feedback_strength": self.feedback,
            "feedback_input_norm": float(np.linalg.norm(feedback_input)),
            "feedback_state_rms": _rms(self.reservoir.state), "reservoir_updates": 1})
        if "interface_metrics" in steering[0]:
            record["interface_metrics"] = steering[0]["interface_metrics"]
        self._pending = None

    def hook_plan(self) -> list[tuple[int, Callable]]:
        return [(index, partial(self._layer_hook, index)) for index in self.hooked_layers]

    def attach(self, model: Any) -> None:
        if self.handles:
            raise RuntimeError("Coupling is already attached")
        # An interrupted pass never reached its update; it is discarded.
        self._pending = None
        self.handles = install_hooks(decoder_layers(model), self.hook_plan())

    def clear_trace(self) -> None:
        super().clear_trace()
        self._pending = None
