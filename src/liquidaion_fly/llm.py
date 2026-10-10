"""Instrument genuine decoder-layer activations with a state-dependent residual."""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .reservoir import Reservoir


class Coupling:
    def __init__(self, reservoir: Reservoir, hidden_size: int, capture_layer: int,
                 steer_layer: int, strength: float = 0.25, feedback: float = 0.25,
                 mode: str = "fly", seed: int = 42):
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
        rng = np.random.default_rng(seed)
        f = reservoir.input.shape[1]
        self.encoder = torch.from_numpy(rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, f)).astype("float32"))
        self.readout = rng.normal(0, 1 / np.sqrt(len(reservoir.state)), (len(reservoir.state), f))
        self.decoder = torch.from_numpy(rng.normal(0, 1 / np.sqrt(f), (f, hidden_size)).astype("float32"))
        self.records: list[dict[str, Any]] = []
        self.states: list[np.ndarray] = []
        self.handles: list[Any] = []
        self.static_state: np.ndarray | None = None
        # Optional per-neuron input added to the capture update only (external-origin cue).
        self.cue: np.ndarray | None = None
        self.current: dict[str, Any] = {}
        self._fired: set[str] = set()
        self.token_position = 0

    @property
    def projection_fingerprint(self) -> str:
        """Identifies the fixed encoder, readout and decoder; independent of the hooked layers."""
        digest = hashlib.sha256()
        for array in (self.encoder.numpy(), self.readout, self.decoder.numpy()):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def features(self, hidden: torch.Tensor) -> np.ndarray:
        x = hidden.detach().float()
        x = x / x.square().mean().sqrt().clamp_min(1e-8)
        projection = self.encoder.to(device=x.device)
        return torch.tanh(x @ projection).cpu().numpy()

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
        self.reservoir.advance(observation, extra=self.cue)
        self._pass_record("capture").update({
            "sequence_length": h.shape[1], "capture_rms": float(h[0, -1].float().square().mean().sqrt()),
            "features": observation.tolist()})

    def steer(self, module: Any, args: tuple, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        state = self.reservoir.state.copy()
        if self.mode == "static":
            if self.static_state is None:
                self.static_state = state.copy()
            state = self.static_state
        compact = torch.from_numpy((state @ self.readout).astype("float32")).to(h.device)
        direction = compact @ self.decoder.to(h.device)
        direction = direction - direction.mean()
        direction = direction / direction.square().mean().sqrt().clamp_min(1.0)
        baseline = h[:, -1, :]
        baseline_rms = baseline.float().square().mean(dim=-1, keepdim=True).sqrt()
        alpha = 0.0 if self.mode == "none" else self.strength
        delta = (alpha * baseline_rms * direction).to(h.dtype)
        if alpha == 0:
            modified = h
        else:
            modified = h.clone()
            modified[:, -1, :] = baseline + delta
        actual_delta = modified[:, -1, :].float() - baseline.float()
        relative = float(actual_delta.square().mean().sqrt() / baseline_rms.mean().clamp_min(1e-8))
        if not torch.isfinite(modified).all():
            raise FloatingPointError("Nonfinite hidden activation after intervention")
        record = self._pass_record("steer")
        record.update({"mode": self.mode, "strength": alpha,
                             "before_rms": float(baseline_rms.mean()),
                             "after_rms": float(modified[:, -1, :].float().square().mean().sqrt()),
                             "delta_relative_rms": relative,
                             "state_rms": float(np.sqrt(np.mean(state**2))),
                             "live_reservoir_rms": float(np.sqrt(np.mean(self.reservoir.state**2))),
                             "simulation_time": self.reservoir.time})
        self.states.append(self.reservoir.state.copy())
        if self.feedback:
            # Modified activations affect later reservoir reads, never their own past.
            feedback_input = self.feedback * self.features(modified[0, -1])
            record["feedback_input_norm"] = float(np.linalg.norm(feedback_input))
            self.reservoir.advance(feedback_input)
        else:
            record["feedback_input_norm"] = 0.0
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    def attach(self, model: Any) -> None:
        if self.handles:
            raise RuntimeError("Coupling is already attached")
        layers = model.model.layers
        if max(self.capture_layer, self.steer_layer) >= len(layers):
            raise ValueError("Coupling layer is outside the model")
        self._fired = set()
        # Registration order matters when both hooks share a layer: capture reads the unsteered output.
        self.handles = [layers[self.capture_layer].register_forward_hook(self.capture),
                        layers[self.steer_layer].register_forward_hook(self.steer)]

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


class LocalLLM:
    """One frozen causal LM. Without a profile this is the original float16/eager loading path."""

    def __init__(self, model: str, device: str = "cpu", revision: str | None = None,
                 threads: int = 4, profile: Any = None):
        torch.set_num_threads(threads)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        started = time.perf_counter()
        self.device = torch.device(device)
        self.profile = profile
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        cpu = self.device.type == "cpu"
        dtype_name = "float32" if cpu else (profile.dtype if profile else "float16")
        dtype = getattr(torch, dtype_name)
        attention = profile.attention if profile else "eager"
        quantization = profile.quantization if profile else None
        if quantization:
            if quantization != "nf4":
                raise ValueError(f"Unsupported quantization {quantization!r}")
            if cpu:
                raise RuntimeError("NF4 quantization requires CUDA; this profile cannot run on CPU")
            from transformers import BitsAndBytesConfig
            config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=False)
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            # bitsandbytes places the weights itself; the model must not be moved with .to() afterwards.
            self.model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, dtype=dtype, quantization_config=config, device_map={"": index},
                trust_remote_code=False, attn_implementation=attention).eval()
            placement = set(str(v) for v in (getattr(self.model, "hf_device_map", {}) or {}).values())
            if placement - {str(index), f"cuda:{index}"}:
                raise RuntimeError(f"Quantized model was not placed entirely on GPU {index}: {placement}")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, dtype=dtype, trust_remote_code=False,
                attn_implementation=attention).to(self.device).eval()
        self.model.requires_grad_(False)
        self.model_name = model
        self.revision = revision or getattr(self.model.config, "_commit_hash", None)
        self.dtype_name = dtype_name
        self.quantization = quantization
        self.attention = attention
        self.load_seconds = time.perf_counter() - started

    def architecture(self) -> dict[str, Any]:
        """Dimensions read from the loaded model, never from profile labels."""
        config = self.model.config
        return {"model_type": config.model_type, "hidden_size": config.hidden_size,
                "num_hidden_layers": len(self.model.model.layers), "vocab_size": config.vocab_size,
                "max_position_embeddings": getattr(config, "max_position_embeddings", None),
                "eos_token_ids": sorted(self.stop_token_ids)}

    @property
    def stop_token_ids(self) -> set[int]:
        return stop_token_ids(self.tokenizer, self.model)

    def prompt_ids(self, prompt: str) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def run(self, prompt: str, coupling: Coupling | None = None, max_tokens: int = 32,
            forced_tokens: list[int] | None = None, use_cache: bool = True) -> dict[str, Any]:
        if max_tokens < 1 or (forced_tokens is not None and len(forced_tokens) == 0):
            raise ValueError("At least one generated or forced token is required")
        ids = self.prompt_ids(prompt)
        prompt_length = ids.shape[1]
        sequence = ids
        generated: list[int] = []
        logits = []
        cache = None
        limit = len(forced_tokens) if forced_tokens is not None else max_tokens
        start = time.perf_counter()
        if coupling:
            coupling.clear_trace()
            coupling.attach(self.model)
        try:
            with torch.inference_mode():
                for step in range(limit):
                    if coupling:
                        coupling.token_position = prompt_length - 1 + step
                    current = sequence if step == 0 or not use_cache else sequence[:, -1:]
                    result = self.model(input_ids=current, past_key_values=cache if use_cache else None,
                                        use_cache=use_cache)
                    distribution = result.logits[0, -1].float().cpu()
                    logits.append(distribution.numpy())
                    token = int(distribution.argmax()) if forced_tokens is None else forced_tokens[step]
                    generated.append(token)
                    sequence = torch.cat([sequence, torch.tensor([[token]], device=self.device)], dim=1)
                    cache = result.past_key_values if use_cache else None
                    if forced_tokens is None and token in self.stop_token_ids:
                        break
        finally:
            if coupling:
                coupling.detach()
        elapsed = time.perf_counter() - start
        return {"prompt": prompt, "prompt_ids": ids[0].tolist(), "token_ids": generated,
                "text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "seconds": elapsed, "tokens_per_second": len(generated) / elapsed,
                "logits": np.stack(logits), "trace": coupling.records.copy() if coupling else [],
                "states": np.stack(coupling.states) if coupling else np.empty((0, 0)),
                "cache_used": use_cache}


def stop_token_ids(tokenizer: Any, model: Any) -> set[int]:
    """End-of-turn ids: Qwen3 ends chat turns with <|im_end|> but may also emit <|endoftext|>."""
    ids: set[int] = set()
    candidates = [getattr(tokenizer, "eos_token_id", None),
                  getattr(getattr(model, "generation_config", None), "eos_token_id", None)]
    for candidate in candidates:
        if isinstance(candidate, int):
            ids.add(candidate)
        elif isinstance(candidate, (list, tuple)):
            ids.update(int(x) for x in candidate if x is not None)
    return ids


def compare_logits(baseline: np.ndarray, intervention: np.ndarray) -> dict[str, float]:
    if baseline.shape != intervention.shape:
        raise ValueError("Compare logits only on identical token contexts")
    a = torch.from_numpy(baseline).double().log_softmax(-1)
    b = torch.from_numpy(intervention).double().log_softmax(-1)
    kl = (a.exp() * (a - b)).sum(-1).numpy()
    return {"mean_kl_baseline_to_condition": float(kl.mean()),
            "first_token_kl": float(kl[0]),
            "logit_rms_difference": float(np.sqrt(np.mean((baseline - intervention)**2))),
            "argmax_disagreement_fraction": float(np.mean(baseline.argmax(-1) != intervention.argmax(-1)))}
