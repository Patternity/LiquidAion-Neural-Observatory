"""Language-model loading and logit comparison (the Qwen3 adapter's loader).

Coupling policies live in coupling.py and the interface mathematics in interfaces.py; they are
re-exported here for existing imports.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .coupling import Coupling, MeasuredCoupling, SynchronousCoupling  # noqa: F401 (re-export)


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
