"""Model adapters: everything model-family-specific that the kernel needs.

The built-in Qwen3 adapter wraps the pinned profiles in models.py and the LocalLLM loader.
A different family would provide the same methods; experiment orchestration does not change.

Ownership: the kernel (ObservatoryEngine) holds the single loaded model, its KV caches and
all hooks. The adapter only loads, describes and renders; it keeps no references.
"""
from __future__ import annotations

from typing import Any, Callable

from .models import LEGACY_PROFILE, PROFILES, ModelProfile, get_profile


class Qwen3Adapter:
    family = "qwen3"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})

    @property
    def profiles(self) -> dict[str, ModelProfile]:
        return PROFILES

    @property
    def default_profile(self) -> str:
        return LEGACY_PROFILE

    def get_profile(self, key: str) -> ModelProfile:
        return get_profile(key)

    def load(self, profile: ModelProfile, source: str, device: str, threads: int, loader: Callable) -> Any:
        """Load one model through the given loader (the kernel passes LocalLLM)."""
        return loader(str(source), device=str(device), revision=profile.revision, threads=threads, profile=profile)

    @staticmethod
    def layers(model: Any) -> Any:
        """The decoder layers that coupling hooks may attach to."""
        return model.model.layers

    @staticmethod
    def hidden_size(model: Any) -> int:
        return int(model.config.hidden_size)

    def render(self, messages: list[dict[str, str]], profile: ModelProfile, tokenizer: Any) -> str:
        """Instruct: the Qwen3 chat template without thinking. Base: raw continuation of the joined texts."""
        if profile.interaction == "raw":
            # Base models: the history is one raw text, each submission and reply appended as is.
            # No chat template, role markers or instructions are added.
            return "".join(message["content"] for message in messages)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "profiles": {k: p.as_dict() for k, p in PROFILES.items()}}
