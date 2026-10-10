"""Registry of selectable Qwen3 language substrates.

A profile states what to load and how; tensor dimensions are never taken from here but
read from the loaded model's configuration. Revisions are immutable Hugging Face commits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_id: str
    display: str
    kind: str                # "instruct" (chat-tuned) or "base" (pretrained causal LM)
    revision: str
    dtype: str               # CUDA weight/compute dtype; CPU always uses float32
    quantization: str | None  # None or "nf4" (bitsandbytes, CUDA only)
    attention: str           # attention implementation passed to transformers
    capture_layer: int
    steer_layer: int

    @property
    def interaction(self) -> str:
        """Chat protocol only for instruction-tuned models; Base models continue raw text."""
        return "chat" if self.kind == "instruct" else "raw"

    def as_dict(self) -> dict:
        return {**asdict(self), "interaction": self.interaction}


LEGACY_PROFILE = "qwen3-0.6b"

# Defaults: the established 6 -> 14 pair for the 28-layer models; for the 36-layer 4B model the
# same relative depths (6/28, 14/28) rounded to 8 -> 18. These are starting points, not tuned values.
PROFILES: dict[str, ModelProfile] = {profile.key: profile for profile in (
    # The published baseline: exactly the original loading path (float16 + eager attention).
    ModelProfile("qwen3-0.6b", "Qwen/Qwen3-0.6B", "Qwen3-0.6B (Instruct)", "instruct",
                 "c1899de289a04d12100db370d81485cdf75e47ca", "float16", None, "eager", 6, 14),
    ModelProfile("qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base", "base",
                 "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "bfloat16", None, "sdpa", 6, 14),
    ModelProfile("qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base", "base",
                 "ea980cb0a6c2ae4b936e82123acc929f1cec04c1", "bfloat16", None, "sdpa", 6, 14),
    ModelProfile("qwen3-4b-base", "Qwen/Qwen3-4B-Base", "Qwen3-4B-Base", "base",
                 "906bfd4b4dc7f14ee4320094d8b41684abff8539", "bfloat16", "nf4", "sdpa", 8, 18),
)}


def get_profile(key: str) -> ModelProfile:
    if key not in PROFILES:
        raise ValueError(f"Unknown model profile {key!r}; available: {', '.join(PROFILES)}")
    return PROFILES[key]


def valid_layer_pair(capture: int, steer: int, layers: int) -> bool:
    """Zero-based: capture in [0, N-2], steering in [capture+1, N-1]."""
    return 0 <= capture <= layers - 2 and capture < steer <= layers - 1
