"""Opt-in per-generation inference log capturing the exact model inputs.

Levels:
  summary - counts, stop reasons, cache/hook/trim events and SHA-256 digests only;
            no conversation text or token ids leave the process.
  full    - additionally the rendered chat template, input token ids, decoded
            input tokens and generated text. Contains the conversation verbatim.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

LEVELS = ("summary", "full")
FULL_ONLY = ("rendered_text", "input_ids", "input_tokens", "generated_ids", "generated_text",
             "latest_user_message", "history_messages")


def digest(value: str | list[int]) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else json.dumps(value).encode()
    return hashlib.sha256(data).hexdigest()


class InferenceDebugLog:
    def __init__(self, path: str | Path, level: str = "summary"):
        if level not in LEVELS:
            raise ValueError(f"Debug log level must be one of {LEVELS}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.level = level
        self._lock = Lock()

    @property
    def full(self) -> bool:
        return self.level == "full"

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {"utc": datetime.now(timezone.utc).isoformat(), "level": self.level, **record}
        if not self.full:
            record = {key: value for key, value in record.items() if key not in FULL_ONLY}
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def read_log(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
