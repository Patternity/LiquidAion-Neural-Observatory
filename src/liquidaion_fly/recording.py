"""Portable session recordings and offline replay, requiring no model load."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = "liquidaion-observatory-v1"


def _validate_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Recording contains a nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite(item)


class SessionRecording:
    def __init__(self, metadata: dict[str, Any], max_events: int = 10000):
        self.metadata = deepcopy(metadata)
        self.metadata.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        self.events: list[dict[str, Any]] = []
        self.max_events = max_events
        self.dropped_events = 0

    def append(self, event_type: str, **data: Any) -> None:
        event = {"type": event_type, "utc": datetime.now(timezone.utc).isoformat(), **deepcopy(data)}
        _validate_finite(event)
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
            self.dropped_events += 1

    def save(self, path: str | Path, messages: dict[str, list], state: dict, settings: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": FORMAT, "metadata": self.metadata, "messages": messages,
                   "state": state, "settings": settings, "events": self.events,
                   "dropped_events": self.dropped_events}
        _validate_finite(payload)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path


class OfflineReplay:
    def __init__(self, payload: dict[str, Any]):
        if payload.get("format") != FORMAT:
            raise ValueError("Unsupported observatory recording format")
        _validate_finite(payload)
        for key in ("metadata", "messages", "state", "settings", "events"):
            if key not in payload:
                raise ValueError(f"Recording is missing {key}")
        self.metadata = deepcopy(payload["metadata"])
        self.messages = deepcopy(payload["messages"])
        self.state = deepcopy(payload["state"])
        self.settings = deepcopy(payload["settings"])
        self.events = deepcopy(payload["events"])
        self.snapshots = [deepcopy(event["snapshot"]) for event in self.events if "snapshot" in event]
        self.dropped_events = int(payload.get("dropped_events", 0))

    def snapshot(self, index: int) -> dict[str, Any]:
        return deepcopy(self.snapshots[index])

    @classmethod
    def load(cls, path: str | Path) -> "OfflineReplay":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


Recording = SessionRecording


def load_recording(path: str | Path) -> OfflineReplay:
    return OfflineReplay.load(path)
