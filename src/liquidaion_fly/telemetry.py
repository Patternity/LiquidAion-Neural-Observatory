"""Small thread-safe copies of measured observatory state."""
from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any


class Telemetry:
    def __init__(self, **initial: Any):
        self._lock = RLock()
        self._values = dict(initial)
        self._values.setdefault("sequence", 0)

    def update(self, **values: Any) -> dict[str, Any]:
        with self._lock:
            self._values.update(deepcopy(values))
            self._values["sequence"] += 1
            return deepcopy(self._values)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._values)
