"""Neural Bypass configuration: capture/steering layer pairs sharing one reservoir.

A configuration is a list of rows ``[capture, steer, enabled]``. The active set is the sorted
set of unique enabled pairs, so row order and duplicate rows never change what is executed.

- One active pair runs the legacy single-bypass coupling (llm.Coupling): unchanged ordering,
  a capture update and a feedback update per token.
- Two or more run the Synchronous Multi-Bypass coupling (llm.SynchronousCoupling): every
  steering layer reads the state from before the token, captures and feedback are averaged
  per unique layer, and the reservoir advances once per token.
"""
from __future__ import annotations

from typing import Any, Iterable

LEGACY_SINGLE = "legacy_single"
SYNCHRONOUS_MULTI = "synchronous_multi"

SYNCHRONIZATION = {
    LEGACY_SINGLE: "Legacy single bypass: hooks act in transformer layer order within the pass; capture "
                   "update and feedback update advance the reservoir separately (two updates per token "
                   "with feedback enabled, one without).",
    SYNCHRONOUS_MULTI: "Synchronous multi-bypass: all steering layers read the reservoir state from before "
                       "the token; captures (mean over unique capture layers) and feedback (mean over unique "
                       "steering layers) are applied in one reservoir update after the forward pass; "
                       "observations influence steering from the next token on.",
}


def normalize_bypasses(rows: Iterable[Any]) -> list[list]:
    """Rows as ``[capture, steer, enabled]``; accepts (c, s), (c, s, enabled) or dicts."""
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            capture, steer, enabled = row["capture"], row["steer"], row.get("enabled", True)
        else:
            row = list(row)
            if len(row) not in (2, 3):
                raise ValueError(f"A bypass is (capture, steer) or (capture, steer, enabled), got {row!r}")
            capture, steer, enabled = row[0], row[1], row[2] if len(row) == 3 else True
        if isinstance(capture, bool) or isinstance(steer, bool) or int(capture) != capture or int(steer) != steer:
            raise ValueError(f"Bypass layers must be integers, got {capture!r} -> {steer!r}")
        capture, steer = int(capture), int(steer)
        if capture < 0 or steer < 0:
            raise ValueError(f"Bypass {capture} -> {steer}: layer indices must be non-negative")
        normalized.append([capture, steer, bool(enabled)])
    return normalized


def active_bypasses(rows: Iterable[Any]) -> list[tuple[int, int]]:
    """Sorted unique enabled pairs: what actually runs, independent of row order and duplicates."""
    return sorted({(capture, steer) for capture, steer, enabled in normalize_bypasses(rows) if enabled})


def coupling_mode(active: list[tuple[int, int]]) -> str:
    return LEGACY_SINGLE if len(active) == 1 else SYNCHRONOUS_MULTI


def invalid_bypasses(rows: Iterable[Any], layers: int) -> list[list]:
    """Rows (enabled or not) that reference a layer the model does not have."""
    return [row for row in normalize_bypasses(rows) if not (row[0] < layers and row[1] < layers)]


def describe_active(active: list[tuple[int, int]]) -> str:
    return ", ".join(f"{capture}→{steer}" for capture, steer in active)
