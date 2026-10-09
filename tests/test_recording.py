"""Offline recording validation and thread-safe telemetry software tests."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from liquidaion_fly.recording import OfflineReplay, SessionRecording, load_recording
from liquidaion_fly.telemetry import Telemetry


def test_recording_roundtrip_snapshots_are_independent(tmp_path):
    recording = SessionRecording({"model_revision": "software-test"})
    state = {"values": [0.1, 0.2], "simulation_time": 1.0, "fingerprint": "test"}
    recording.append("token", snapshot={"state": [0.1, 0.2]})
    path = recording.save(tmp_path / "nested" / "session.json", {"clean": [], "fly": []}, state, {"strength": 0.25})
    replay = load_recording(path)
    replay.snapshot(0)["state"][0] = 99
    assert replay.snapshot(0)["state"][0] == 0.1
    assert replay.metadata["model_revision"] == "software-test"
    assert not path.with_name(path.name + ".tmp").exists()


def test_bounded_events_and_invalid_records(tmp_path):
    recording = SessionRecording({}, max_events=2)
    for i in range(5):
        recording.append("tick", value=i)
    assert len(recording.events) == 2 and recording.dropped_events == 3
    with pytest.raises(ValueError, match="nonfinite"):
        recording.append("bad", value=float("nan"))
    with pytest.raises(ValueError, match="Unsupported"):
        OfflineReplay({"format": "other"})


def test_telemetry_thread_safe_and_copy_isolated():
    telemetry = Telemetry(state=[0, 0])
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: telemetry.update(value=i), range(100)))
    assert telemetry.snapshot()["sequence"] == 100
    telemetry.snapshot()["state"][0] = 99
    assert telemetry.snapshot()["state"] == [0, 0]
