"""Render a real recording through a model-free Qt worker and verify state identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import shiboken6
from PIL import Image
from PySide6.QtCore import QObject, QSettings, QTimer, Slot
from PySide6.QtWidgets import QApplication

from liquidaion_fly.observatory import engine_config, load_configuration
from liquidaion_fly.observatory_gui import ObservatoryWindow
from liquidaion_fly.recording import load_recording


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/observatory/replay-desktop"))
    args = parser.parse_args()
    recording = load_recording(args.recording)
    snapshots = recording.snapshots
    if not snapshots:
        raise ValueError("Recording has no measured snapshots")
    peak = max(range(len(snapshots)), key=lambda index: snapshots[index]["state_rms"])
    indices = list(dict.fromkeys([0, peak, len(snapshots) // 2, len(snapshots) - 1]))
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    config = engine_config(load_configuration(Path("configs/observatory.json")))
    # Validation must not overwrite the owner's persisted GUI preferences.
    preferences = QSettings(str(args.output / "preferences.ini"), QSettings.IniFormat)
    window = ObservatoryWindow(config, "/model-not-required", Path("data/cx512"),
        Path("data/anatomy"), replay_path=args.recording, preferences=preferences)
    evidence = {"passed": False, "errors": [], "verified_indices": [], "snapshots": len(snapshots)}

    class Driver(QObject):
        @Slot(dict)
        def initialized(self, info):
            assert info["mode"] == "replay"
            assert window.worker.engine is None
            evidence["model_loaded"] = False
            QTimer.singleShot(0, self.advance)

        @Slot(str, object)
        def completed(self, command, result):
            if command != "replay_snapshot":
                return
            index = indices[len(evidence["verified_indices"])]
            np.testing.assert_array_equal(window.brain.current_state, snapshots[index]["state"])
            assert window.latest["state_rms"] == snapshots[index]["state_rms"]
            if index == len(snapshots) - 1:
                for branch in ("clean", "fly"):
                    panel = window.clean_chat if branch == "clean" else window.fly_chat
                    actual = [{"role": role, "content": text} for role, text in panel.history]
                    # Recorded assistant turns also carry stop metadata; the panels show role and text.
                    assert actual == [{"role": m["role"], "content": m["content"]} for m in recording.messages[branch]]
                evidence["final_conversations_match"] = True
            evidence["verified_indices"].append(index)
            QTimer.singleShot(0, self.advance)

        @Slot()
        def advance(self):
            if len(evidence["verified_indices"]) < len(indices):
                index = indices[len(evidence["verified_indices"])]
                window.timeline.blockSignals(True)
                window.timeline.setValue(index)
                window.timeline.blockSignals(False)
                window.timeline_label.setText(f"{index + 1} / {len(snapshots)}")
                window.command.emit("replay_snapshot", index)
            else:
                window.select_neuron(int(np.argmax(np.abs(window.brain.current_state))))
                window.export_dashboard(args.output)
                for filename in ("dashboard.png", "anatomy-state.png"):
                    with Image.open(args.output / filename) as exported:
                        exported.verify()
                evidence["exported_index"] = indices[-1]
                evidence["png_integrity_verified"] = True
                evidence["passed"] = not evidence["errors"]
                self.finish()

        @Slot(str)
        def error(self, message):
            evidence["errors"].append(message)
            self.finish()

        def finish(self):
            (args.output / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
            window.close()
            QTimer.singleShot(100, app.quit)

    driver = Driver(window)
    window.worker.initialized.connect(driver.initialized)
    window.worker.finished.connect(driver.completed)
    window.worker.error.connect(driver.error)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: driver.error("Replay validation timed out"))
    watchdog.start(60000)
    window.show()
    app.exec()
    watchdog.stop()
    shiboken6.delete(window)
    print(json.dumps(evidence, indent=2))
    raise SystemExit(0 if evidence["passed"] else 1)


if __name__ == "__main__":
    main()
