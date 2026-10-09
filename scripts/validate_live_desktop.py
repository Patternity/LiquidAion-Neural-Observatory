"""Exercise actual pretrained inference through the desktop worker and export evidence.

Run from the project root with PYTHONPATH=src. QT_QPA_PLATFORM=offscreen uses
the real VTK EGL renderer for unattended validation, not the native interactor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from PySide6.QtCore import QObject, QSettings, QTimer, Slot
from PySide6.QtWidgets import QApplication
import shiboken6
from PIL import Image

from liquidaion_fly.observatory import engine_config, load_configuration
from liquidaion_fly.observatory_gui import ObservatoryWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/observatory/live-desktop"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    config = engine_config(load_configuration(Path("configs/observatory.json"), cpu=True))
    # Validation must not overwrite the owner's persisted GUI preferences.
    preferences = QSettings(str(args.output / "preferences.ini"), QSettings.IniFormat)
    window = ObservatoryWindow(config, args.model_path, Path("data/cx512"), Path("data/anatomy"), device="cpu",
                               preferences=preferences)
    window.resize(1700, 1100)
    evidence = {"device": "cpu", "renderer": "VTK", "errors": [], "generations": [], "heartbeat_count": 0}
    start = time.monotonic()
    phase = {"number": 0, "rms": 0.0}
    heartbeat = QTimer()
    heartbeat.setInterval(50)
    heartbeat.timeout.connect(lambda: evidence.update(heartbeat_count=evidence["heartbeat_count"] + 1))
    heartbeat.start()

    def fail(message):
        evidence["errors"].append(str(message))
        finish()

    def initialized(info):
        if info["mode"] != "live":
            fail("Expected real live model")
            return
        evidence["initialization_seconds"] = time.monotonic() - start
        window.max_tokens.setValue(16)
        window.prompt.setText("Reply with exactly: Hello, world!")
        QTimer.singleShot(0, window.send_prompt)

    def idle_complete():
        evidence["idle_rms_before"] = phase["rms"]
        evidence["idle_rms_after"] = window.latest["state_rms"]
        if not 0 <= evidence["idle_rms_after"] < phase["rms"]:
            fail("Actual idle state failed to decay")
            return
        window.command.emit("save_recording", str(args.output / "session.json"))

    def completed(command, result):
        if command == "generate":
            if any(result[b].get("cancelled") or not result[b]["text"].strip() for b in ("clean", "fly")):
                fail("Generation produced no complete branch evidence")
                return
            evidence["generations"].append({b: {k: result[b][k] for k in ("text", "seconds", "tokens_per_second", "cache_reused", "token_ids")} for b in ("clean", "fly")})
            phase["number"] += 1
            if phase["number"] == 1:
                window.prompt.setText("Now reply with exactly: Goodbye!")
                QTimer.singleShot(0, window.send_prompt)
            else:
                evidence["model_instances"] = 1
                evidence["peak_state_rms"] = window.latest["state_rms"]
                window.export_dashboard(args.output)
                for filename in ("dashboard.png", "anatomy-state.png"):
                    with Image.open(args.output / filename) as exported:
                        exported.verify()
                evidence["png_integrity_verified"] = True
                phase["rms"] = window.latest["state_rms"]
                QTimer.singleShot(1000, idle_complete)
        elif command == "save_recording":
            window.command.emit("reset_state", None)
        elif command == "reset_state":
            evidence["reset_state_rms"] = window.latest["state_rms"]
            evidence["reset_max_alpha"] = int(window.brain.points["activity_rgba"][:, 3].max())
            if evidence["reset_state_rms"] != 0 or evidence["reset_max_alpha"] != 0:
                fail("Reset did not produce a zero state and transparent activity")
                return
            finish()

    def finish():
        heartbeat.stop()
        evidence["duration_seconds"] = time.monotonic() - start
        evidence["passed"] = not evidence["errors"] and len(evidence["generations"]) == 2 and evidence.get("reset_max_alpha") == 0
        (args.output / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
        window.close()
        QTimer.singleShot(100, app.quit)

    class Driver(QObject):
        # QObject receiver affinity queues these callbacks onto the GUI thread.
        @Slot(dict)
        def initialized(self, information):
            initialized(information)

        @Slot(str, object)
        def completed(self, command, result):
            completed(command, result)

        @Slot(str)
        def error(self, message):
            fail(message)

    driver = Driver(window)
    window.worker.initialized.connect(driver.initialized)
    window.worker.finished.connect(driver.completed)
    window.worker.error.connect(driver.error)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: fail("Desktop validation timed out"))
    watchdog.start(180000)
    window.show()
    app.exec()
    watchdog.stop()
    shiboken6.delete(window)
    print(json.dumps(evidence, indent=2))
    raise SystemExit(0 if evidence.get("passed") else 1)


if __name__ == "__main__":
    main()
