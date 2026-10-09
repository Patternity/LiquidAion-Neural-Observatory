"""Capture real screenshots and an activity animation from a live Observatory session.

Every image comes from the actual application: the real model generates both
branches, the reservoir state is the measured state, and the animation replays
snapshots recorded during that same session. Nothing is drawn synthetically.

Run from the project root (QT_QPA_PLATFORM=offscreen uses the VTK EGL renderer):
  QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/capture_publication_assets.py \
      --output publication/hero --prompt "Describe a calm morning in two sentences."
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import shiboken6
from PIL import Image
from PySide6.QtCore import QObject, QSettings, QTimer, Slot
from PySide6.QtWidgets import QApplication

from liquidaion_fly.observatory import engine_config, load_configuration
from liquidaion_fly.observatory_gui import ObservatoryWindow
from liquidaion_fly.recording import load_recording


def animation(window, recording_path: Path, output: Path, frames: int, width: int) -> dict:
    """Render recorded Fly-branch and idle snapshots through the real anatomy view."""
    recording = load_recording(recording_path)
    events = [event for event in recording.events if "snapshot" in event]
    first_fly = next((i for i, event in enumerate(events) if event["snapshot"].get("branch") == "fly"), None)
    if first_fly is None:
        raise RuntimeError("Recording contains no Fly snapshots")
    # Fly-branch token snapshots and the idle decay that follows them; Clean leaves the reservoir untouched.
    chosen = [event for event in events[first_fly:] if event["snapshot"].get("branch") == "fly"
              or event["type"] == "idle_tick"]
    indices = np.linspace(0, len(chosen) - 1, min(frames, len(chosen))).round().astype(int)
    brain = window.brain
    images = []
    for index in dict.fromkeys(indices.tolist()):
        snapshot = chosen[index]["snapshot"]
        label = (f"Fly branch · token {snapshot.get('token_index')}" if snapshot.get("branch") == "fly"
                 else "Idle dynamics (no input)")
        brain.plotter.add_text(f"{label}\nstate RMS {snapshot['state_rms']:.4f} · t = {snapshot['simulation_time']:.1f}",
                               position="lower_left", font_size=9, color="#c9d6ee", name="frame_label")
        brain.update_state(snapshot["state"])
        pixels = brain.plotter.screenshot(return_img=True)
        image = Image.fromarray(pixels)
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        images.append(image.convert("P", palette=Image.ADAPTIVE, colors=128))
    brain.plotter.remove_actor("frame_label")
    images[0].save(output / "activity.gif", save_all=True, append_images=images[1:], duration=90, loop=0,
                   optimize=True)
    return {"frames": len(images), "source_snapshots": len(chosen), "recording": recording_path.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True, help="Repeat for several turns")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--feedback", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--size", default="1920x1200")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        raise SystemExit("Set QT_QPA_PLATFORM=offscreen so the capture is reproducible and unattended")
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    config = load_configuration(Path("configs/observatory.json"), cpu=args.cpu)
    preferences = QSettings(str(args.output / "preferences.ini"), QSettings.IniFormat)
    window = ObservatoryWindow(engine_config(config), config["model"], Path("data/cx512"), Path("data/anatomy"),
                               device=config["device"], preferences=preferences)
    width, height = (int(x) for x in args.size.split("x"))
    window.resize(width, height)
    evidence = {"device": config["device"], "strength": args.strength, "feedback": args.feedback,
                "max_new_tokens": args.max_new_tokens, "prompts": args.prompt, "turns": [], "errors": []}
    state = {"turn": 0}

    def finish():
        evidence["passed"] = not evidence["errors"]
        (args.output / "capture.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        window.close()
        QTimer.singleShot(100, app.quit)

    def send_next():
        window.prompt.setText(args.prompt[state["turn"]])
        window.send_prompt()

    class Driver(QObject):
        @Slot(dict)
        def initialized(self, information):
            window.strength.setValue(args.strength)
            window.feedback_strength.setValue(args.feedback)
            window.max_tokens.setValue(args.max_new_tokens)
            QTimer.singleShot(500, send_next)

        @Slot(str, object)
        def completed(self, command, result):
            if command == "generate":
                evidence["turns"].append({branch: {key: result[branch].get(key) for key in
                                          ("text", "stop_reason", "token_ids", "seconds", "tokens_per_second")}
                                          for branch in ("clean", "fly")})
                state["turn"] += 1
                if state["turn"] < len(args.prompt):
                    QTimer.singleShot(200, send_next)
                else:
                    # Let real idle dynamics run so the recording contains decay frames.
                    QTimer.singleShot(int(args.idle_seconds * 1000), capture)
            elif command == "save_recording":
                evidence["animation"] = animation(window, args.output / "session.json", args.output,
                                                  args.frames, args.gif_width)
                finish()

        @Slot(str)
        def error(self, message):
            evidence["errors"].append(message)
            finish()

    def capture():
        window.export_dashboard(args.output)
        window.command.emit("save_recording", str(args.output / "session.json"))

    driver = Driver(window)
    window.worker.initialized.connect(driver.initialized)
    window.worker.finished.connect(driver.completed)
    window.worker.error.connect(driver.error)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: (evidence["errors"].append("Capture timed out"), finish()))
    watchdog.start(600000)
    started = time.monotonic()
    window.show()
    app.exec()
    evidence["seconds"] = time.monotonic() - started
    shiboken6.delete(window)
    print(json.dumps({key: evidence[key] for key in ("passed", "errors", "seconds")}))
    raise SystemExit(0 if evidence.get("passed") else 1)


if __name__ == "__main__":
    main()
