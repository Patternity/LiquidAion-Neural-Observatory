"""Screenshot the continuous-generation window during a real run (offscreen VTK EGL rendering).

  QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/capture_continuous_ui.py \
      --output publication/continuous/ui --tokens 1000 --strength 0.70 --feedback 0.45
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import shiboken6
from PySide6.QtCore import QObject, QSettings, QTimer, Slot
from PySide6.QtWidgets import QApplication

from liquidaion_fly.observatory import engine_config, load_configuration
from liquidaion_fly.observatory_gui import ObservatoryWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--condition", default="fly")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--feedback", type=float, default=0.25)
    parser.add_argument("--eos", default="suppress")
    parser.add_argument("--prompt-policy", default="raw")
    args = parser.parse_args()
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        raise SystemExit("Set QT_QPA_PLATFORM=offscreen")
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    config = load_configuration(Path("configs/observatory.json"))
    window = ObservatoryWindow(engine_config(config), config["model"], Path("data/cx512"), Path("data/anatomy"),
                               device=config["device"],
                               preferences=QSettings(str(args.output / "preferences.ini"), QSettings.IniFormat))
    window.resize(1920, 1200)
    evidence = {"errors": []}

    class Driver(QObject):
        @Slot(dict)
        def initialized(self, information):
            window.open_continuous()
            dialog = window.continuous_dialog
            dialog.resize(900, 1000)
            dialog.output_root = args.output / "run"
            dialog.window_size.setValue(args.window)
            dialog.total.setValue(args.tokens)
            dialog.condition.setCurrentIndex(dialog.condition.findData(args.condition))
            dialog.strength.setValue(args.strength)
            dialog.feedback.setValue(args.feedback)
            dialog.eos.setCurrentIndex(dialog.eos.findData(args.eos))
            dialog.prompt_policy.setCurrentIndex(dialog.prompt_policy.findData(args.prompt_policy))
            QTimer.singleShot(300, dialog.request_start)

        @Slot(str, object)
        def completed(self, command, result):
            if command != "continuous":
                return
            evidence["summary"] = {key: result.get(key) for key in
                                   ("stop_reason", "generated_tokens", "tokens_per_second", "output")}
            QTimer.singleShot(500, capture)

        @Slot(str)
        def error(self, message):
            evidence["errors"].append(message)
            finish()

    def capture():
        app.processEvents()
        window.grab().save(str(args.output / "observatory.png"))
        window.continuous_dialog.grab().save(str(args.output / "continuous-dialog.png"))
        finish()

    def finish():
        evidence["passed"] = not evidence["errors"]
        (args.output / "capture.json").write_text(json.dumps({**evidence, **vars(args), "output": str(args.output)},
                                                             ensure_ascii=False, indent=2, default=str) + "\n")
        window.continuous_dialog.close() if window.continuous_dialog else None
        window.close()
        QTimer.singleShot(100, app.quit)

    driver = Driver(window)
    window.worker.initialized.connect(driver.initialized)
    window.worker.finished.connect(driver.completed)
    window.worker.error.connect(driver.error)
    window.show()
    app.exec()
    shiboken6.delete(window)
    print(json.dumps(evidence, ensure_ascii=False))
    raise SystemExit(0 if evidence.get("passed") else 1)


if __name__ == "__main__":
    main()
