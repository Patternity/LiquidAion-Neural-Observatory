"""GUI acceptance for model selection and coupling layers (real models, real Qt widgets, offscreen).

Sequence: Qwen3-0.6B (launch) -> 0.6B-Base, layers set to 10 -> 20 with the spin boxes, Continuous run
whose telemetry must show hooks firing exactly on 10 and 20 -> 1.7B-Base -> 4B-Base (NF4) with a run ->
back to Qwen3-0.6B and a chat turn. Each switch is made with the selector and Load button.

  QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/gui_acceptance_models.py --output results/gui-acceptance-models
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import shiboken6
from PySide6.QtCore import QObject, QSettings, QTimer, Slot
from PySide6.QtWidgets import QApplication

from liquidaion_fly.continuous import load_run
from liquidaion_fly.observatory import engine_config, load_configuration
from liquidaion_fly.observatory_gui import ObservatoryWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
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
    window.confirm_model_switch = lambda display: True
    evidence = {"checks": [], "errors": [], "memory": []}
    plan = ["qwen3-0.6b-base", "layers", "run", "qwen3-1.7b-base", "qwen3-4b-base", "run", "qwen3-0.6b", "chat"]
    state = {"step": -1}
    panel = window.model_panel

    def check(name, ok, detail=""):
        evidence["checks"].append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else "FAIL ") + name + (f" [{detail}]" if detail else ""), flush=True)

    def check_ranges(info):
        layers = info["num_hidden_layers"]
        check(f"{info['profile']}: layer ranges follow {layers} layers",
              panel.capture.maximum() == layers - 1 and panel.steer.maximum() == layers - 1
              and panel.steer.minimum() == 0 and f"0…{layers - 1}" in panel.layer_range.text(),
              f"capture 0..{panel.capture.maximum()}, steer {panel.steer.minimum()}..{panel.steer.maximum()}")
        check(f"{info['profile']}: model info shown", info["display"] in panel.model_info.text()
              and f"hidden {info['hidden_size']}" in panel.model_info.text(), panel.model_info.text().replace("\n", " | "))
        evidence["memory"].append({"profile": info["profile"], "allocated_after_load": info.get("gpu_allocated_bytes"),
                                   "dtype": info["dtype"], "quantization": info["quantization"],
                                   "load_seconds": info.get("load_seconds")})

    def advance():
        state["step"] += 1
        if state["step"] >= len(plan):
            finish()
            return
        action = plan[state["step"]]
        if action.startswith("qwen3"):
            panel.combo.setCurrentIndex(panel.combo.findData(action))
            panel.load.click()
        elif action == "layers":
            panel.capture.setValue(10)
            panel.steer.setValue(20)
        elif action == "run":
            window.open_continuous()
            dialog = window.continuous_dialog
            dialog.output_root = args.output / "runs"
            dialog.total.setValue(120)
            dialog.condition.setCurrentIndex(dialog.condition.findData("fly"))
            dialog.decoding.setCurrentIndex(dialog.decoding.findData("sample"))
            dialog.eos.setCurrentIndex(dialog.eos.findData("suppress"))
            dialog.request_start()
        elif action == "chat":
            window.prompt.setText("Привет! Кто ты?")
            window.max_tokens.setValue(32)
            window.send_prompt()

    class Driver(QObject):
        @Slot(dict)
        def initialized(self, information):
            # Offscreen VTK EGL renders take about 2 s per frame on this machine (measured before any model
            # is loaded); with 4 idle telemetry frames per second they would starve the GUI thread. The
            # standard "Pause view" control stops 3D rendering only; dynamics and telemetry continue.
            window.toggle_visualization()
            info = information["model"]
            check("launch: original Qwen3-0.6B profile", info["profile"] == "qwen3-0.6b" and
                  (info["capture_layer"], info["steer_layer"]) == (6, 14) and "Chat / Instruct" in panel.interaction.text())
            check_ranges(info)
            QTimer.singleShot(500, advance)

        @Slot(str, object)
        def completed(self, command, result):
            if isinstance(result, dict) and result.get("error"):
                evidence["errors"].append(f"{command}: {result['error']}")
            if command == "switch_model" and isinstance(result, dict) and not result.get("error"):
                check(f"switched to {result['profile']}", result["profile"] == plan[state["step"]]
                      and panel.combo.currentData() == result["profile"], result.get("display"))
                check_ranges(result)
                if result["profile"] == "qwen3-4b-base":
                    check("4B runs NF4 entirely on GPU", result["quantization"] == "nf4" and result["device"] == "cuda"
                          and "NF4" in panel.model_info.text(), panel.model_info.text().replace("\n", " | "))
                if result["kind"] == "base":
                    check(f"{result['profile']}: raw continuation mode shown", "Raw continuation" in panel.interaction.text()
                          and "Raw text" in window.prompt.placeholderText())
                window.grab().save(str(args.output / f"switched-{result['profile']}.png"))
                QTimer.singleShot(300, advance)
            elif command == "configure" and plan[state["step"]] == "layers" and result.get("capture_layer") == 10 \
                    and result.get("steer_layer") == 20:
                check("layer change applied via spin boxes", panel.state.text() == "Applied: 10 → 20", panel.state.text())
                QTimer.singleShot(300, advance)
            elif command == "continuous":
                saved = load_run(Path(result["output"]))
                info = saved["metadata"]["language_model"]
                fired = {(row["capture_layer"], row["steer_layer"]) for row in saved["telemetry"]}
                expected = (info["capture_layer"], info["steer_layer"])
                check(f"{info['profile']}: hooks fired exactly on the selected layers {expected}",
                      fired == {expected}, sorted(fired))
                check(f"{info['profile']}: run metadata records model, layers and projections",
                      info["profile"] in result["output"] and info.get("projection_fingerprint"),
                      f"{info['profile']} {expected} {str(info.get('projection_fingerprint'))[:12]}")
                window.continuous_dialog.grab().save(str(args.output / f"continuous-{info['profile']}.png"))
                window.continuous_dialog.close()
                evidence.setdefault("runs", []).append({"profile": info["profile"], "layers": expected,
                                                        "tokens_per_second": result["tokens_per_second"],
                                                        "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"]})
                QTimer.singleShot(300, advance)
            elif command == "generate" and plan[state["step"]] == "chat":
                check("back on Qwen3-0.6B: chat reply", window.model_panel.combo.currentData() == "qwen3-0.6b"
                      and result["fly"]["token_ids"], result["fly"]["text"][:60])
                QTimer.singleShot(300, advance)

        @Slot(str)
        def error(self, message):
            evidence["errors"].append(message)
            finish()

    def finish():
        evidence["passed"] = not evidence["errors"] and all(c["passed"] for c in evidence["checks"])
        (args.output / "acceptance.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n")
        window.close()
        QTimer.singleShot(100, app.quit)

    driver = Driver(window)
    window.worker.initialized.connect(driver.initialized)
    window.worker.finished.connect(driver.completed)
    window.worker.error.connect(driver.error)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: (evidence["errors"].append("timeout"), finish()))
    watchdog.start(1800000)
    window.show()
    app.exec()
    shiboken6.delete(window)
    print("errors:", evidence["errors"], "passed:", evidence.get("passed"))
    raise SystemExit(0 if evidence.get("passed") else 1)


if __name__ == "__main__":
    main()
