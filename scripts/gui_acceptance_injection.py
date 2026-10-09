"""End-to-end GUI acceptance for live injection, the DA cue and leak control (real model, real widgets).

Drives the actual Observatory window and Continuous Generation dialog offscreen (VTK EGL):
1-9   sampled stream, Russian injection typed into the dialog during generation, provenance colors,
      no reservoir reset, POST-INJECTION, return to AUTONOMOUS after window eviction;
10    the same with the DA cue, inspecting measured DA activity;
11    leak changed with the dialog control while the run is active; classic-mode idle decay at two leaks;
12-13 Stop, Export, and consistency of the exported events, settings and states.

  QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/gui_acceptance_injection.py --output results/gui-acceptance
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
from pathlib import Path

import numpy as np
import shiboken6
from PySide6.QtCore import QObject, QSettings, QTimer, Slot
from PySide6.QtWidgets import QApplication

from liquidaion_fly.continuous import load_run
from liquidaion_fly.observatory import engine_config, load_configuration
from liquidaion_fly.observatory_gui import ObservatoryWindow

TEXT = " Из-за деревьев неожиданно появился огромный зелёный мамонт."


def check(evidence, name, condition, detail=""):
    evidence["checks"].append({"check": name, "passed": bool(condition), "detail": detail})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=700)
    parser.add_argument("--inject-at", type=int, default=300)
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
    evidence = {"checks": [], "runs": [], "errors": []}
    state = {"run": 0, "injected": False, "leak_changed": False, "categories": [], "shots": set(),
             "idle": [], "phase": "continuous"}

    def dialog():
        return window.continuous_dialog

    def start_run():
        d = dialog()
        d.output_root = args.output / "runs"
        d.total.setValue(args.tokens)
        d.window_size.setValue(128)
        d.condition.setCurrentIndex(d.condition.findData("fly"))
        d.decoding.setCurrentIndex(d.decoding.findData("sample"))
        d.temperature.setValue(1.0)
        d.top_p.setValue(0.95)
        d.eos.setCurrentIndex(d.eos.findData("suppress"))
        d.leak.set_value(0.2)
        d.cue.setCurrentIndex(d.cue.findData("da" if state["run"] == 1 else "off"))
        d.cue_strength.setValue(0.5)
        state.update(injected=False, leak_changed=False, categories=[])
        d.request_start()

    def on_progress(update):
        d = dialog()
        if not state["categories"] or state["categories"][-1] != update["category"]:
            state["categories"].append(update["category"])
        if not state["injected"] and update["step"] >= args.inject_at:
            # Steps 4-5: the operator types a Russian sentence and presses Inject during generation.
            d.inject_text.setText(TEXT)
            d.inject_button.click()
            state["injected"] = True
        name = f"run{state['run']}"
        if update["category"] == "post_injection" and f"{name}_post" not in state["shots"]:
            state["shots"].add(f"{name}_post")
            d.grab().save(str(args.output / f"{name}-post-injection.png"))
        if state["run"] == 1 and not state["leak_changed"] and update["step"] >= args.inject_at + 150:
            d.leak.spin.setValue(0.02)  # Step 11: change the leak with the control while generating.
            state["leak_changed"] = True

    class Driver(QObject):
        # Every handler is a slot of a GUI-thread QObject, so worker signals are queued to the GUI
        # thread; plain Python callables would run in the worker thread and must not touch widgets.
        @Slot(dict)
        def initialized(self, information):
            window.open_continuous()
            dialog().resize(900, 1100)
            window.worker.continuous_progress.connect(self.progress)
            QTimer.singleShot(300, start_run)

        @Slot(dict)
        def progress(self, update):
            on_progress(update)

        @Slot(str, object)
        def completed(self, command, result):
            if command == "continuous":
                finish_run(result)
            elif command == "generate" and state["phase"] == "classic":
                QTimer.singleShot(200, lambda: measure_idle(0.2))
            elif command == "generate" and state["phase"] == "classic2":
                QTimer.singleShot(200, lambda: measure_idle(0.02))

        @Slot(dict)
        def telemetry(self, snapshot):
            if state["phase"].startswith("idle") and snapshot.get("status") == "idle":
                state["idle"].append((state["phase"], snapshot["state_rms"], snapshot["simulation_time"]))

        @Slot(str)
        def error(self, message):
            evidence["errors"].append(message)
            finish()

    def finish_run(summary):
        d = dialog()
        name = f"run{state['run']}"
        d.grab().save(str(args.output / f"{name}-finished.png"))
        saved = load_run(Path(summary["output"]))
        rows = saved["telemetry"]
        record = summary["injections"][0] if summary["injections"] else {}
        start, end, left = record.get("start_position"), record.get("end_position"), record.get("left_window_position")
        visible = [c for c in state["categories"]]
        check(evidence, f"{name}: injected text inserted exactly", record.get("status") == "inserted"
              and "".join(r["fragment"] for r in rows if r["source"] == "external") == TEXT, str(record.get("status")))
        check(evidence, f"{name}: provenance sequence", [r["category"] for r in rows][0] == "autonomous"
              and {"external", "post_injection"} <= {r["category"] for r in rows}, str(visible))
        after = [r["category"] for r in rows if left is not None and r["position"] >= left]
        check(evidence, f"{name}: AUTONOMOUS after the injection left the window",
              left is not None and after and set(after) == {"autonomous"}, f"left at {left}")
        check(evidence, f"{name}: no reservoir reset at injection",
              not any(r["reset"] for r in rows) and rows[start]["state_rms"] > 0.05, f"rms {rows[start]['state_rms']:.3f}")
        html = d.text.toHtml()
        check(evidence, f"{name}: three provenance colors shown", all(c in html for c in ("#c9d3e3", "#f0a35e", "#6fa8ff")))
        # QTextDocument.toPlainText() documents that it returns non-breaking spaces as plain spaces.
        exported_text = saved["text"].replace("\u00a0", " ")
        check(evidence, f"{name}: GUI text equals exported text (U+00A0 shown as space by Qt)",
              d.text.toPlainText() == exported_text, f"{saved['text'].count(chr(0xa0))} non-breaking spaces")
        cue_rows = [r for r in rows if r["cue_applied"]]
        if state["run"] == 1:
            before = np.mean([r["da_mean"] for r in rows[start - 20:start]])
            during = np.mean([r["da_mean"] for r in cue_rows])
            check(evidence, "run1: DA cue applied on every injected-token update",
                  [r["position"] for r in cue_rows] == list(range(start + 1, end + 1)), f"{len(cue_rows)} updates")
            check(evidence, "run1: measured DA activity rises during the cue", during > before + 0.05,
                  f"before {before:+.3f}, during {during:+.3f}")
            change = next((e for e in summary["events"] if e["type"] == "leak_changed"), None)
            check(evidence, "run1: live leak change recorded and applied", change is not None
                  and abs(change["new"] - 0.02) < 1e-9 and all(abs(r["leak"] - 0.02) < 1e-9 for r in rows
                                                               if r["position"] >= change["position"]), str(change))
        else:
            check(evidence, "run0: cue off applies no cue", not cue_rows)
        exported = args.output / "exported"
        d.export_results(exported)
        copy = exported / Path(summary["output"]).name
        check(evidence, f"{name}: exported copy identical",
              all(filecmp.cmp(Path(summary["output"]) / f, copy / f, shallow=False) for f in
                  ("metadata.json", "summary.json", "token_ids.json", "telemetry.jsonl", "text.txt", "states.npz")))
        check(evidence, f"{name}: exported settings consistent",
              saved["metadata"]["config"]["cue_target"] == ("da" if state["run"] == 1 else "off")
              and len(saved["tokens"]["stream"]) == len(rows) == summary["stream_tokens"]
              and saved["states"]["events"].keys() >= {"injection1_start", "injection1_end", "injection1_left_window"})
        evidence["runs"].append({"run": name, "summary": {k: summary[k] for k in ("stop_reason", "generated_tokens",
                                 "external_tokens", "tokens_per_second", "peak_gpu_allocated_bytes")},
                                 "injection": record, "visible_categories": visible, "output": summary["output"]})
        state["run"] += 1
        if state["run"] < 2:
            QTimer.singleShot(500, start_run)
        else:
            dialog().close()
            state["phase"] = "classic"
            window.prompt.setText("Опиши зимний лес.")
            window.max_tokens.setValue(24)
            QTimer.singleShot(300, window.send_prompt)

    def measure_idle(leak):
        # Classic mode: idle reservoir decay at leak 0.2, then at 0.02 set with the main-panel control.
        state["phase"] = f"idle{leak}"
        if leak == 0.2:
            QTimer.singleShot(3000, lambda: (window.leak.spin.setValue(0.02),
                                             QTimer.singleShot(400, lambda: (window.send_prompt_text(), None))))
        else:
            QTimer.singleShot(3000, evaluate_idle)

    def send_again():
        window.prompt.setText("Опиши зимний лес.")
        window.send_prompt()
    window.send_prompt_text = lambda: (state.update(phase="classic2"), send_again())


    def evaluate_idle():
        def ratio(phase):
            values = [v for p, v, _ in state["idle"] if p == phase]
            return (values[-1] / values[0]) ** (1 / max(len(values) - 1, 1)), len(values)
        fast, n_fast = ratio("idle0.2")
        slow, n_slow = ratio("idle0.02")
        check(evidence, "classic: leak 0.02 decays more slowly per idle update than 0.2", slow > fast,
              f"per-update RMS factor {fast:.4f} at 0.2 ({n_fast} samples) vs {slow:.4f} at 0.02 ({n_slow})")
        check(evidence, "classic: engine leak changed", abs(window.worker.engine.reservoir.leak - 0.02) < 1e-12)
        window.grab().save(str(args.output / "classic-leak-0.02.png"))
        finish()

    def finish():
        evidence["passed"] = not evidence["errors"] and all(c["passed"] for c in evidence["checks"])
        (args.output / "acceptance.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n")
        window.close()
        QTimer.singleShot(100, app.quit)

    driver = Driver(window)
    window.worker.initialized.connect(driver.initialized)
    window.worker.finished.connect(driver.completed)
    window.worker.telemetry.connect(driver.telemetry)
    window.worker.error.connect(driver.error)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: (evidence["errors"].append("timeout"), finish()))
    watchdog.start(900000)
    window.show()
    app.exec()
    shiboken6.delete(window)
    for item in evidence["checks"]:
        print(("PASS " if item["passed"] else "FAIL ") + item["check"] + (f"  [{item['detail']}]" if item["detail"] else ""))
    print("errors:", evidence["errors"])
    raise SystemExit(0 if evidence.get("passed") else 1)


if __name__ == "__main__":
    main()
