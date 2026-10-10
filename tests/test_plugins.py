"""Python plugins against a real (tiny random-weight) engine: API, results, cancellation, failures."""
import json
from pathlib import Path
from threading import Event

import pytest

from conftest import assert_no_hooks
from liquidaion_fly.plugins import load_plugin, run_plugin

EXAMPLES = Path(__file__).resolve().parents[1] / "plugins" / "examples"


def script(tmp_path, body, name="plugin.py"):
    path = tmp_path / name
    path.write_text(body)
    return path


def execute(engine, path, tmp_path, cancel=None):
    lines = []
    result = run_plugin(engine, path, cancel or Event(), lines.append, output_root=tmp_path / "out")
    return result, lines


def test_plugin_configures_generates_and_saves_through_the_engine(engine, tmp_path):
    path = script(tmp_path, '''
def run(api):
    api.new_session()
    coupling = api.set_bypasses([(0, 2), (2, 0), (0, 2)])
    assert coupling["coupling_mode"] == "synchronous_multi" and coupling["reservoir_updates_per_token"] == 1
    api.set_coupling(strength=0.5)
    result = api.generate("Hi", max_tokens=2, mode="fly")
    assert result.branch == "fly" and len(result.token_ids) == 2 and len(result.telemetry) == 2
    state = api.get_reservoir_state()
    assert state["neurons"] == 3 and not state["state"].flags.writeable
    api.save_result("fly_run", result, extra={"note": 1})
    try:
        api.save_result("fly_run", result)
    except FileExistsError:
        api.log("no overwrite")
    both = api.generate("Hi", max_tokens=1, mode="both")
    api.save_result("both", both)
    api.log(f"done {api.get_config()['settings']['strength']}")
''')
    result, lines = execute(engine, path, tmp_path)
    assert result["status"] == "finished" and "error" not in result, lines
    assert "no overwrite" in lines and "done 0.5" in lines
    saved = json.loads(Path(result["saved"][0]).read_text())
    assert saved["coupling"]["active_bypasses"] == [[0, 2], [2, 0]]
    assert saved["result"]["token_ids"] and saved["model"]["coupling_mode"] == "synchronous_multi"
    assert saved["settings"]["strength"] == 0.5 and saved["extra"] == {"note": 1}
    assert set(json.loads(Path(result["saved"][1]).read_text())["result"]) == {"clean", "fly"}
    manifest = json.loads((Path(result["output"]) / "plugin_run.json").read_text())
    assert manifest["status"] == "finished" and len(manifest["saved"]) == 2
    assert engine.config.active_bypasses == [(0, 2), (2, 0)], "Plugin and GUI share one configuration"
    assert any(e["type"] == "plugin_finished" for e in engine.recording.events)
    assert_no_hooks(engine)
    second, _ = execute(engine, path, tmp_path)
    assert second["output"] != result["output"], "A new run never reuses an earlier output directory"


def test_plugin_cancellation_and_errors_release_the_engine(engine, tmp_path):
    cancel = Event()
    path = script(tmp_path, '''
def run(api):
    for step in range(100):
        api.check_cancelled()
        api.generate("Hi", max_tokens=1, mode="clean")
''')
    cancel.set()
    result, _ = execute(engine, path, tmp_path, cancel)
    assert result["status"] == "cancelled"
    failing = script(tmp_path, '''
def run(api):
    api.set_bypasses([(0, 1), (1, 0)])
    api.generate("Hi", max_tokens=1)
    raise RuntimeError("deliberate failure")
''', "failing.py")
    result, lines = execute(engine, failing, tmp_path)
    assert result["status"] == "error" and "deliberate failure" in result["error"]
    assert any("Traceback" in line and "deliberate failure" in line for line in lines)
    assert_no_hooks(engine)
    assert engine.generate_pair("Hi", max_new_tokens=1)["fly"]["token_ids"], "The engine stays usable"
    leaking = script(tmp_path, '''
def run(api):
    api._engine.llm.model.model.layers[0].register_forward_hook(lambda *a: None)
''', "leaking.py")
    result, lines = execute(engine, leaking, tmp_path)
    assert result["status"] == "finished" and any("Removed 1 forward hook" in line for line in lines)
    assert_no_hooks(engine)
    for bad in ("x = 1\n", "def run(api):\n    raise SystemExit(3)\n", "def run(api) syntax error\n"):
        result, _ = execute(engine, script(tmp_path, bad, "bad.py"), tmp_path)
        assert result["status"] == "error"


def test_example_plugins_load():
    for name in ("bypass_sweep.py", "two_bypasses.py"):
        assert callable(load_plugin(EXAMPLES / name))
