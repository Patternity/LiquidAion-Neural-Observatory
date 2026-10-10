"""Plugin architecture: registry, discovery, component selection and failure isolation (tiny weights)."""
import json
from pathlib import Path
import textwrap
from threading import Event

import numpy as np
import pytest
import torch
from scipy import sparse

from conftest import assert_no_hooks, build_engine
from liquidaion_fly.builtin_plugins import DEFAULT_COMPONENTS
from liquidaion_fly.connectomes import Connectome
from liquidaion_fly.continuous import ContinuousConfig, ContinuousRun
from liquidaion_fly.interfaces import OriginalNeuralInterface
from liquidaion_fly.plugin_api import (API_VERSION, ComponentSpec, Parameter, PluginError, PluginRegistry,
                                       build_registry, discover)
from liquidaion_fly.plugins import ExperimentAPI, run_experiment
from liquidaion_fly.reservoir import Reservoir


def test_builtin_components_are_registered_with_metadata():
    registry = build_registry()
    for category, component_id in DEFAULT_COMPONENTS.items():
        spec = registry.get(component_id, category)
        assert spec.plugin_id == "builtin" and spec.api_version == API_VERSION and spec.version
    assert {s.component_id for s in registry.list("coupling")} >= {
        "builtin.coupling.auto", "builtin.coupling.legacy_single", "builtin.coupling.synchronous_multi"}
    assert "builtin.connectome.random_control" in registry and "builtin.modulation.neuron_cue" in registry
    described = registry.get("builtin.reservoir.leaky_tanh").describe()
    assert described["capabilities"] == ["idle", "extra_input", "set_leak", "snapshot"]
    assert [p["name"] for p in described["parameters"]] == ["dt", "gain", "input_strength"]


@pytest.mark.parametrize("hidden, fingerprint", [
    (1024, "44cd2e997893e23ed6e06cab915645c14e190997a4628591339cfe180270a194"),
    (2048, "27c3690325aba3dfd2b2babe69a41efd20f6574f09f62c78ddce15bcf3f70719"),
    (2560, "d049749d55758ae5b7dd129e85779414c841725ac84eac00a2dde00f4e1f2bcc")])
def test_original_interface_reproduces_the_recorded_projection_fingerprints(hidden, fingerprint):
    """The fingerprints recorded by the pre-plugin engine in results/model-acceptance and GUI runs."""
    assert OriginalNeuralInterface(hidden, 512, 64, 42).fingerprint == fingerprint


def test_registry_rejects_duplicates_bad_ids_versions_and_parameters():
    registry = PluginRegistry()

    def factory(context):
        return None
    registry.register_plugin("one", lambda p: p.neural_interface("one.iface", "I", factory))
    def twice(p):
        p.neural_interface("two.iface", "I", factory)
        p.neural_interface("two.iface", "I", factory)
    with pytest.raises(PluginError, match="Duplicate"):
        registry.register_plugin("two", twice)
    assert "two.iface" not in registry, "A failing plugin commits nothing"
    with pytest.raises(PluginError, match="already loaded"):
        registry.register_plugin("one", lambda p: None)
    with pytest.raises(PluginError, match="must start with"):
        registry.register_plugin("three", lambda p: p.neural_interface("other.iface", "I", factory))
    with pytest.raises(PluginError, match="plugin API"):
        registry.register_plugin("four", lambda p: p.neural_interface("four.iface", "I", factory, api_version="2.0"))
    with pytest.raises(PluginError, match="unknown category"):
        registry.register_plugin("five", lambda p: p.add(ComponentSpec("five.x", "telepathy", "x", factory)))
    spec = registry.get("one.iface")
    with pytest.raises(PluginError, match="not registered"):
        registry.get("missing.iface", "neural_interface")
    with pytest.raises(PluginError, match="is a neural_interface"):
        registry.get("one.iface", "coupling")
    p = Parameter("gain", "float", 1.0, minimum=0.0, maximum=2.0)
    assert p.validate(1.5) == 1.5
    for bad in (3.0, "x", float("nan")):
        with pytest.raises(PluginError, match="gain"):
            p.validate(bad)
    assert Parameter("n", "layer", 3).validate(5) == 5 and Parameter("e", "enum", "a", choices=("a", "b")).validate("b")
    with pytest.raises(PluginError, match="unknown parameters"):
        spec.resolve_params({"nope": 1})


def write_plugin(root, plugin_id, body, manifest=None):
    directory = root / plugin_id
    directory.mkdir(parents=True)
    (directory / "plugin.json").write_text(json.dumps(manifest or {"id": plugin_id, "name": plugin_id,
                                                                    "version": "0.1.0", "api_version": "1.0"}))
    (directory / "plugin.py").write_text(textwrap.dedent(body))
    return directory


def test_discovery_loads_only_enabled_plugins_in_dependency_order(tmp_path):
    root = tmp_path / "plugins"
    write_plugin(root, "base_plugin", """
        from liquidaion_fly.plugin_api import Parameter
        from .helper import factory
        def register(plugin):
            plugin.neural_interface("base_plugin.iface", "Base", factory, parameters=(Parameter("k", "float", 1.0),))
    """)
    (root / "base_plugin" / "helper.py").write_text("def factory(context):\n    return context\n")
    write_plugin(root, "dependent", "def register(plugin):\n    plugin.experiment('dependent.run', 'Run', lambda api: None)\n",
                 {"id": "dependent", "name": "d", "version": "1", "api_version": "1.0", "requires": ["base_plugin"]})
    write_plugin(root, "never_imported", "raise RuntimeError('a disabled plugin must not be executed')\n")
    write_plugin(root, "broken", "def register(plugin):\n    raise ValueError('boom')\n")
    write_plugin(root, "future", "def register(plugin):\n    pass\n",
                 {"id": "future", "name": "f", "version": "1", "api_version": "9.0"})
    (root / "invalid").mkdir()
    (root / "invalid" / "plugin.json").write_text("{\"id\": \"Bad Id\"}")
    manifests, errors = discover([root])
    assert {m.plugin_id for m in manifests} == {"base_plugin", "dependent", "never_imported", "broken", "future"}
    assert any("invalid" in key for key in errors)
    registry = build_registry(enabled=["dependent", "base_plugin", "broken", "future", "orphan"], roots=[root])
    assert set(registry.plugins) == {"base_plugin", "dependent"}
    assert "dependent.run" in registry and "base_plugin.iface" in registry
    assert "boom" in registry.errors["broken"] and "plugin API" in registry.errors["future"]
    assert "not found" in registry.errors["orphan"]
    lonely = build_registry(enabled=["dependent"], roots=[root])
    assert "dependent" not in lonely.plugins and "not enabled" in lonely.errors["dependent"]


class ScaledInterface(OriginalNeuralInterface):
    """Original mathematics with the steering delta multiplied by a factor (test only)."""
    recurrent = True

    def __init__(self, *args, factor=2.0, bad_shape=False):
        super().__init__(*args)
        self.factor, self.bad_shape, self.calls = factor, bad_shape, 0

    def steer(self, h, state, strength):
        self.calls += 1
        modified, internals = super().steer(h, state, min(1.0, strength * self.factor))
        if self.bad_shape:
            return modified[:, :, :-1], internals
        internals["metrics"] = {"calls": self.calls}
        return modified, internals

    def reset(self):
        self.calls = 0


def make_registry():
    registry = build_registry()

    def register(plugin):
        plugin.neural_interface(
            "testplug.scaled", "Scaled", lambda c: ScaledInterface(c.hidden_size, c.reservoir_neurons, c.features, c.seed,
                                                                   factor=c.params["factor"], bad_shape=c.params["bad"]),
            parameters=(Parameter("factor", "float", 2.0, minimum=0.0), Parameter("bad", "bool", False)))
        plugin.connectome("testplug.ring", "Ring", lambda c: Connectome(
            "testplug.ring", sparse.csr_matrix(np.roll(np.eye(4), 1, axis=1)), ["a", "b", "c", "d"]))
        plugin.reservoir("testplug.plain", "No cue support", lambda c: PlainReservoir(c.connectome.weights, c.features,
                                                                                       c.seed))
        plugin.experiment("testplug.demo", "Demo", demo_experiment)
        plugin.add(ComponentSpec("testplug.adapter", "model_adapter", "Another adapter",
                                 lambda c: __import__("liquidaion_fly.model_adapters").model_adapters.Qwen3Adapter()))
    registry.register_plugin("testplug", register)
    return registry


class PlainReservoir(Reservoir):
    capabilities = ("idle", "set_leak", "snapshot")


def demo_experiment(api):
    api.new_session()
    api.select_components(neural_interface="testplug.scaled")
    result = api.generate("Hi", max_tokens=2)
    api.save_result("demo", result)
    api.select_components(neural_interface="builtin.interface.original")


def test_custom_interface_is_selected_validated_recorded_and_reset(monkeypatch, tmp_path):
    engine = build_engine(monkeypatch, tmp_path, registry=make_registry())
    reference = engine.generate_pair("Hello", max_new_tokens=3)["fly"]["token_ids"]
    engine.new_session()
    engine.configure(components={"neural_interface": "testplug.scaled"}, component_params={"testplug.scaled": {"factor": 3.0}})
    info = engine.model_info()
    assert info["components"]["neural_interface"]["id"] == "testplug.scaled"
    assert info["components"]["neural_interface"]["params"] == {"factor": 3.0, "bad": False}
    assert info["components"]["neural_interface"]["recurrent"] is True
    assert any(e["type"] == "components_changed" for e in engine.recording.events)
    pair = engine.generate_pair("Hello", max_new_tokens=3)
    assert pair["fly"]["trace"][-1]["interface_metrics"]["calls"] > 0
    assert engine.coupling.interface.calls > 0
    engine.new_session()
    assert engine.coupling.interface.calls == 0, "new_session resets interface state"
    engine.configure(component_params={"testplug.scaled": {"factor": 0.0}})
    zero = engine.generate_pair("Hello", max_new_tokens=3)["fly"]
    assert all(r["delta_relative_rms"] == 0 for r in zero["trace"]), "factor 0 removes steering"
    engine.configure(component_params={"testplug.scaled": {"bad": True}})
    with pytest.raises(ValueError, match="expected a tensor of shape"):
        engine.generate_pair("Hello", max_new_tokens=2)
    assert_no_hooks(engine)
    with pytest.raises(PluginError, match="unknown parameters"):
        engine.configure(component_params={"testplug.scaled": {"oops": 1}})
    engine.configure(components={"neural_interface": "builtin.interface.original"})
    engine.new_session()
    assert engine.generate_pair("Hello", max_new_tokens=3)["fly"]["token_ids"] == reference, \
        "Switching back to the original interface reproduces the original output"


def test_failed_component_change_keeps_the_running_configuration(monkeypatch, tmp_path):
    engine = build_engine(monkeypatch, tmp_path, registry=make_registry())
    before = (engine.config, engine.coupling, engine.reservoir)
    with pytest.raises(PluginError, match="not registered"):
        engine.configure(components={"neural_interface": "nobody.interface"})
    with pytest.raises(ValueError, match="switch_model"):
        engine.configure(components={"model_adapter": "testplug.adapter"})
    assert (engine.config, engine.coupling, engine.reservoir) == before
    engine.generate_pair("Hi", max_new_tokens=1)


def test_custom_connectome_and_reservoir_providers(monkeypatch, tmp_path):
    engine = build_engine(monkeypatch, tmp_path, registry=make_registry(), components={"connectome": "testplug.ring"})
    assert engine.node_ids == ["a", "b", "c", "d"] and len(engine.reservoir.state) == 4
    assert engine.metadata["connectome"]["id"] == "testplug.ring"
    pair = engine.generate_pair("Hi", max_new_tokens=2)
    assert len(pair["fly"]["token_ids"]) == 2 and len(engine.telemetry.snapshot()["state"]) == 4
    with pytest.raises(ValueError, match="no neurotransmitter labels"):
        ContinuousRun(engine, ContinuousConfig(seed_text="Once", window=8, total_tokens=2, checkpoints=(),
                                               probe_tokens=1, cue_target="da"))
    engine.configure(components={"connectome": "builtin.connectome.flywire_cx512", "reservoir": "testplug.plain"})
    assert len(engine.reservoir.state) == 3 and not engine.reservoir.state.any(), "A new reservoir starts at zero"
    with pytest.raises(PluginError, match="extra per-neuron input"):
        ContinuousRun(engine, ContinuousConfig(seed_text="Once", window=8, total_tokens=2, checkpoints=(),
                                               probe_tokens=1, cue_target="da"))
    random = engine.control_connectome()
    assert random.parameters["base"] == "builtin.connectome.flywire_cx512" and random.weights.nnz == engine.graph.nnz
    assert_no_hooks(engine)


def test_coupling_policy_selection_keeps_declared_timing(monkeypatch, tmp_path):
    engine = build_engine(monkeypatch, tmp_path, registry=make_registry())
    assert engine.coupling_info()["coupling_mode"] == "legacy_single"
    engine.configure(components={"coupling": "builtin.coupling.synchronous_multi"})
    info = engine.coupling_info()
    assert info["coupling_mode"] == "synchronous_multi" and info["reservoir_updates_per_token"] == 1
    trace = engine.generate_pair("Hi", max_new_tokens=2)["fly"]["trace"]
    assert all(r["reservoir_updates"] == 1 for r in trace)
    with pytest.raises(ValueError, match="exactly one active bypass"):
        engine.configure(components={"coupling": "builtin.coupling.legacy_single"},
                         bypasses=[[0, 1, True], [2, 0, True]])
    assert engine.config.components["coupling"] == "builtin.coupling.synchronous_multi"


def test_registered_experiment_runs_through_the_experiment_api(monkeypatch, tmp_path):
    engine = build_engine(monkeypatch, tmp_path, registry=make_registry())
    lines = []
    result = run_experiment(engine, "testplug.demo", Event(), lines.append, output_root=tmp_path / "out")
    assert result["status"] == "finished", lines
    saved = json.loads(open(result["saved"][0]).read())
    assert saved["model"]["components"]["neural_interface"]["id"] == "testplug.scaled"
    assert engine.config.components["neural_interface"] == "builtin.interface.original"
    api = ExperimentAPI(engine, tmp_path / "x.py", Event(), lines.append, output_root=tmp_path / "out")
    assert {c["id"] for c in api.list_components("neural_interface")} == {"builtin.interface.original", "testplug.scaled"}
    assert api.components()["coupling"]["id"] == "builtin.coupling.auto"
    assert_no_hooks(engine)


def test_example_plugins_run_headless_without_qt(monkeypatch, tmp_path):
    import subprocess
    import sys
    code = ("import sys; from liquidaion_fly.plugin_api import build_registry; "
            "import liquidaion_fly.observatory_engine, liquidaion_fly.plugins; "
            "r = build_registry(enabled=['example_normalized_interface', 'example_telemetry_panel', "
            "'example_research_interface']); assert not r.errors, r.errors; "
            "assert 'PySide6' not in sys.modules, 'headless use imported Qt'; print(len(r.list()))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert result.returncode == 0, result.stderr
    registry = build_registry(enabled=["example_normalized_interface"])
    engine = build_engine(monkeypatch, tmp_path, registry=registry)
    lines = []
    result = run_experiment(engine, "example_normalized_interface.compare", Event(), lines.append,
                            output_root=tmp_path / "out")
    assert result["status"] == "finished", lines
    saved = json.loads(open(result["saved"][0]).read())
    assert set(saved["result"]) == {"builtin.interface.original", "example_normalized_interface.unit_rms"}
    assert engine.config.components["neural_interface"] == "builtin.interface.original"
    assert_no_hooks(engine)
