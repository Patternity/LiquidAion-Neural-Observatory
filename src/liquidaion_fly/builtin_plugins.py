"""Built-in components: the reference LiquidAion configuration, registered like any plugin.

Factories receive a context object; the fields per category are documented in docs/PLUGINS.md.
"""
from __future__ import annotations

from typing import Any

from .connectomes import cue_group, flywire_cx512, random_control_connectome
from .coupling import MeasuredCoupling, SynchronousCoupling
from .interfaces import OriginalNeuralInterface
from .model_adapters import Qwen3Adapter
from .plugin_api import ComponentSpec, Parameter, PluginRegistrar
from .reservoir import Dynamics, Reservoir

DEFAULT_COMPONENTS = {
    "model_adapter": "builtin.model.qwen3",
    "connectome": "builtin.connectome.flywire_cx512",
    "reservoir": "builtin.reservoir.leaky_tanh",
    "neural_interface": "builtin.interface.original",
    "coupling": "builtin.coupling.auto",
}
LEGACY_SEMANTICS = {"mode": "legacy_single", "updates_per_token": "2 with feedback (capture, feedback); 1 without",
                    "timing": "hooks in transformer layer order within the pass; capture updates the reservoir "
                              "immediately; steering reads the current state"}
SYNCHRONOUS_SEMANTICS = {"mode": "synchronous_multi", "updates_per_token": "1",
                         "timing": "all steering layers read the pre-token snapshot; captures and feedback are "
                                   "averaged per unique layer and applied in one update after the pass "
                                   "(next-token coupling)"}


def leaky_tanh_reservoir(context: Any) -> Reservoir:
    p = context.params
    reservoir = Reservoir(context.connectome.weights, features=context.features, seed=context.seed,
                          dynamics=Dynamics(dt=p["dt"], gain=p["gain"], input_strength=p["input_strength"]))
    reservoir.set_leak(context.leak)
    return reservoir


def original_interface(context: Any) -> OriginalNeuralInterface:
    return OriginalNeuralInterface(context.hidden_size, context.reservoir_neurons, context.features, context.seed)


def _coupling(policy: str):
    def factory(context: Any) -> Any:
        active = list(context.bypasses)
        legacy = policy == "legacy" or (policy == "auto" and len(active) == 1)
        if legacy:
            if len(active) != 1:
                raise ValueError(f"builtin.coupling.legacy_single runs exactly one active bypass; {len(active)} are "
                                 "enabled. Use builtin.coupling.synchronous_multi or builtin.coupling.auto.")
            capture, steer = active[0]
            return MeasuredCoupling(context.reservoir, context.hidden_size, capture, steer, context.strength,
                                    context.feedback, mode=context.mode, seed=context.seed, interface=context.interface)
        return SynchronousCoupling(context.reservoir, context.hidden_size, active, context.strength, context.feedback,
                                   mode=context.mode, seed=context.seed, interface=context.interface)
    return factory


def coupling_semantics(component_id: str, active_pairs: int) -> dict[str, str]:
    if component_id == "builtin.coupling.legacy_single" or (component_id == "builtin.coupling.auto" and active_pairs == 1):
        return LEGACY_SEMANTICS
    return SYNCHRONOUS_SEMANTICS


def register(plugin: PluginRegistrar) -> None:
    plugin.add(ComponentSpec(
        "builtin.model.qwen3", "model_adapter", "Qwen3 (pinned profiles)", lambda context: Qwen3Adapter(context.params),
        description="Qwen3-0.6B Instruct, 0.6B/1.7B Base and 4B Base (NF4) at pinned revisions; chat template for "
                    "Instruct, raw continuation for Base.", capabilities=("chat", "raw_continuation", "kv_cache")))
    plugin.connectome(
        "builtin.connectome.flywire_cx512", "FlyWire CX512", flywire_cx512,
        description="512 central-complex neurons, FlyWire FAFB materialization 783 (data/cx512).",
        capabilities=("nt_types", "provenance"),
        parameters=(Parameter("directory", "path", "", "Directory with nodes.csv/edges.csv; empty = the launch graph"),))
    plugin.connectome(
        "builtin.connectome.random_control", "Random topology control", random_control_connectome,
        description="Same nodes, edge count, weight multiset and self-loop count as the base connectome; "
                    "uniformly random off-diagonal positions (reservoir.random_control).",
        parameters=(Parameter("base", "str", "builtin.connectome.flywire_cx512", "Base connectome id"),
                    Parameter("seed", "int", 42, "Randomization seed", minimum=0)))
    plugin.reservoir(
        "builtin.reservoir.leaky_tanh", "Sparse leaky-tanh rate reservoir", leaky_tanh_reservoir,
        description="x <- (1-leak) x + leak tanh(W x + input_strength U u + extra); W row-normalized by "
                    "absolute in-weight and scaled by gain; U fixed from the seed.",
        capabilities=("idle", "extra_input", "set_leak", "snapshot"),
        parameters=(Parameter("dt", "float", 0.2, "Time step", minimum=1e-6),
                    Parameter("gain", "float", 0.9, "Recurrent gain", minimum=0.0, maximum=0.999999),
                    Parameter("input_strength", "float", 0.5, "Input scale", minimum=0.0)))
    plugin.neural_interface(
        "builtin.interface.original", "Original neural interface", original_interface,
        description="Reference LM<->reservoir mathematics: tanh projection in, readout/decoder out, "
                    "delta = strength * RMS(h) * centered / max(RMS(centered), 1).")
    for policy, component_id, name, semantics in (
            ("auto", "builtin.coupling.auto", "Automatic (legacy for one pair, synchronous for several)",
             {"mode": "legacy_single or synchronous_multi by active pair count"}),
            ("legacy", "builtin.coupling.legacy_single", "Legacy single bypass", LEGACY_SEMANTICS),
            ("synchronous", "builtin.coupling.synchronous_multi", "Synchronous multi-bypass", SYNCHRONOUS_SEMANTICS)):
        plugin.add(ComponentSpec(component_id, "coupling", name, _coupling(policy), metadata={"semantics": semantics}))
    plugin.add(ComponentSpec(
        "builtin.modulation.neuron_cue", "modulation", "Neuron-group cue", lambda context: cue_group(
            context.connectome, context.target, context.seed),
        description="Extra input to a neuron group while injected tokens are processed: 'da' = DA-labelled "
                    "neurons, 'random' = as many non-DA neurons (seeded).", capabilities=("requires:extra_input",)))
