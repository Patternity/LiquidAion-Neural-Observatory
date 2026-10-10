"""Example A: register an alternative neural interface and an experiment that compares it.

Enable: Plugins -> Manage plugins... (then restart), or --plugin example_normalized_interface,
or LIQUIDAION_PLUGINS=example_normalized_interface. Select it in the Components panel, or run
the experiment from the Plugins panel. Nothing is selected automatically.
"""
from liquidaion_fly.plugin_api import Parameter

from .normalized_neural_interface import UnitRmsNeuralInterface

INTERFACE = "example_normalized_interface.unit_rms"


def make_interface(context):
    # context: hidden_size, reservoir_neurons, features, seed, params, model, num_layers, device
    return UnitRmsNeuralInterface(context.hidden_size, context.reservoir_neurons, context.features, context.seed,
                                  min_rms=context.params["min_rms"])


def compare(api):
    """Same prompt from a fresh session with the original and the unit-RMS interface."""
    previous = api.components()["neural_interface"]["id"]
    results = {}
    try:
        for interface in ("builtin.interface.original", INTERFACE):
            api.check_cancelled()
            api.new_session()
            api.select_components(neural_interface=interface)
            result = api.generate("Однажды в студёную зимнюю пору", max_tokens=48, mode="fly")
            relative = [row["delta_relative_rms"] for row in result.telemetry]
            api.log(f"{interface}: median relative intervention "
                    f"{sorted(relative)[len(relative) // 2]:.4f}; text {result.text[:60]!r}")
            results[interface] = result
        api.save_result("comparison", results)
    finally:
        api.select_components(neural_interface=previous)


def register(plugin):
    plugin.neural_interface(
        INTERFACE, "Unit-RMS steering (experimental)", make_interface, version="0.1.0",
        description="Original interface with direction = centered / RMS(centered); experimental example.",
        parameters=(Parameter("min_rms", "float", 1e-6, "No steering below this RMS of the centered vector",
                              minimum=0.0),))
    plugin.experiment("example_normalized_interface.compare", "Compare original vs unit-RMS steering", compare,
                      description="Fly turns from fresh sessions with both interfaces; saves both results.")
