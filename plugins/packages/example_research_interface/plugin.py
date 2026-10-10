"""Example C: one plugin, three contributions (interface, panel, experiment) sharing one id prefix."""
from liquidaion_fly.plugin_api import Parameter

from .gated_interface import GatedSteeringInterface

INTERFACE = "example_research_interface.gated"


def make_interface(context):
    return GatedSteeringInterface(context.hidden_size, context.reservoir_neurons, context.features, context.seed,
                                  gate_gain=context.params["gate_gain"], floor=context.params["floor"])


def create_panel(context):
    from .panel import GatedInterfacePanel
    return GatedInterfacePanel(context)


def demonstration(api):
    """Original vs gated from fresh sessions on the same prompt; the previous selection is restored."""
    previous = api.components()["neural_interface"]["id"]
    try:
        results = {}
        for interface in ("builtin.interface.original", INTERFACE):
            api.check_cancelled()
            api.new_session()
            api.select_components(neural_interface=interface)
            results[interface] = api.generate("Однажды в студёную зимнюю пору", max_tokens=32, mode="fly")
            gates = [row.get("interface_metrics", {}).get("gate") for row in results[interface].telemetry]
            api.log(f"{interface}: {len(results[interface].token_ids)} tokens; gates {gates[:4]}")
        api.save_result("gated_vs_original", results)
    finally:
        api.select_components(neural_interface=previous)


def register(plugin):
    plugin.neural_interface(
        INTERFACE, "Gated steering (experimental)", make_interface, version="0.1.0",
        description="Steering scaled by tanh(gate_gain * RMS(reservoir state)); experimental example.",
        parameters=(Parameter("gate_gain", "float", 10.0, "Gate sensitivity to reservoir RMS", minimum=0.0, maximum=1000.0),
                    Parameter("floor", "float", 1e-6, "Smallest RMS used to normalize the direction", minimum=1e-12)))
    plugin.panel("example_research_interface.panel", "Example: gated steering", create_panel, area="right",
                 description="Parameters, selection and a live gate readout for the gated interface.")
    plugin.experiment("example_research_interface.demo", "Gated vs original steering", demonstration)
