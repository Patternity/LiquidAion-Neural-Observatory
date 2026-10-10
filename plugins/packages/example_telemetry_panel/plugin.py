"""Example B: a dockable panel. Qt is imported only inside the factory, so headless use never needs it."""


def create_panel(context):
    from .panel import TelemetryPanel  # imported on the GUI thread, only when the panel is shown
    return TelemetryPanel(context)


def register(plugin):
    plugin.panel("example_telemetry_panel.live", "Example: live telemetry", create_panel, area="right",
                 description="State RMS, relative intervention and steering norm from the telemetry event.")
