"""Spatial Dopamine Field Prototype (SHADOW MODE: nothing is fed back into the reservoir).

1. Audits the CX512 anatomy actually present (data_availability.json).
2. Obtains per-token reservoir states: a short Continuous Generation run (optionally with the
   existing DA-neuron cue) or an existing run directory saved with snapshot_interval=1.
3. Maps DA-classified neurons' signed activations to a nonnegative release proxy, integrates
   dC/dt = D lap(C) - k C + S on a voxel grid around the community-marked FlyWire positions
   (a MARKER-PROXY: not release sites), and computes hypothetical local leak rates f(C_i),
   which are never applied.
Run from Experiments -> Run Python Script... Output: results/plugins/dopamine-field/<time>/
All numbers below are illustrative, not biological constants. See docs/RESEARCH_PLUGINS.md.
"""
from pathlib import Path

import numpy as np

from liquidaion_fly.continuous import load_run
from liquidaion_fly.dopamine_field import FieldConfig, LeakResponse, audit_anatomy, load_geometry, simulate, write_outputs

# --- activity source -------------------------------------------------------------------------
SOURCE = "continuous"             # "continuous" (new short run) or "existing_run"
EXISTING_RUN = None               # path to a continuous run directory saved with snapshot_interval=1
SEED_TEXT = "Однажды в студёную зимнюю пору"
TOKENS = 160
CUE_TARGET = "da"                 # existing cue on the 16 DA neurons while injected tokens are processed; "off"
CUE_STRENGTH = 0.5
INJECTION = (40, " Из-за деревьев неожиданно появился огромный зелёный мамонт.")  # None for no injection
ANATOMY_DIR = None                # None = <graph dir>/../anatomy

# --- field (illustrative units: µm, s, release units) ------------------------------------------
DIFFUSION_UM2_PER_S = 200.0
CLEARANCE_PER_S = 0.5
VOXEL_UM = 5.0
TOKEN_INTERVAL_S = 0.1            # field time per token step
POST_SECONDS = 10.0               # sources off afterwards
BOUNDARY = "noflux"               # or "absorbing"
RELEASE_TRANSFER = "relu"         # or "softplus"
RELEASE_GAIN = 1.0
RELEASE_THRESHOLD = 0.0           # signed reservoir activation above which release starts
RELEASE_TAU_S = 0.2

# --- hypothetical concentration -> leak response (never applied) -------------------------------
BASELINE_CONCENTRATION = 0.0
SENSITIVITY = 1.0e4               # per (units/µm³); choose relative to the concentration range reached
DIRECTION = +1                    # +1: leak rises with C; -1: leak falls with C
LEAK_MIN, LEAK_BASELINE, LEAK_MAX = 0.05, 0.20, 0.35
SATURATING = True
SAVE_PLOTS = True


def run(api):
    api.set_output_name("dopamine-field")
    graph_dir = api.graph_dir
    anatomy_dir = Path(ANATOMY_DIR) if ANATOMY_DIR else graph_dir.parent / "anatomy"
    audit = audit_anatomy(graph_dir, anatomy_dir)
    api.log(f"DA neurons: {audit['da_neurons']['count']}; spatial field possible: {audit['spatial_field_possible']} "
            f"({audit['grounding']})")
    if SOURCE == "existing_run":
        run_dir = Path(EXISTING_RUN)
    else:
        api.new_session()
        injections = (INJECTION,) if INJECTION else ()
        settings = api.get_config()["settings"]
        result = api.run_continuous(name="activity-run", seed_text=SEED_TEXT, total_tokens=TOKENS, condition="fly",
                                    decoding="greedy", eos_policy="suppress", snapshot_interval=1, checkpoints=(),
                                    cue_target=CUE_TARGET, cue_strength=CUE_STRENGTH, injections=injections,
                                    strength=settings["strength"], feedback=settings["feedback"],
                                    leak_rate=settings["leak_rate"])
        run_dir = Path(result["output"])
    recorded = load_run(run_dir)
    steps, states = recorded["states"]["steps"], recorded["states"]["states"]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError("The run must be saved with snapshot_interval=1 (one state per token)")
    cue = {row["step"]: bool(row.get("cue_applied")) for row in recorded["telemetry"]}
    stimulated = np.array([cue.get(step, False) for step in steps])
    source = {"run": str(run_dir), "tokens": len(steps), "cue_target": recorded["metadata"]["config"].get("cue_target"),
              "cued_tokens": int(stimulated.sum()), "state": "reservoir state after each token step"}
    geometry = load_geometry(graph_dir, anatomy_dir)
    if geometry is None:
        write_outputs(api.output_dir, audit, None, None, source)
        api.log("No defensible coordinates for all DA neurons: only data_availability.json was written.")
        return
    before = api.get_reservoir_state()["state"].copy()
    field = simulate(states, geometry,
                     FieldConfig(DIFFUSION_UM2_PER_S, CLEARANCE_PER_S, VOXEL_UM, token_interval_s=TOKEN_INTERVAL_S,
                                 post_seconds=POST_SECONDS, boundary=BOUNDARY, release_transfer=RELEASE_TRANSFER,
                                 release_gain=RELEASE_GAIN, release_threshold=RELEASE_THRESHOLD,
                                 release_tau_s=RELEASE_TAU_S),
                     LeakResponse(BASELINE_CONCENTRATION, SENSITIVITY, DIRECTION, LEAK_MIN, LEAK_BASELINE, LEAK_MAX,
                                  SATURATING), stimulated=stimulated)
    if not np.array_equal(before, api.get_reservoir_state()["state"]):
        raise RuntimeError("Shadow mode violated: the reservoir changed during the field computation")
    write_outputs(api.output_dir, audit, field, geometry, source, SAVE_PLOTS)
    api.log(f"grid {field['grid_shape']} at {VOXEL_UM} µm, {field['substeps_per_token']} substeps/token; "
            f"min C {field['min_concentration']:.3g}; hypothetical leak range {field['leak_range']}; "
            f"clearance half-time {field['clearance_half_time_s']} s (k-only expectation "
            f"{field['clearance_expected_half_time_s']})")
    api.log(f"Saved to {api.output_dir}")
