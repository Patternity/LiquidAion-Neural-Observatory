"""Shadow-mode spatial dopamine field prototype (hypothetical; never fed back into the reservoir).

Reads recorded reservoir states, maps the 16 DA-classified neurons' activity to a nonnegative
release proxy and integrates a 3D reaction-diffusion field

    dC/dt = D ∇²C − k C + S(r, t)

on a voxel grid around the CX512 neurons' community-marked FlyWire positions. Hypothetical
local leak rates alpha_i = f(C_i) are computed but never applied.

Anatomical grounding (see audit_anatomy): the only per-neuron coordinates are community-marked
positions (one per neuron; not soma centroids, not release sites). Synapse and release-site
coordinates, DA axon geometry and receptor expression are not in the repository. Therefore the
field is a MARKER-PROXY illustration: sources sit at the DA neurons' marker points. It is not a
reconstructed dopamine release field. Units: lengths in µm (from FlyWire nm), time in seconds
of field time, concentration in uncalibrated release units per µm³. D, k, gains and the
token-to-time mapping are illustrative parameters, not measured constants.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.sparse import csgraph

MARKER_PROXY = ("marker-proxy: sources at community-marked FlyWire positions of DA-classified neurons; "
                "not soma centroids, not release sites, not synapse locations")


def audit_anatomy(graph_dir: str | Path, anatomy_dir: str | Path) -> dict[str, Any]:
    """What the repository actually contains for the CX512 neurons (no external data is fetched)."""
    graph_dir, anatomy_dir = Path(graph_dir), Path(anatomy_dir)
    nodes = pd.read_csv(graph_dir / "nodes.csv", dtype={"root_id": str})
    edges = pd.read_csv(graph_dir / "edges.csv", nrows=3)
    report: dict[str, Any] = {
        "neurons": int(len(nodes)), "identifiers": "FlyWire root_id (materialization 783)",
        "node_columns": list(nodes.columns), "edge_columns": list(edges.columns),
        "neurotransmitter_counts": {str(k): int(v) for k, v in nodes["nt_type"].fillna("unassigned").value_counts().items()},
        "neurotransmitter_semantics": "Codex predicted transmitter (nt_type) with prediction score; a "
                                      "classifier output, not a measured identity",
        "connectivity": "aggregated synapse counts per directed neuron pair (edges.csv); no synapse positions",
        "synapse_positions": False, "release_site_positions": False, "receptor_expression": False,
        "axon_or_neurite_geometry_for_da": False}
    da = nodes[nodes["nt_type"] == "DA"]
    report["da_neurons"] = {"count": int(len(da)), "root_ids": da["root_id"].tolist(),
                            "nt_type_score": {"min": float(da["nt_type_score"].min()),
                                              "max": float(da["nt_type_score"].max())}
                            if len(da) and "nt_type_score" in da else None}
    manifest_path = anatomy_dir / "manifest.json"
    positions_path = anatomy_dir / "positions.csv"
    if manifest_path.exists() and positions_path.exists():
        manifest = json.loads(manifest_path.read_text())
        positions = pd.read_csv(positions_path, dtype={"root_id": str})
        marked = pd.read_csv(anatomy_dir / "marked_positions.csv", dtype={"root_id": str}) \
            if (anatomy_dir / "marked_positions.csv").exists() else positions
        skeletons = set(manifest.get("skeletons", {}))
        report["coordinates"] = {
            "space": manifest.get("coordinate_space"), "units": manifest.get("units"),
            "semantics": manifest.get("position_semantics"),
            "neurons_with_position": int(positions["root_id"].isin(nodes["root_id"]).sum()),
            "marked_positions_total": int(len(marked)),
            "da_neurons_with_position": int(positions["root_id"].isin(da["root_id"]).sum()),
            "da_marked_positions_total": int(marked["root_id"].isin(da["root_id"]).sum())}
        report["skeletons"] = {"neurons": len(skeletons), "da_neurons": len(skeletons & set(da["root_id"])),
                               "semantics": manifest.get("skeleton_semantics")}
        report["axon_or_neurite_geometry_for_da"] = bool(skeletons & set(da["root_id"]))
    else:
        report["coordinates"] = None
        report["skeletons"] = None
    coordinates = report["coordinates"]
    report["spatial_field_possible"] = bool(coordinates and coordinates["da_neurons_with_position"] == len(da) > 0
                                            and coordinates["neurons_with_position"] == len(nodes))
    report["grounding"] = (MARKER_PROXY if report["spatial_field_possible"] else
                           "no defensible coordinates for all DA neurons; no spatial field is computed")
    report["missing_for_a_grounded_field"] = [
        "presynaptic site (T-bar) coordinates of DA neurons, e.g. the FlyWire synapse table for materialization 783",
        "DA neuron axon/neurite skeletons or meshes (none of the 16 DA neurons has a skeleton here)",
        "receptor expression per receiving neuron (e.g. Dop1R1/Dop1R2/Dop2R/DopEcR) from transcriptomic data",
        "neuropil extracellular volume fraction and tortuosity for an effective diffusion coefficient",
        "dopamine transporter (DAT) density for clearance"]
    return report


def load_geometry(graph_dir: str | Path, anatomy_dir: str | Path) -> dict[str, Any] | None:
    """Marker positions (µm) for all neurons in graph order, DA indices and the graph; None if unavailable."""
    audit = audit_anatomy(graph_dir, anatomy_dir)
    if not audit["spatial_field_possible"]:
        return None
    nodes = pd.read_csv(Path(graph_dir) / "nodes.csv", dtype={"root_id": str})
    positions = pd.read_csv(Path(anatomy_dir) / "positions.csv", dtype={"root_id": str}).set_index("root_id")
    xyz = positions.loc[nodes["root_id"], ["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=float) / 1000.0
    edges = pd.read_csv(Path(graph_dir) / "edges.csv", dtype={"pre_root_id": str, "post_root_id": str})
    index = {root: i for i, root in enumerate(nodes["root_id"])}
    graph = sparse.csr_matrix((np.ones(len(edges)), (edges["pre_root_id"].map(index), edges["post_root_id"].map(index))),
                              shape=(len(nodes), len(nodes)))
    return {"root_ids": nodes["root_id"].tolist(), "positions_um": xyz,
            "da_indices": np.flatnonzero(nodes["nt_type"].to_numpy() == "DA").tolist(), "graph": graph,
            "audit": audit}


@dataclass
class FieldConfig:
    diffusion_um2_per_s: float = 200.0   # effective D (illustrative, not calibrated)
    clearance_per_s: float = 0.5         # first-order removal k (illustrative)
    voxel_um: float = 5.0
    margin_um: float = 20.0
    token_interval_s: float = 0.1        # field time per recorded forward pass (an explicit mapping)
    post_seconds: float = 10.0           # sources off afterwards: spread and clearance only
    boundary: str = "noflux"             # "noflux" (Neumann) or "absorbing" (C = 0 outside the box)
    release_transfer: str = "relu"       # "relu" or "softplus" of (gain * (x - threshold))
    release_gain: float = 1.0            # release units per second per unit activation above threshold
    release_threshold: float = 0.0
    release_tau_s: float = 0.2           # release kinetics: dr/dt = (target - r) / tau; 0 = instantaneous
    cfl: float = 0.9                     # fraction of the positivity limit used for the explicit step
    arrival_fraction: float = 0.1        # arrival = first time C_i reaches this fraction of its own peak

    def validate(self) -> None:
        if self.diffusion_um2_per_s < 0 or self.clearance_per_s < 0 or self.voxel_um <= 0:
            raise ValueError("D and k must be non-negative and the voxel size positive")
        if self.token_interval_s <= 0 or self.post_seconds < 0 or not 0 < self.cfl <= 1:
            raise ValueError("Token interval must be positive, post period non-negative and 0 < cfl <= 1")
        if self.boundary not in {"noflux", "absorbing"} or self.release_transfer not in {"relu", "softplus"}:
            raise ValueError("Unknown boundary or release transfer")
        if self.release_gain < 0 or self.release_tau_s < 0:
            raise ValueError("Release gain and tau must be non-negative")


@dataclass
class LeakResponse:
    """Hypothetical concentration -> local leak rate curve (illustrative values, not biology).

    m = direction * tanh(sensitivity * (C - baseline_concentration))   (saturating)
        direction * sensitivity * (C - baseline_concentration)        (linear, then clipped)
    leak = leak_baseline + m * (leak_max - leak_baseline)   for m >= 0
           leak_baseline + m * (leak_baseline - leak_min)   for m < 0
    direction = +1 raises leak with concentration, -1 lowers it.
    """
    baseline_concentration: float = 0.0
    sensitivity: float = 1.0
    direction: int = 1
    leak_min: float = 0.05
    leak_baseline: float = 0.20
    leak_max: float = 0.35
    saturating: bool = True

    def validate(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        if not 0 < self.leak_min <= self.leak_baseline <= self.leak_max <= 1:
            raise ValueError("Require 0 < leak_min <= leak_baseline <= leak_max <= 1")
        if self.sensitivity < 0:
            raise ValueError("sensitivity must be non-negative")

    def __call__(self, concentration: np.ndarray) -> np.ndarray:
        u = self.sensitivity * (np.asarray(concentration, dtype=float) - self.baseline_concentration)
        m = self.direction * (np.tanh(u) if self.saturating else u)
        leak = np.where(m >= 0, self.leak_baseline + m * (self.leak_max - self.leak_baseline),
                        self.leak_baseline + m * (self.leak_baseline - self.leak_min))
        return np.clip(leak, self.leak_min, self.leak_max)


class Field3D:
    """Explicit finite-difference reaction-diffusion on a regular voxel grid (µm, s)."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray, config: FieldConfig):
        config.validate()
        self.config = config
        self.h = config.voxel_um
        self.origin = np.asarray(lower, dtype=float)
        self.shape = tuple(int(math.ceil((u - l) / self.h)) + 1 for l, u in zip(lower, upper))
        self.C = np.zeros(self.shape)
        self.voxel_volume = self.h ** 3
        rate = 6 * config.diffusion_um2_per_s / self.h ** 2 + config.clearance_per_s
        # Positivity of the explicit update: dt * (6D/h^2 + k) <= 1 (all update coefficients >= 0).
        self.dt_limit = math.inf if rate == 0 else 1.0 / rate
        self.time = 0.0
        self.boundary_loss = 0.0

    def index(self, points_um: np.ndarray) -> np.ndarray:
        return (np.asarray(points_um, dtype=float) - self.origin) / self.h

    def laplacian(self, C: np.ndarray) -> np.ndarray:
        padded = np.pad(C, 1, mode="edge" if self.config.boundary == "noflux" else "constant")
        core = padded[1:-1, 1:-1, 1:-1]
        return (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1] + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
                + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2] - 6 * core) / self.h ** 2

    def step(self, dt: float, source_voxels: np.ndarray, source_rates: np.ndarray) -> None:
        """Advance by dt (must respect dt_limit); sources add rate * dt amount to their voxel."""
        if dt > self.dt_limit * (1 + 1e-12):
            raise ValueError(f"dt {dt} exceeds the positivity limit {self.dt_limit}")
        config = self.config
        before = self.C.sum() * self.voxel_volume
        self.C = self.C + dt * (config.diffusion_um2_per_s * self.laplacian(self.C) - config.clearance_per_s * self.C)
        if len(source_voxels):
            np.add.at(self.C, tuple(source_voxels.T), source_rates * dt / self.voxel_volume)
        if config.boundary == "absorbing":
            expected = before + dt * (source_rates.sum() - config.clearance_per_s * before)
            self.boundary_loss += expected - self.C.sum() * self.voxel_volume
        self.time += dt

    def sample(self, points_um: np.ndarray) -> np.ndarray:
        """Trilinear interpolation at points (inside the grid)."""
        return ndimage.map_coordinates(self.C, self.index(points_um).T, order=1, mode="nearest")

    def gradient_magnitude(self) -> np.ndarray:
        gx, gy, gz = np.gradient(self.C, self.h)
        return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def release_target(activity: np.ndarray, config: FieldConfig) -> np.ndarray:
    """Nonnegative release proxy from SIGNED reservoir activations (not firing rates)."""
    drive = config.release_gain * (np.asarray(activity, dtype=float) - config.release_threshold)
    return np.maximum(drive, 0.0) if config.release_transfer == "relu" else np.logaddexp(0.0, drive)


def simulate(states: np.ndarray, geometry: dict[str, Any], config: FieldConfig, response: LeakResponse,
             snapshot_steps: int = 4, stimulated: np.ndarray | None = None) -> dict[str, Any]:
    """Integrate the field over recorded token steps (T x 512 states), then a source-off period."""
    config.validate()
    response.validate()
    positions = geometry["positions_um"]
    da = np.asarray(geometry["da_indices"], dtype=int)
    receivers = np.setdiff1d(np.arange(len(positions)), da)
    lower, upper = positions.min(axis=0) - config.margin_um, positions.max(axis=0) + config.margin_um
    field = Field3D(lower, upper, config)
    source_voxels = np.rint(field.index(positions[da])).astype(int)
    substeps = max(1, math.ceil(config.token_interval_s / (config.cfl * field.dt_limit)))
    dt = config.token_interval_s / substeps
    post_steps = int(round(config.post_seconds / config.token_interval_s))
    total_steps = len(states) + post_steps
    snapshot_at = sorted({int(round(x)) for x in np.linspace(0, total_steps - 1, snapshot_steps)})
    rate = np.zeros(len(da))
    series, snapshots, leak_trace, concentration_trace, mass, release_trace = [], {}, [], [], [], []
    for step in range(total_steps):
        stimulating = step < len(states)
        target = release_target(states[step, da], config) if stimulating else np.zeros(len(da))
        for _ in range(substeps):
            if config.release_tau_s > 0:
                rate = target + (rate - target) * math.exp(-dt / config.release_tau_s)
            else:
                rate = target
            expected = field.C.sum() * field.voxel_volume + dt * (rate.sum() - config.clearance_per_s *
                                                                  field.C.sum() * field.voxel_volume)
            field.step(dt, source_voxels, rate)
            mass.append(expected - field.C.sum() * field.voxel_volume)
        release_trace.append(rate.copy())
        at_neurons = field.sample(positions)
        concentration_trace.append(at_neurons)
        leak_trace.append(response(at_neurons))
        gradient = field.gradient_magnitude()
        active = field.C > 0.01 * field.C.max() if field.C.max() > 0 else np.zeros_like(field.C, dtype=bool)
        series.append({"step": step, "time_s": field.time, "phase": "stimulation" if stimulating else "sources_off",
                       "stimulated_token": bool(stimulated[step]) if stimulated is not None and stimulating else None,
                       "release_total": float(rate.sum()), "release_max": float(rate.max()) if len(rate) else 0.0,
                       "total_amount": float(field.C.sum() * field.voxel_volume), "c_max": float(field.C.max()),
                       "c_mean": float(field.C.mean()), "c_var": float(field.C.var()),
                       "c_sources_mean": float(at_neurons[da].mean()), "c_receivers_mean": float(at_neurons[receivers].mean()),
                       "grad_mean_active": float(gradient[active].mean()) if active.any() else 0.0,
                       "grad_receivers_mean": float(ndimage.map_coordinates(gradient, field.index(positions[receivers]).T,
                                                                            order=1, mode="nearest").mean()),
                       "c_min": float(field.C.min())})
        if step in snapshot_at:
            snapshots[step] = {"time_s": field.time, "max_projection_z": field.C.max(axis=2).copy()}
    concentration = np.array(concentration_trace)
    leaks = np.array(leak_trace)
    distance = np.min(np.linalg.norm(positions[:, None, :] - positions[da][None, :, :], axis=2), axis=1)
    hops = csgraph.shortest_path(geometry["graph"], unweighted=True, indices=da).min(axis=0)
    peaks = concentration.max(axis=0)
    arrival = np.array([series[int(np.argmax(concentration[:, i] >= config.arrival_fraction * peaks[i]))]["time_s"]
                        if peaks[i] > 0 else math.nan for i in range(len(positions))])
    bins = [0, 10, 25, 50, 100, 200, math.inf]
    by_distance = []
    for low, high in zip(bins, bins[1:]):
        mask = (distance >= low) & (distance < high)
        mask[da] = False
        if mask.any():
            by_distance.append({"distance_um": [low, high], "receivers": int(mask.sum()),
                                "median_arrival_s": float(np.nanmedian(arrival[mask])) if np.isfinite(arrival[mask]).any() else None,
                                "median_peak_concentration": float(np.median(peaks[mask]))})
    stop = len(states)
    total = np.array([row["total_amount"] for row in series])
    after = total[stop:] if stop < len(total) else np.array([])
    half = next((series[stop + i]["time_s"] - series[stop - 1]["time_s"] for i, v in enumerate(after)
                 if stop > 0 and v <= 0.5 * total[stop - 1]), None) if stop > 0 else None
    finite = np.isfinite(hops[receivers])
    return {"config": asdict(config), "response": asdict(response), "grid_shape": list(field.shape),
            "grid_origin_um": field.origin.tolist(), "voxel_um": field.h, "substeps_per_token": substeps,
            "dt_s": dt, "dt_positivity_limit_s": field.dt_limit, "series": series, "snapshots": snapshots,
            "concentration_at_neurons": concentration, "hypothetical_leak": leaks,
            "release_rates": np.array(release_trace), "da_indices": da.tolist(),
            "min_concentration": float(min(row["c_min"] for row in series)),
            "max_mass_balance_error": float(np.max(np.abs(mass))) if config.boundary == "noflux" and mass else None,
            "boundary_loss_amount": field.boundary_loss if config.boundary == "absorbing" else 0.0,
            "leak_range": [float(leaks.min()), float(leaks.max())],
            "spread_by_distance": by_distance,
            "clearance_half_time_s": half, "clearance_expected_half_time_s":
                math.log(2) / config.clearance_per_s if config.clearance_per_s > 0 and config.boundary == "noflux" else None,
            "distance_vs_graph": {"spatial_distance_um_to_nearest_da": distance.tolist(),
                                  "graph_hops_from_nearest_da": [None if not np.isfinite(h) else int(h) for h in hops],
                                  "spearman_receivers": _spearman(distance[receivers][finite], hops[receivers][finite]),
                                  "note": "graph hops are synaptic topology, not extracellular distance"},
            "grounding": MARKER_PROXY}


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def write_outputs(directory: Path, audit: dict[str, Any], result: dict[str, Any] | None, geometry: dict | None,
                  source: dict[str, Any], save_plots: bool = True) -> list[str]:
    """data_availability.json always; field results, CSVs, arrays and plots only when computed."""
    import csv
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data_availability.json").write_text(json.dumps(audit, indent=2) + "\n")
    written = ["data_availability.json"]
    if result is None:
        return written
    metadata = {key: value for key, value in result.items()
                if key not in {"series", "snapshots", "concentration_at_neurons", "hypothetical_leak",
                               "release_rates", "distance_vs_graph"}}
    metadata.update(source=source, shadow_mode="computed from recorded states; never applied to the reservoir",
                    hypothetical="release proxy, diffusion and leak response are illustrative assumptions; "
                                 "no receptor identities are assigned")
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=float) + "\n")
    with (directory / "field_series.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["series"][0]))
        writer.writeheader()
        writer.writerows(result["series"])
    np.savez_compressed(directory / "field_arrays.npz", concentration_at_neurons=result["concentration_at_neurons"],
                        hypothetical_leak=result["hypothetical_leak"], release_rates=result["release_rates"],
                        da_indices=np.array(result["da_indices"]), positions_um=geometry["positions_um"],
                        **{f"snapshot_step{k}_max_projection_z": v["max_projection_z"]
                           for k, v in result["snapshots"].items()})
    (directory / "distance_vs_graph.json").write_text(json.dumps(result["distance_vs_graph"]) + "\n")
    written += ["metadata.json", "field_series.csv", "field_arrays.npz", "distance_vs_graph.json"]
    if save_plots:
        written += plot_field(directory / "plots", result, geometry)
    return written


def plot_field(directory: Path, result: dict[str, Any], geometry: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    series = result["series"]
    t = np.array([row["time_s"] for row in series])
    stop = next((row["time_s"] for row in series if row["phase"] == "sources_off"), None)
    label = "MARKER-PROXY field (illustrative; not reconstructed release sites)"
    names = []

    def finish(fig, axis, name, ylabel):
        if stop is not None:
            axis.axvline(stop, color="gray", linestyle="--", linewidth=1)
        axis.set_xlabel("field time (s; forward passes × token_interval_s)")
        axis.set_ylabel(ylabel)
        axis.set_title(label, fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / name, dpi=110)
        plt.close(fig)
        names.append(name)
    fig, axis = plt.subplots(figsize=(8, 3.6))
    axis.plot(t, result["release_rates"], linewidth=0.8)
    finish(fig, axis, "01_da_release.png", "release proxy per DA neuron (units/s)")
    fig, axis = plt.subplots(figsize=(8, 3.6))
    for key, text in (("c_max", "max"), ("c_sources_mean", "mean at DA markers"),
                      ("c_receivers_mean", "mean at receiver markers"), ("c_mean", "grid mean")):
        axis.plot(t, [row[key] for row in series], label=text)
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.legend(fontsize=7)
    finish(fig, axis, "02_concentration.png", "C (units/µm³)")
    snapshots = result["snapshots"]
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 3.6), squeeze=False)
    vmax = max(v["max_projection_z"].max() for v in snapshots.values()) or 1.0
    origin, h = np.array(result["grid_origin_um"]), result["voxel_um"]
    positions, da = geometry["positions_um"], result["da_indices"]
    for axis, (step, snap) in zip(axes[0], sorted(snapshots.items())):
        image = snap["max_projection_z"].T
        extent = [origin[0], origin[0] + h * (image.shape[1] - 1), origin[1], origin[1] + h * (image.shape[0] - 1)]
        axis.imshow(image, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        axis.scatter(positions[:, 0], positions[:, 1], s=2, c="cyan", alpha=0.4)
        axis.scatter(positions[da, 0], positions[da, 1], s=14, facecolors="none", edgecolors="lime")
        axis.set_title(f"t = {snap['time_s']:.1f} s · max over z", fontsize=8)
        axis.set_xlabel("x (µm)")
    axes[0][0].set_ylabel("y (µm)")
    fig.suptitle(label + " · cyan = marker positions, green = DA markers", fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "03_spatial_snapshots.png", dpi=110)
    plt.close(fig)
    names.append("03_spatial_snapshots.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(t, [row["grad_mean_active"] for row in series], label="mean |∇C| where C > 1% max")
    axes[0].plot(t, [row["grad_receivers_mean"] for row in series], label="mean |∇C| at receivers")
    axes[0].legend(fontsize=7)
    axes[0].set_xlabel("field time (s)")
    distance = np.array(result["distance_vs_graph"]["spatial_distance_um_to_nearest_da"])
    peaks = result["concentration_at_neurons"].max(axis=0)
    axes[1].scatter(distance, peaks, s=4)
    axes[1].set_yscale("symlog", linthresh=1e-8)
    axes[1].set_xlabel("distance to nearest DA marker (µm)")
    axes[1].set_ylabel("peak C at marker")
    fig.suptitle(label, fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "04_gradients.png", dpi=110)
    plt.close(fig)
    names.append("04_gradients.png")
    leaks = result["hypothetical_leak"]
    fig, axis = plt.subplots(figsize=(8, 3.6))
    for q, style in ((0, ":"), (10, "--"), (50, "-"), (90, "--"), (100, ":")):
        axis.plot(t, np.percentile(leaks, q, axis=1), style, color="C0", label=f"p{q}")
    axis.legend(fontsize=7, ncol=5)
    finish(fig, axis, "05_hypothetical_leak.png", "hypothetical local leak (NOT applied)")
    response = LeakResponse(**result["response"])
    top = max(float(result["concentration_at_neurons"].max()) * 1.5, response.baseline_concentration + 1e-9)
    c = np.linspace(0, top, 400)
    fig, axis = plt.subplots(figsize=(6, 3.6))
    axis.plot(c, response(c))
    axis.axvspan(0, float(result["concentration_at_neurons"].max()), color="gray", alpha=0.15,
                 label="concentration range reached")
    axis.set_xlabel("C (units/µm³)")
    axis.set_ylabel("leak")
    axis.set_title("configured concentration → leak response (hypothetical)", fontsize=8)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "06_response_curve.png", dpi=110)
    plt.close(fig)
    names.append("06_response_curve.png")
    hops = result["distance_vs_graph"]["graph_hops_from_nearest_da"]
    fig, axis = plt.subplots(figsize=(6, 3.6))
    axis.scatter(distance, [np.nan if h is None else h for h in hops], s=4)
    axis.set_xlabel("spatial distance to nearest DA marker (µm)")
    axis.set_ylabel("graph hops from nearest DA neuron")
    axis.set_title("spatial distance vs synaptic graph distance (different quantities)", fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "07_distance_vs_graph.png", dpi=110)
    plt.close(fig)
    names.append("07_distance_vs_graph.png")
    return names
