#!/usr/bin/env python3
"""Generate activation-only vs activation-with-coagulation comparison plots."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from analysis.activation_analysis import (
    activation_fraction,
    cloud_droplet_concentration,
    final_distribution,
    size_resolved_activation,
)
from analysis.coagulation_analysis import normalized_diameter, normalized_number
from analysis.figure_style import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    draw_replicates_with_mean,
    finish_panel,
    save_png,
)
from analysis.smoothing import (
    dense_log_grid,
    linear_edges_from_centers,
    log_edges_from_centers,
    replicate_kde_heatmap,
    select_replicate_for_heatmap,
)
from analysis.stochparticles_io import DATA_DIR, FIG_DIR, read_scene


SCENE_NAME = "activation_coagulation_comparison"
DATA_PATH = DATA_DIR / f"{SCENE_NAME}.h5"
COMPOSITE_PATH = FIG_DIR / f"{SCENE_NAME}.png"
PANEL_DIR = FIG_DIR / SCENE_NAME

PANEL_PATHS = {
    "a": PANEL_DIR / "a_activation_fraction.png",
    "b": PANEL_DIR / "b_cloud_droplet_concentration.png",
    "c": PANEL_DIR / "c_number_and_wet_size.png",
    "d": PANEL_DIR / "d_activation_only_heatmap.png",
    "e": PANEL_DIR / "e_activation_with_coagulation_heatmap.png",
    "f": PANEL_DIR / "f_final_distribution_overlay.png",
    "g": PANEL_DIR / "g_final_distribution_difference.png",
    "h": PANEL_DIR / "h_size_resolved_activation.png",
}

CASE_LABELS = {
    "activation_only": "Activation only",
    "activation_with_coagulation": "With coagulation",
}

CASE_COLORS = {
    "activation_only": PALETTE["activation_only"],
    "activation_with_coagulation": PALETTE["activation_with_coagulation"],
}

KDE_DENSITY_LABEL = "KDE number density (m$^{-3}$ dex$^{-1}$)"
KDE_MAX_BANDWIDTH_DEX = 0.06


def _all_positive_samples(scene_reps, key: str, final_only: bool = False) -> np.ndarray:
    samples = []
    for reps in scene_reps:
        for rep in reps:
            values = np.asarray(rep.arrays[key], dtype=float)
            if final_only and values.ndim >= 2:
                values = values[-1, :]
            values = values[np.isfinite(values) & (values > 0.0)]
            if values.size:
                samples.append(values)
    if not samples:
        raise ValueError(f"no positive finite samples found for {key!r}")
    return np.concatenate(samples)


def _log_grid_from_samples(samples: np.ndarray, n: int) -> np.ndarray:
    lo, hi = np.percentile(samples, [0.5, 99.5])
    lo = min(float(lo), float(np.min(samples)))
    hi = max(float(hi), float(np.max(samples)))
    return dense_log_grid(lo, hi, n=n)


def _log_grid_from_bin_edges_or_samples(reps, samples: np.ndarray, n: int) -> np.ndarray:
    for rep in reps:
        edges = np.asarray(rep.arrays.get("bin_edges", []), dtype=float)
        edges = edges[np.isfinite(edges) & (edges > 0.0)]
        if edges.size >= 2 and edges[-1] > edges[0]:
            return dense_log_grid(float(edges[0]), float(edges[-1]), n=n)
    return _log_grid_from_samples(samples, n=n)


def _time_minutes(time: np.ndarray) -> np.ndarray:
    return np.asarray(time, dtype=float) / 60.0


def _style_time_axis(ax, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#d9d9d9", lw=0.45, alpha=0.7)


def _draw_replicates(
    ax,
    x: np.ndarray,
    y_matrix: np.ndarray,
    color: str,
    label: str,
    *,
    linestyle: str = "-",
    lw_mean: float = 1.8,
) -> np.ndarray:
    line_start = len(ax.lines)
    mean = draw_replicates_with_mean(ax, x, y_matrix, color, label, lw_mean=lw_mean)
    for line in ax.lines[line_start:]:
        line.set_linestyle(linestyle)
    return mean


def _apply_log_x_distribution_style(ax, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("Wet diameter (um)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", color="#d9d9d9", lw=0.45, alpha=0.65)


def _kde_norm_from_reference(*heatmaps: np.ndarray) -> Normalize:
    finite_values = []
    positive_values = []
    for heatmap in heatmaps:
        values = np.asarray(heatmap, dtype=float)
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0.0]
        if finite.size:
            finite_values.append(finite)
        if positive.size:
            positive_values.append(positive)

    if not positive_values:
        if finite_values:
            vmax = float(np.max(np.concatenate(finite_values)))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
        else:
            vmax = 1.0
        return Normalize(vmin=0.0, vmax=vmax)

    positive = np.concatenate(positive_values)
    vmax = float(np.percentile(positive, 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(positive))
    return Normalize(vmin=0.0, vmax=max(vmax, 1.0))


def _plot_heatmap(
    ax,
    time: np.ndarray,
    diameter_grid: np.ndarray,
    heatmap: np.ndarray,
    norm: Normalize,
):
    cmap = plt.get_cmap("cividis").copy()
    mesh = ax.pcolormesh(
        linear_edges_from_centers(_time_minutes(time)),
        log_edges_from_centers(diameter_grid * 1e6),
        np.ma.masked_invalid(np.asarray(heatmap, dtype=float).T),
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(_time_minutes(time)[0]), float(_time_minutes(time)[-1]))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Wet diameter (um)")
    ax.grid(False)
    return mesh


def prepare_scene():
    scene = read_scene(DATA_PATH)
    try:
        only_reps = scene["activation_only"]
        coag_reps = scene["activation_with_coagulation"]
    except KeyError as exc:
        available = ", ".join(sorted(scene))
        raise KeyError(f"required case missing from {DATA_PATH}; available cases: {available}") from exc

    wet_samples = _all_positive_samples((only_reps, coag_reps), "diameter_samples")
    wet_grid = _log_grid_from_bin_edges_or_samples(
        only_reps + coag_reps, wet_samples, n=280
    )
    dry_samples = _all_positive_samples(
        (only_reps, coag_reps), "dry_diameter_samples", final_only=True
    )
    dry_edges = _log_grid_from_samples(dry_samples, n=44)

    heatmap_rep_index = 0
    only_rep = select_replicate_for_heatmap(only_reps, heatmap_rep_index)
    coag_rep = select_replicate_for_heatmap(coag_reps, heatmap_rep_index)
    only_time, _, only_heatmap = replicate_kde_heatmap(
        only_rep,
        grid=wet_grid,
        bandwidth_factor=1.1,
        max_bandwidth=KDE_MAX_BANDWIDTH_DEX,
        smooth_passes=1,
    )
    coag_time, _, coag_heatmap = replicate_kde_heatmap(
        coag_rep,
        grid=wet_grid,
        bandwidth_factor=1.1,
        max_bandwidth=KDE_MAX_BANDWIDTH_DEX,
        smooth_passes=1,
    )
    heatmap_norm = _kde_norm_from_reference(only_heatmap, coag_heatmap)

    activation_radius = float(
        only_reps[0].attrs.get("activation_radius", coag_reps[0].attrs.get("activation_radius", 0.0))
    )

    return {
        "activation_only": only_reps,
        "activation_with_coagulation": coag_reps,
        "wet_grid": wet_grid,
        "dry_edges": dry_edges,
        "heatmaps": {
            "activation_only": (only_time, wet_grid, only_heatmap),
            "activation_with_coagulation": (coag_time, wet_grid, coag_heatmap),
        },
        "heatmap_norm": heatmap_norm,
        "heatmap_replicate_index": heatmap_rep_index,
        "activation_radius": activation_radius,
    }


def plot_activation_fraction(ax, data) -> None:
    for case in ("activation_only", "activation_with_coagulation"):
        time, rows = activation_fraction(data[case])
        _draw_replicates(
            ax,
            _time_minutes(time),
            rows,
            CASE_COLORS[case],
            CASE_LABELS[case],
        )
    _style_time_axis(ax, "Activated fraction", ylim=(-0.02, 1.02))
    ax.legend(loc="best")


def plot_cloud_droplet_concentration(ax, data) -> None:
    for case in ("activation_only", "activation_with_coagulation"):
        time, rows = cloud_droplet_concentration(data[case])
        _draw_replicates(
            ax,
            _time_minutes(time),
            rows / 1.0e6,
            CASE_COLORS[case],
            CASE_LABELS[case],
        )
    _style_time_axis(ax, "Cloud droplets (cm$^{-3}$)", ylim=(0.0, None))
    ax.legend(loc="best")


def plot_number_and_wet_size(ax, data) -> None:
    ax_diameter = ax.twinx()

    number_handles = []
    diameter_handles = []
    for case in ("activation_only", "activation_with_coagulation"):
        color = CASE_COLORS[case]
        time, number_rows = normalized_number(data[case])
        _draw_replicates(
            ax,
            _time_minutes(time),
            number_rows,
            color,
            f"{CASE_LABELS[case]} N/N0",
            linestyle="-",
            lw_mean=1.75,
        )
        number_handles.append(ax.lines[-1])

        d_time, diameter_rows = normalized_diameter(data[case], key="mean_diameter")
        _draw_replicates(
            ax_diameter,
            _time_minutes(d_time),
            diameter_rows,
            color,
            f"{CASE_LABELS[case]} D/D0",
            linestyle="--",
            lw_mean=1.65,
        )
        diameter_handles.append(ax_diameter.lines[-1])

    _style_time_axis(ax, "N/N0", ylim=(0.0, 1.08))
    ax_diameter.set_ylabel("Mean wet diameter / D0")
    ax_diameter.grid(False)
    ax_diameter.spines["right"].set_visible(True)

    handles = number_handles + diameter_handles
    ax.legend(handles, [handle.get_label() for handle in handles], loc="best", ncols=2)


def plot_activation_only_heatmap(ax, fig, data, *, colorbar: bool = True):
    time, grid, heatmap = data["heatmaps"]["activation_only"]
    mesh = _plot_heatmap(ax, time, grid, heatmap, data["heatmap_norm"])
    ax.set_title("Activation only", pad=4)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(KDE_DENSITY_LABEL)
    return mesh


def plot_activation_with_coagulation_heatmap(ax, fig, data, *, colorbar: bool = True):
    time, grid, heatmap = data["heatmaps"]["activation_with_coagulation"]
    mesh = _plot_heatmap(ax, time, grid, heatmap, data["heatmap_norm"])
    ax.set_title("With coagulation", pad=4)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(KDE_DENSITY_LABEL)
    return mesh


def plot_final_distribution_overlay(ax, data) -> None:
    grid = data["wet_grid"]
    diameter_um = grid * 1e6
    for case in ("activation_only", "activation_with_coagulation"):
        rows = final_distribution(data[case], grid)
        _draw_replicates(
            ax,
            diameter_um,
            rows,
            CASE_COLORS[case],
            CASE_LABELS[case],
        )
    _apply_log_x_distribution_style(ax, f"Final {KDE_DENSITY_LABEL}")
    ax.legend(loc="best")


def plot_final_distribution_difference(ax, data) -> None:
    grid = data["wet_grid"]
    only_rows = final_distribution(data["activation_only"], grid)
    coag_rows = final_distribution(data["activation_with_coagulation"], grid)
    n_pair = min(only_rows.shape[0], coag_rows.shape[0])
    diameter_um = grid * 1e6

    if n_pair:
        pair_differences = coag_rows[:n_pair] - only_rows[:n_pair]
        for row in pair_differences:
            ax.plot(diameter_um, row, color=PALETTE["neutral"], lw=0.7, alpha=0.25)
        difference = np.nanmean(pair_differences, axis=0)
    else:
        difference = np.nanmean(coag_rows, axis=0) - np.nanmean(only_rows, axis=0)

    ax.axhline(0.0, color="#555555", lw=0.7)
    ax.plot(diameter_um, difference, color="#252525", lw=1.8, label="Mean difference")
    ax.fill_between(
        diameter_um,
        0.0,
        difference,
        where=difference >= 0.0,
        color=PALETTE["positive"],
        alpha=0.18,
        interpolate=True,
    )
    ax.fill_between(
        diameter_um,
        0.0,
        difference,
        where=difference < 0.0,
        color=PALETTE["negative"],
        alpha=0.18,
        interpolate=True,
    )
    _apply_log_x_distribution_style(
        ax, "With coag. - activation only\n(m$^{-3}$ dex$^{-1}$)"
    )
    ax.legend(loc="best")


def plot_size_resolved_activation(ax, data) -> None:
    for case in ("activation_only", "activation_with_coagulation"):
        centers, rows = size_resolved_activation(
            data[case], data["dry_edges"], data["activation_radius"]
        )
        valid_bins = np.any(np.isfinite(rows), axis=0)
        if not np.any(valid_bins):
            continue
        _draw_replicates(
            ax,
            centers[valid_bins] * 1e6,
            rows[:, valid_bins],
            CASE_COLORS[case],
            CASE_LABELS[case],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Current dry diameter (um)")
    ax.set_ylabel("Activated fraction")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, which="both", color="#d9d9d9", lw=0.45, alpha=0.65)
    ax.legend(loc="best")


def _finish_standalone(fig, ax, label: str, path) -> None:
    add_panel_label(ax, label)
    finish_panel(fig, path)


def save_standalone_panels(data) -> None:
    panel_specs = [
        ("a", lambda ax, fig: plot_activation_fraction(ax, data)),
        ("b", lambda ax, fig: plot_cloud_droplet_concentration(ax, data)),
        ("c", lambda ax, fig: plot_number_and_wet_size(ax, data)),
        ("d", lambda ax, fig: plot_activation_only_heatmap(ax, fig, data, colorbar=True)),
        (
            "e",
            lambda ax, fig: plot_activation_with_coagulation_heatmap(
                ax, fig, data, colorbar=True
            ),
        ),
        ("f", lambda ax, fig: plot_final_distribution_overlay(ax, data)),
        ("g", lambda ax, fig: plot_final_distribution_difference(ax, data)),
        ("h", lambda ax, fig: plot_size_resolved_activation(ax, data)),
    ]

    for label, plotter in panel_specs:
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        plotter(ax, fig)
        _finish_standalone(fig, ax, label, PANEL_PATHS[label])


def save_full_outputs(data) -> None:
    save_composite(data)
    save_standalone_panels(data)


def save_composite(data) -> None:
    fig = plt.figure(figsize=(7.4, 12.2), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 0.95, 1.1, 1.0, 0.95])

    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, :]),
        "d": fig.add_subplot(gs[2, 0]),
        "e": fig.add_subplot(gs[2, 1]),
        "f": fig.add_subplot(gs[3, 0]),
        "g": fig.add_subplot(gs[3, 1]),
        "h": fig.add_subplot(gs[4, :]),
    }

    plot_activation_fraction(axes["a"], data)
    plot_cloud_droplet_concentration(axes["b"], data)
    plot_number_and_wet_size(axes["c"], data)
    mesh = plot_activation_only_heatmap(axes["d"], fig, data, colorbar=False)
    plot_activation_with_coagulation_heatmap(axes["e"], fig, data, colorbar=False)
    cbar = fig.colorbar(mesh, ax=[axes["d"], axes["e"]], pad=0.015, fraction=0.035)
    cbar.set_label(KDE_DENSITY_LABEL)
    plot_final_distribution_overlay(axes["f"], data)
    plot_final_distribution_difference(axes["g"], data)
    plot_size_resolved_activation(axes["h"], data)

    for label, ax in axes.items():
        add_panel_label(ax, label)

    save_png(fig, COMPOSITE_PATH)
    plt.close(fig)


def main() -> None:
    apply_publication_style()
    data = prepare_scene()
    save_full_outputs(data)

    print(f"Saved composite PNG: {COMPOSITE_PATH}")
    print(f"Saved {len(PANEL_PATHS)} panel PNGs: {PANEL_DIR}")


if __name__ == "__main__":
    main()
