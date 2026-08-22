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
from matplotlib.lines import Line2D

from analysis.activation_analysis import (
    activation_fraction,
    cloud_droplet_concentration,
    final_distribution,
    size_resolved_activation,
)
from analysis.coagulation_analysis import (
    dataset_series,
    normalized_diameter,
    normalized_number,
)
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
    "a": PANEL_DIR / "a_mode_number.png",
    "b": PANEL_DIR / "b_mean_wet_diameter.png",
    "c": PANEL_DIR / "c_kernel_attribution_aitken.png",
    "d": PANEL_DIR / "d_kernel_attribution_droplet.png",
    "e": PANEL_DIR / "e_kernel_attribution_cross.png",
    "f": PANEL_DIR / "f_size_resolved_activation.png",
    "g": PANEL_DIR / "g_with_coagulation_heatmap.png",
    "h": PANEL_DIR / "h_activation_only_heatmap.png",
}

CASE_LABELS = {
    "activation_only": "Activation only",
    "activation_with_coagulation": "With coagulation",
}

CASE_COLORS = {
    "activation_only": PALETTE["activation_only"],
    "activation_with_coagulation": PALETTE["activation_with_coagulation"],
}

MODE_LABELS = {"aitken": "Aitken (20 nm)", "droplet": "Droplet (200 nm)", "cross": "Aitken-droplet"}
MODE_COLORS = {"aitken": PALETTE["aerosol"], "droplet": PALETTE["cloud"]}
MODE_SERIES = {
    "aitken": "number_concentration_aitken",
    "droplet": "number_concentration_droplet",
}
MODE_ACTIVATION = {
    "aitken": "activation_fraction_aitken",
    "droplet": "activation_fraction_droplet",
}
ATTRIBUTION_COMPONENTS = (
    ("brownian", "Brownian"),
    ("gravitational", "Gravitational"),
    ("turbulent", "Turbulent"),
)

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
    wet_grid = _log_grid_from_samples(wet_samples, n=280)
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


def _mode_case_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color=MODE_COLORS["aitken"], lw=1.8, label="Aitken (20 nm)"),
        Line2D([0], [0], color=MODE_COLORS["droplet"], lw=1.8, label="Droplet (200 nm)"),
        Line2D([0], [0], color="0.25", lw=1.8, ls="-", label="With coagulation"),
        Line2D([0], [0], color="0.25", lw=1.8, ls="--", label="Activation only"),
    ]
    ax.legend(handles=handles, fontsize=6, ncols=2, loc="best", framealpha=0.85)


def plot_mode_number(ax, data) -> None:
    for mode, series in MODE_SERIES.items():
        for case in ("activation_only", "activation_with_coagulation"):
            time, rows = dataset_series(data[case], series)
            normalized = rows / rows[:, :1]
            linestyle = "-" if case == "activation_with_coagulation" else "--"
            _draw_replicates(
                ax,
                _time_minutes(time),
                normalized,
                MODE_COLORS[mode],
                None,
                linestyle=linestyle,
                lw_mean=1.75,
            )
    _style_time_axis(ax, "N / N$_0$", ylim=(0.0, 1.08))
    _mode_case_legend(ax)


def plot_mean_wet_diameter(ax, data) -> None:
    for case in ("activation_only", "activation_with_coagulation"):
        time, rows = dataset_series(data[case], "mean_diameter")
        _draw_replicates(
            ax,
            _time_minutes(time),
            rows * 1e6,
            CASE_COLORS[case],
            CASE_LABELS[case],
            lw_mean=1.75,
        )
    ax.set_yscale("log")
    _style_time_axis(ax, "Mean wet diameter ($\mu$m)")
    ax.legend(loc="best", fontsize=6)


def plot_kernel_attribution(ax, data, mode_class: str) -> None:
    case = "activation_with_coagulation"
    time, _ = dataset_series(data[case], f"kernel_fraction_{mode_class}_brownian")
    stack = np.vstack([
        np.nanmean(
            dataset_series(data[case], f"kernel_fraction_{mode_class}_{comp}")[1],
            axis=0,
        )
        for comp, _ in ATTRIBUTION_COMPONENTS
    ])
    ax.stackplot(
        _time_minutes(time),
        stack,
        colors=[PALETTE[comp] for comp, _ in ATTRIBUTION_COMPONENTS],
        labels=[label for _, label in ATTRIBUTION_COMPONENTS],
        alpha=0.85,
    )
    total = np.vstack([
        np.asarray(rep.arrays["number_concentration_aitken"], dtype=float) +
        np.asarray(rep.arrays["number_concentration_droplet"], dtype=float)
        for rep in data[case]
    ])
    active = np.nanmean(total / total[:, :1], axis=0) >= 0.05
    t_max = float(time[np.max(np.nonzero(active)[0])]) if active.any() else float(time[-1])
    _style_time_axis(
        ax, f"Kernel share ({MODE_LABELS[mode_class]} pairs)", ylim=(0.0, 1.0)
    )
    ax.set_xlim(0.0, max(t_max / 60.0, 5.0))
    ax.legend(loc="center right", fontsize=6)


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
        ("a", lambda ax, fig: plot_mode_number(ax, data)),
        ("b", lambda ax, fig: plot_mean_wet_diameter(ax, data)),
        ("c", lambda ax, fig: plot_kernel_attribution(ax, data, "aitken")),
        ("d", lambda ax, fig: plot_kernel_attribution(ax, data, "droplet")),
        ("e", lambda ax, fig: plot_kernel_attribution(ax, data, "cross")),
        ("f", lambda ax, fig: plot_size_resolved_activation(ax, data)),
        (
            "g",
            lambda ax, fig: plot_activation_with_coagulation_heatmap(
                ax, fig, data, colorbar=True
            ),
        ),
        ("h", lambda ax, fig: plot_activation_only_heatmap(ax, fig, data, colorbar=True)),
    ]

    for label, plotter in panel_specs:
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        plotter(ax, fig)
        _finish_standalone(fig, ax, label, PANEL_PATHS[label])


def save_full_outputs(data) -> None:
    save_composite(data)
    save_standalone_panels(data)


def save_composite(data) -> None:
    fig = plt.figure(figsize=(10.0, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.15])
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, 0]),
        "d": fig.add_subplot(gs[1, 1]),
        "e": fig.add_subplot(gs[2, 0]),
        "f": fig.add_subplot(gs[2, 1]),
        "g": fig.add_subplot(gs[3, 0]),
        "h": fig.add_subplot(gs[3, 1]),
    }
    plot_mode_number(axes["a"], data)
    plot_mean_wet_diameter(axes["b"], data)
    plot_kernel_attribution(axes["c"], data, "aitken")
    plot_kernel_attribution(axes["d"], data, "droplet")
    plot_kernel_attribution(axes["e"], data, "cross")
    plot_size_resolved_activation(axes["f"], data)
    mesh = plot_activation_with_coagulation_heatmap(axes["g"], fig, data, colorbar=False)
    plot_activation_only_heatmap(axes["h"], fig, data, colorbar=False)
    cbar = fig.colorbar(mesh, ax=[axes["g"], axes["h"]], pad=0.015, fraction=0.035)
    cbar.set_label(KDE_DENSITY_LABEL)
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
