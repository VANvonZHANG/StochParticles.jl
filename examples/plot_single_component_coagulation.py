#!/usr/bin/env python3
"""Generate publication-style plots for the single-component coagulation scene."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from analysis.coagulation_analysis import (
    kernel_fraction_series,
    kernel_total_bars,
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
    kde_log_diameter,
    linear_edges_from_centers,
    log_edges_from_centers,
    replicate_kde_heatmap,
    select_replicate_for_heatmap,
)
from analysis.stochparticles_io import (
    DATA_DIR,
    FIG_DIR,
    ReplicateData,
    clean_sample_matrix,
    read_scene,
)


SCENE_NAME = "single_component_coagulation"
DATA_PATH = DATA_DIR / f"{SCENE_NAME}.h5"
OUTPUT_ROOT = FIG_DIR
COMPOSITE_PATH = OUTPUT_ROOT / f"{SCENE_NAME}.png"
PANEL_DIR = OUTPUT_ROOT / SCENE_NAME

PANEL_PATHS = {
    "a": PANEL_DIR / "a_aerosol_number_decay.png",
    "b": PANEL_DIR / "b_aerosol_spectrum_heatmap.png",
    "c": PANEL_DIR / "c_aerosol_distribution_shift.png",
    "d": PANEL_DIR / "d_cloud_number_decay.png",
    "e": PANEL_DIR / "e_cloud_spectrum_heatmap.png",
    "f": PANEL_DIR / "f_cloud_kernel_fraction_timeseries.png",
    "g": PANEL_DIR / "g_cloud_kernel_total_bar.png",
}

KERNEL_COLORS = {
    "Brownian": PALETTE["brownian"],
    "Gravitational": PALETTE["gravitational"],
    "Turbulent": PALETTE["turbulent"],
}

KDE_MAX_BANDWIDTH_DEX = 0.06


def _positive_diameter_samples(reps: list[ReplicateData]) -> np.ndarray:
    samples = []
    for rep in reps:
        values = np.asarray(rep.arrays["diameter_samples"], dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            samples.append(values)
    if not samples:
        raise ValueError("no positive finite diameter samples found")
    return np.concatenate(samples)


def _diameter_grid(reps: list[ReplicateData], n: int = 260) -> np.ndarray:
    if reps and "bin_edges" in reps[0].arrays:
        edges = np.asarray(reps[0].arrays["bin_edges"], dtype=float)
        edges = edges[np.isfinite(edges) & (edges > 0.0)]
        if edges.size >= 2 and edges[-1] > edges[0]:
            return dense_log_grid(float(edges[0]), float(edges[-1]), n=n)
    samples = _positive_diameter_samples(reps)
    lo, hi = np.percentile(samples, [0.5, 99.5])
    lo = min(float(lo), float(np.min(samples)))
    hi = max(float(hi), float(np.max(samples)))
    return dense_log_grid(lo, hi, n=n)


def _replicate_volumes(rep: ReplicateData) -> np.ndarray:
    time = np.asarray(rep.arrays["time"], dtype=float)
    volume = np.asarray(rep.arrays.get("volume", rep.attrs.get("volume", 1.0)), dtype=float)
    if volume.ndim == 1 and volume.size == time.size:
        return volume
    if volume.size == 1:
        return np.full(time.size, float(volume.ravel()[0]))
    return np.full(time.size, float(rep.attrs.get("volume", 1.0)))


def _style_time_series_axis(ax, xlabel: str, ylabel: str, ylim: tuple[float, float] | None = None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#d9d9d9", lw=0.45, alpha=0.7)


def plot_number_decay(
    ax,
    reps: list[ReplicateData],
    *,
    color: str,
    label: str,
    time_scale: float,
    xlabel: str,
):
    time, normalized = normalized_number(reps)
    draw_replicates_with_mean(ax, time / time_scale, normalized, color, label)
    _style_time_series_axis(ax, xlabel, "N/N0", ylim=(0.0, 1.05))
    ax.legend(loc="best")


def plot_spectrum_heatmap(
    ax,
    fig,
    reps: list[ReplicateData],
    *,
    time_scale: float,
    xlabel: str,
    colorbar: bool = True,
):
    grid = _diameter_grid(reps)
    rep = select_replicate_for_heatmap(reps)
    time, diameter_grid, heatmap = replicate_kde_heatmap(
        rep,
        grid=grid,
        bandwidth_factor=1.1,
        max_bandwidth=KDE_MAX_BANDWIDTH_DEX,
        smooth_passes=1,
    )
    time_axis = time / time_scale
    diameter_um = diameter_grid * 1e6
    z = np.asarray(heatmap, dtype=float).T
    positive = z[np.isfinite(z) & (z > 0.0)]
    if positive.size:
        vmax = float(np.percentile(positive, 99.5))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.max(positive))
        norm = Normalize(vmin=0.0, vmax=vmax) if vmax > 0.0 else None
    else:
        norm = None

    cmap = plt.get_cmap("cividis").copy()
    mesh = ax.pcolormesh(
        linear_edges_from_centers(time_axis),
        log_edges_from_centers(diameter_um),
        np.ma.masked_invalid(z),
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(time_axis[0]), float(time_axis[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Diameter (um)")
    ax.grid(False)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("KDE number density (m$^{-3}$ dex$^{-1}$)")
    return mesh


def _distribution_shift_rows(
    reps: list[ReplicateData], grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    initial_rows = []
    final_rows = []
    for rep in reps:
        samples_by_time = clean_sample_matrix(rep.arrays["diameter_samples"])
        volumes = _replicate_volumes(rep)
        initial_rows.append(kde_log_diameter(samples_by_time[0], grid, volumes[0], 1.2))
        final_rows.append(kde_log_diameter(samples_by_time[-1], grid, volumes[-1], 1.2))
    return np.vstack(initial_rows), np.vstack(final_rows)


def plot_distribution_shift(ax, reps: list[ReplicateData]):
    grid = _diameter_grid(reps)
    initial_rows, final_rows = _distribution_shift_rows(reps, grid)
    diameter_um = grid * 1e6
    draw_replicates_with_mean(
        ax,
        diameter_um,
        initial_rows,
        PALETTE["neutral"],
        "Initial",
    )
    draw_replicates_with_mean(
        ax,
        diameter_um,
        final_rows,
        PALETTE["aerosol"],
        "Final",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Diameter (um)")
    ax.set_ylabel("KDE number density (m$^{-3}$)")
    ax.grid(True, which="both", color="#d9d9d9", lw=0.45, alpha=0.65)
    ax.legend(loc="best")


def plot_kernel_fraction_timeseries(ax, reps: list[ReplicateData]):
    time, series = kernel_fraction_series(reps)
    for label in ("Brownian", "Gravitational", "Turbulent"):
        draw_replicates_with_mean(
            ax,
            time,
            series[label],
            KERNEL_COLORS[label],
            label,
            lw_mean=1.65,
        )
    _style_time_series_axis(ax, "Time (s)", "Kernel fraction", ylim=(-0.02, 1.02))
    ax.legend(loc="best", ncols=1)


def plot_kernel_total_bar(ax, reps: list[ReplicateData]):
    totals = kernel_total_bars(reps)
    labels = ["Brownian", "Gravitational", "Turbulent"]
    x = np.arange(len(labels))
    rows = [np.asarray(totals[label], dtype=float) for label in labels]
    means = [float(np.nanmean(row)) for row in rows]
    colors = [KERNEL_COLORS[label] for label in labels]

    ax.bar(x, means, color=colors, alpha=0.82, width=0.62, edgecolor="white", linewidth=0.6)
    for idx, row in enumerate(rows):
        offsets = np.zeros(row.size) if row.size == 1 else np.linspace(-0.12, 0.12, row.size)
        ax.scatter(
            np.full(row.size, x[idx]) + offsets,
            row,
            s=14,
            facecolor="white",
            edgecolor=colors[idx],
            linewidth=0.75,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Time-averaged fraction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", color="#d9d9d9", lw=0.45, alpha=0.7)


def _finish_standalone(fig, ax, label: str, path):
    add_panel_label(ax, label)
    finish_panel(fig, path)


def save_standalone_panels(aerosol_reps: list[ReplicateData], cloud_reps: list[ReplicateData]):
    panel_specs = [
        (
            "a",
            lambda ax, fig: plot_number_decay(
                ax,
                aerosol_reps,
                color=PALETTE["aerosol"],
                label="Aerosol",
                time_scale=60.0,
                xlabel="Time (min)",
            ),
        ),
        (
            "b",
            lambda ax, fig: plot_spectrum_heatmap(
                ax,
                fig,
                aerosol_reps,
                time_scale=60.0,
                xlabel="Time (min)",
            ),
        ),
        ("c", lambda ax, fig: plot_distribution_shift(ax, aerosol_reps)),
        (
            "d",
            lambda ax, fig: plot_number_decay(
                ax,
                cloud_reps,
                color=PALETTE["cloud"],
                label="Cloud",
                time_scale=1.0,
                xlabel="Time (s)",
            ),
        ),
        (
            "e",
            lambda ax, fig: plot_spectrum_heatmap(
                ax,
                fig,
                cloud_reps,
                time_scale=1.0,
                xlabel="Time (s)",
            ),
        ),
        ("f", lambda ax, fig: plot_kernel_fraction_timeseries(ax, cloud_reps)),
        ("g", lambda ax, fig: plot_kernel_total_bar(ax, cloud_reps)),
    ]

    for label, plotter in panel_specs:
        fig, ax = plt.subplots(figsize=(3.35, 2.45))
        plotter(ax, fig)
        _finish_standalone(fig, ax, label, PANEL_PATHS[label])


def save_composite(aerosol_reps: list[ReplicateData], cloud_reps: list[ReplicateData]):
    fig = plt.figure(figsize=(7.2, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.15, 1.0, 1.0])

    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, :]),
        "d": fig.add_subplot(gs[2, 0]),
        "e": fig.add_subplot(gs[2, 1]),
        "f": fig.add_subplot(gs[3, 0]),
        "g": fig.add_subplot(gs[3, 1]),
    }

    plot_number_decay(
        axes["a"],
        aerosol_reps,
        color=PALETTE["aerosol"],
        label="Aerosol",
        time_scale=60.0,
        xlabel="Time (min)",
    )
    plot_spectrum_heatmap(
        axes["b"],
        fig,
        aerosol_reps,
        time_scale=60.0,
        xlabel="Time (min)",
    )
    plot_distribution_shift(axes["c"], aerosol_reps)
    plot_number_decay(
        axes["d"],
        cloud_reps,
        color=PALETTE["cloud"],
        label="Cloud",
        time_scale=1.0,
        xlabel="Time (s)",
    )
    plot_spectrum_heatmap(
        axes["e"],
        fig,
        cloud_reps,
        time_scale=1.0,
        xlabel="Time (s)",
    )
    plot_kernel_fraction_timeseries(axes["f"], cloud_reps)
    plot_kernel_total_bar(axes["g"], cloud_reps)

    for label, ax in axes.items():
        add_panel_label(ax, label)

    save_png(fig, COMPOSITE_PATH)
    plt.close(fig)


def load_scene_cases() -> tuple[list[ReplicateData], list[ReplicateData]]:
    scene = read_scene(DATA_PATH)
    try:
        return scene["aerosol_brownian"], scene["cloud_composite"]
    except KeyError as exc:
        available = ", ".join(sorted(scene))
        raise KeyError(f"required case missing from {DATA_PATH}; available cases: {available}") from exc


def main() -> None:
    apply_publication_style()
    aerosol_reps, cloud_reps = load_scene_cases()
    save_composite(aerosol_reps, cloud_reps)
    save_standalone_panels(aerosol_reps, cloud_reps)

    print(f"Saved composite PNG: {COMPOSITE_PATH}")
    print(f"Saved {len(PANEL_PATHS)} panel PNGs: {PANEL_DIR}")


if __name__ == "__main__":
    main()
