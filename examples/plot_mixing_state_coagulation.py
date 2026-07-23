#!/usr/bin/env python3
"""Generate publication-style plots for the mixing-state coagulation scene."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from analysis.coagulation_analysis import normalized_number
from analysis.figure_style import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    draw_replicates_with_mean,
    finish_panel,
    save_png,
)
from analysis.mixing_state_analysis import (
    final_bc_fraction,
    initial_bc_fraction,
    mixing_index,
    normalized_species_mass,
)
from analysis.stochparticles_io import DATA_DIR, FIG_DIR, ReplicateData, read_scene


SCENE_NAME = "mixing_state_coagulation"
DATA_PATH = DATA_DIR / f"{SCENE_NAME}.h5"
COMPOSITE_PATH = FIG_DIR / f"{SCENE_NAME}.png"
PANEL_DIR = FIG_DIR / SCENE_NAME

PANEL_PATHS = {
    "a": PANEL_DIR / "a_mixing_state_index.png",
    "b": PANEL_DIR / "b_number_decay.png",
    "c": PANEL_DIR / "c_species_mass_conservation.png",
    "d": PANEL_DIR / "d_composition_distribution.png",
    "e": PANEL_DIR / "e_size_composition_map.png",
}

FRACTION_BINS = np.linspace(0.0, 1.0, 41)
FRACTION_HEATMAP_BINS = np.linspace(0.0, 1.0, 101)


def _time_minutes(time: np.ndarray) -> np.ndarray:
    return np.asarray(time, dtype=float) / 60.0


def _style_time_axis(ax, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#d9d9d9", lw=0.45, alpha=0.7)


def _draw_time_replicates(
    ax,
    time: np.ndarray,
    rows: np.ndarray,
    color: str,
    label: str,
    *,
    lw_mean: float = 1.8,
) -> np.ndarray:
    return draw_replicates_with_mean(
        ax,
        _time_minutes(time),
        rows,
        color,
        label,
        lw_mean=lw_mean,
    )


def _species_names(reps: list[ReplicateData]) -> list[str]:
    if not reps:
        return []
    names = reps[0].attrs.get("species_names", [])
    if isinstance(names, str):
        return [names]
    return [str(name) for name in names]


def _species_index(names: list[str], target: str, fallback: int) -> int:
    lowered = [name.lower() for name in names]
    try:
        return lowered.index(target.lower())
    except ValueError:
        return fallback


def _species_mass_ylim(rows: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(rows, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.995, 1.005
    lo = min(float(np.min(finite)), 1.0)
    hi = max(float(np.max(finite)), 1.0)
    span = hi - lo
    pad = max(span * 0.35, 1.0e-3)
    return lo - pad, hi + pad


def _fraction_histogram_rows(
    fraction_rows: list[np.ndarray], bins: np.ndarray = FRACTION_BINS
) -> tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    widths = np.diff(bins)
    for fractions in fraction_rows:
        values = np.asarray(fractions, dtype=float)
        values = values[np.isfinite(values) & (values >= 0.0) & (values <= 1.0)]
        if values.size == 0:
            rows.append(np.full(centers.size, np.nan))
            continue
        counts, _ = np.histogram(values, bins=bins)
        density = counts / (np.sum(counts) * widths)
        rows.append(density)
    return centers, np.vstack(rows) if rows else np.empty((0, centers.size))


def _final_size_fraction_pairs(reps: list[ReplicateData]) -> tuple[np.ndarray, np.ndarray]:
    diameters = []
    fractions = []
    for rep in reps:
        diameter_values = np.asarray(rep.arrays["diameter_samples"], dtype=float)
        fraction_values = np.asarray(rep.arrays["bc_mass_fraction_samples"], dtype=float)
        diameter_row = diameter_values[-1, :] if diameter_values.ndim == 2 else diameter_values
        fraction_row = fraction_values[-1, :] if fraction_values.ndim == 2 else fraction_values
        n = min(diameter_row.size, fraction_row.size)
        diameter_row = np.ravel(diameter_row)[:n]
        fraction_row = np.ravel(fraction_row)[:n]
        mask = (
            np.isfinite(diameter_row)
            & (diameter_row > 0.0)
            & np.isfinite(fraction_row)
            & (fraction_row >= 0.0)
            & (fraction_row <= 1.0)
        )
        if np.any(mask):
            diameters.append(diameter_row[mask])
            fractions.append(fraction_row[mask])
    if not diameters:
        raise ValueError("no valid final diameter/BC-fraction samples found")
    return np.concatenate(diameters), np.concatenate(fractions)


def _log_edges(values: np.ndarray, n: int = 100) -> np.ndarray:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("cannot build log bin edges without positive values")
    lo, hi = np.percentile(positive, [0.5, 99.5])
    lo = min(float(lo), float(np.min(positive)))
    hi = max(float(hi), float(np.max(positive)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(positive)) * 0.9
        hi = float(np.max(positive)) * 1.1
    return np.geomspace(lo, hi, n)


def _count_norm(counts: np.ndarray) -> LogNorm | Normalize:
    positive = counts[np.isfinite(counts) & (counts > 0.0)]
    if positive.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmax = float(np.max(positive))
    if vmax > 1.0:
        vmin = max(float(np.percentile(positive, 2.0)), np.finfo(float).tiny)
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=0.0, vmax=1.0)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / np.sum(kernel)


def _smooth_2d_counts(
    counts: np.ndarray, sigma_x: float = 1.1, sigma_y: float = 0.75
) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    kernel_x = _gaussian_kernel1d(sigma_x)
    kernel_y = _gaussian_kernel1d(sigma_y)
    radius_x = kernel_x.size // 2
    radius_y = kernel_y.size // 2

    padded_x = np.pad(values, ((radius_x, radius_x), (0, 0)), mode="edge")
    smoothed_x = np.empty_like(values, dtype=float)
    for i in range(values.shape[0]):
        smoothed_x[i, :] = np.sum(
            padded_x[i : i + kernel_x.size, :] * kernel_x[:, None], axis=0
        )

    padded_y = np.pad(smoothed_x, ((0, 0), (radius_y, radius_y)), mode="edge")
    smoothed = np.empty_like(values, dtype=float)
    for j in range(values.shape[1]):
        smoothed[:, j] = np.sum(
            padded_y[:, j : j + kernel_y.size] * kernel_y[None, :], axis=1
        )
    return smoothed


def load_replicates() -> list[ReplicateData]:
    scene = read_scene(DATA_PATH)
    try:
        return scene[SCENE_NAME]
    except KeyError as exc:
        available = ", ".join(sorted(scene))
        raise KeyError(f"required case missing from {DATA_PATH}; available cases: {available}") from exc


def plot_mixing_state_index(ax, reps: list[ReplicateData]) -> None:
    time, rows = mixing_index(reps)
    _draw_time_replicates(
        ax,
        time,
        rows,
        PALETTE["activation_with_coagulation"],
        "Mean",
        lw_mean=1.9,
    )
    _style_time_axis(ax, "Mixing state index $\\chi$")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")


def plot_number_decay(ax, reps: list[ReplicateData]) -> None:
    time, rows = normalized_number(reps)
    _draw_time_replicates(ax, time, rows, PALETTE["aerosol"], "Mean", lw_mean=1.9)
    _style_time_axis(ax, "N/N0", ylim=(0.0, 1.02))
    ax.legend(loc="best")


def plot_species_mass_conservation(ax, reps: list[ReplicateData]) -> None:
    time, rows = normalized_species_mass(reps)
    names = _species_names(reps)
    so4_idx = _species_index(names, "SO4", 0)
    bc_idx = _species_index(names, "BC", 1 if rows.shape[-1] > 1 else 0)
    ax.axhline(1.0, color="#555555", lw=0.7, alpha=0.8)
    _draw_time_replicates(
        ax,
        time,
        rows[:, :, so4_idx],
        PALETTE["so4"],
        "SO4",
        lw_mean=1.75,
    )
    _draw_time_replicates(
        ax,
        time,
        rows[:, :, bc_idx],
        PALETTE["bc"],
        "BC",
        lw_mean=1.75,
    )
    _style_time_axis(ax, "Mass / initial mass", ylim=_species_mass_ylim(rows))
    ax.legend(loc="best", ncols=2)


def plot_composition_distribution(ax, reps: list[ReplicateData]) -> None:
    centers, initial_rows = _fraction_histogram_rows(initial_bc_fraction(reps))
    _, final_rows = _fraction_histogram_rows(final_bc_fraction(reps))
    line_start = len(ax.lines)
    draw_replicates_with_mean(
        ax,
        centers,
        initial_rows,
        PALETTE["neutral"],
        "Initial",
        lw_mean=1.75,
    )
    for line in ax.lines[line_start:]:
        line.set_linestyle("--")
    draw_replicates_with_mean(
        ax,
        centers,
        final_rows,
        PALETTE["activation_with_coagulation"],
        "Final",
        lw_mean=1.75,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("BC mass fraction")
    ax.set_ylabel("Probability density")
    ax.grid(True, color="#d9d9d9", lw=0.45, alpha=0.7)
    ax.legend(loc="best")


def plot_size_composition_map(ax, fig, reps: list[ReplicateData], *, colorbar: bool = True):
    diameters, fractions = _final_size_fraction_pairs(reps)
    diameter_edges = _log_edges(diameters)
    counts, x_edges, y_edges = np.histogram2d(
        diameters * 1e6,
        fractions,
        bins=[diameter_edges * 1e6, FRACTION_HEATMAP_BINS],
    )
    smoothed_counts = _smooth_2d_counts(counts)
    smoothed_counts[smoothed_counts < 0.05] = np.nan
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(color="white")
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(smoothed_counts.T),
        cmap=cmap,
        norm=_count_norm(smoothed_counts),
        shading="flat",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Final diameter (um)")
    ax.set_ylabel("BC mass fraction")
    ax.grid(False)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Smoothed particle count")
    return mesh


def _finish_standalone(fig, ax, label: str, path) -> None:
    add_panel_label(ax, label)
    finish_panel(fig, path)


def save_standalone_panels(reps: list[ReplicateData]) -> None:
    panel_specs = [
        ("a", lambda ax, fig: plot_mixing_state_index(ax, reps), (3.45, 2.45)),
        ("b", lambda ax, fig: plot_number_decay(ax, reps), (3.45, 2.45)),
        ("c", lambda ax, fig: plot_species_mass_conservation(ax, reps), (3.45, 2.45)),
        ("d", lambda ax, fig: plot_composition_distribution(ax, reps), (3.45, 2.45)),
        ("e", lambda ax, fig: plot_size_composition_map(ax, fig, reps), (3.75, 2.65)),
    ]

    for label, plotter, figsize in panel_specs:
        fig, ax = plt.subplots(figsize=figsize)
        plotter(ax, fig)
        _finish_standalone(fig, ax, label, PANEL_PATHS[label])


def save_composite(reps: list[ReplicateData]) -> None:
    fig = plt.figure(figsize=(7.3, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2])

    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, 0]),
        "d": fig.add_subplot(gs[1, 1]),
        "e": fig.add_subplot(gs[2, :]),
    }

    plot_mixing_state_index(axes["a"], reps)
    plot_number_decay(axes["b"], reps)
    plot_species_mass_conservation(axes["c"], reps)
    plot_composition_distribution(axes["d"], reps)
    plot_size_composition_map(axes["e"], fig, reps, colorbar=True)

    for label, ax in axes.items():
        add_panel_label(ax, label)

    save_png(fig, COMPOSITE_PATH)
    plt.close(fig)


def main() -> None:
    apply_publication_style()
    reps = load_replicates()
    save_composite(reps)
    save_standalone_panels(reps)

    print(f"Saved composite PNG: {COMPOSITE_PATH}")
    print(f"Saved {len(PANEL_PATHS)} panel PNGs: {PANEL_DIR}")


if __name__ == "__main__":
    main()
