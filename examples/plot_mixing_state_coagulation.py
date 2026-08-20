#!/usr/bin/env python3
"""Generate publication-style plots for the mixing-state coagulation scene."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

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
from analysis.smoothing import (
    dense_log_grid,
    linear_edges_from_centers,
    log_edges_from_centers,
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
    "e": PANEL_DIR / "e_size_composition_map_initial.png",
    "f": PANEL_DIR / "f_size_composition_map_final.png",
}

FRACTION_BINS = np.linspace(0.0, 1.0, 41)
FRACTION_KDE_GRID = np.linspace(0.0, 1.0, 101)
FRACTION_KDE_BANDWIDTH = 0.025
DIAMETER_GRID_POINTS = 240
DIAMETER_GRID_PAD_DEX = 0.05
KDE_MAX_BANDWIDTH_DEX = 0.06


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


def _volume_at_row(rep: ReplicateData, row_index: int) -> float:
    time = np.asarray(rep.arrays["time"], dtype=float)
    values = np.asarray(
        rep.arrays.get("volume", rep.attrs.get("volume", 1.0)), dtype=float
    )
    if values.ndim == 1 and values.size == time.size:
        return float(values[row_index])
    return float(values.ravel()[0])


def _size_fraction_rows(reps: list[ReplicateData], row_index: int) -> list[tuple]:
    """Per-replicate (diameters, BC fractions, volume) samples at one time row."""

    rows = []
    for rep in reps:
        diameter_values = np.asarray(rep.arrays["diameter_samples"], dtype=float)
        fraction_values = np.asarray(rep.arrays["bc_mass_fraction_samples"], dtype=float)
        diameter_row = diameter_values[row_index, :] if diameter_values.ndim == 2 else diameter_values
        fraction_row = fraction_values[row_index, :] if fraction_values.ndim == 2 else fraction_values
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
            rows.append(
                (diameter_row[mask], fraction_row[mask], _volume_at_row(rep, row_index))
            )
    if not rows:
        raise ValueError(f"no valid diameter/BC-fraction samples found at time row {row_index}")
    return rows


def _diameter_kde_grid(rows_per_time: list[list[tuple]]) -> np.ndarray:
    pooled = np.concatenate([diameters for rows in rows_per_time for diameters, _, _ in rows])
    lo, hi = float(np.min(pooled)), float(np.max(pooled))
    return dense_log_grid(
        10.0 ** (np.log10(lo) - DIAMETER_GRID_PAD_DEX),
        10.0 ** (np.log10(hi) + DIAMETER_GRID_PAD_DEX),
        n=DIAMETER_GRID_POINTS,
    )


def _size_fraction_kde(
    rows: list[tuple], diameter_grid: np.ndarray, fraction_grid: np.ndarray
) -> np.ndarray:
    """Gaussian KDE of pooled (log10 diameter, BC fraction) samples.

    The log-diameter bandwidth follows Silverman's rule capped at
    KDE_MAX_BANDWIDTH_DEX, mirroring the single-component spectrum heatmaps,
    while the fraction bandwidth is fixed. Values are number densities in
    m^-3 dex^-1 (the fraction axis is dimensionless).
    """

    log_grid = np.log10(diameter_grid)
    density = np.zeros((diameter_grid.size, fraction_grid.size), dtype=float)
    for diameters, fractions, volume in rows:
        log_samples = np.log10(diameters)
        bandwidth = 1.06 * np.std(log_samples, ddof=1) * log_samples.size ** (-1.0 / 5.0)
        if log_samples.size == 1 or not np.isfinite(bandwidth) or bandwidth <= 0.0:
            bandwidth = 0.05
        bandwidth = min(bandwidth * 1.1, KDE_MAX_BANDWIDTH_DEX)

        z_d = (log_grid[:, None] - log_samples[None, :]) / bandwidth
        kernel_d = np.exp(-0.5 * z_d * z_d) / (np.sqrt(2.0 * np.pi) * bandwidth)
        z_f = (fraction_grid[None, :] - fractions[:, None]) / FRACTION_KDE_BANDWIDTH
        kernel_f = np.exp(-0.5 * z_f * z_f) / (np.sqrt(2.0 * np.pi) * FRACTION_KDE_BANDWIDTH)
        density += (kernel_d / volume) @ kernel_f
    return density


def _density_norm(density: np.ndarray) -> Normalize:
    positive = density[np.isfinite(density) & (density > 0.0)]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return Normalize(vmin=0.0, vmax=vmax)


def _size_composition_map_data(reps: list[ReplicateData]):
    """Initial and final size-composition KDE maps on a shared grid, each with its own scale."""

    initial_rows = _size_fraction_rows(reps, 0)
    final_rows = _size_fraction_rows(reps, -1)
    diameter_grid = _diameter_kde_grid([initial_rows, final_rows])
    fraction_grid = FRACTION_KDE_GRID
    initial_density = _size_fraction_kde(initial_rows, diameter_grid, fraction_grid)
    final_density = _size_fraction_kde(final_rows, diameter_grid, fraction_grid)
    return (
        diameter_grid,
        fraction_grid,
        initial_density,
        _density_norm(initial_density),
        final_density,
        _density_norm(final_density),
    )


def _row_time_hours(reps: list[ReplicateData], row_index: int) -> float:
    time = np.asarray(reps[0].arrays["time"], dtype=float)
    return float(time[row_index]) / 3600.0


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


def plot_size_composition_map(
    ax,
    fig,
    density: np.ndarray,
    diameter_grid: np.ndarray,
    fraction_grid: np.ndarray,
    *,
    title: str,
    norm: Normalize,
    colorbar: bool = True,
):
    mesh = ax.pcolormesh(
        log_edges_from_centers(diameter_grid * 1e6),
        linear_edges_from_centers(fraction_grid),
        np.ma.masked_invalid(density.T),
        shading="flat",
        cmap=plt.get_cmap("cividis"),
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Diameter (um)")
    ax.set_ylabel("BC mass fraction")
    ax.set_title(title, pad=4)
    ax.grid(False)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("KDE number density (m$^{-3}$ dex$^{-1}$)")
    return mesh


def _finish_standalone(fig, ax, label: str, path) -> None:
    add_panel_label(ax, label)
    finish_panel(fig, path)


def save_standalone_panels(reps: list[ReplicateData]) -> None:
    diameter_grid, fraction_grid, initial_density, initial_norm, final_density, final_norm = (
        _size_composition_map_data(reps)
    )
    initial_title = f"Initial (t = {_row_time_hours(reps, 0):g} h)"
    final_title = f"Final (t = {_row_time_hours(reps, -1):g} h)"
    panel_specs = [
        ("a", lambda ax, fig: plot_mixing_state_index(ax, reps), (3.45, 2.45)),
        ("b", lambda ax, fig: plot_number_decay(ax, reps), (3.45, 2.45)),
        ("c", lambda ax, fig: plot_species_mass_conservation(ax, reps), (3.45, 2.45)),
        ("d", lambda ax, fig: plot_composition_distribution(ax, reps), (3.45, 2.45)),
        (
            "e",
            lambda ax, fig: plot_size_composition_map(
                ax,
                fig,
                initial_density,
                diameter_grid,
                fraction_grid,
                title=initial_title,
                norm=initial_norm,
            ),
            (3.75, 2.65),
        ),
        (
            "f",
            lambda ax, fig: plot_size_composition_map(
                ax,
                fig,
                final_density,
                diameter_grid,
                fraction_grid,
                title=final_title,
                norm=final_norm,
            ),
            (3.75, 2.65),
        ),
    ]

    for label, plotter, figsize in panel_specs:
        fig, ax = plt.subplots(figsize=figsize)
        plotter(ax, fig)
        _finish_standalone(fig, ax, label, PANEL_PATHS[label])


def save_composite(reps: list[ReplicateData]) -> None:
    fig = plt.figure(figsize=(7.3, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15])

    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, 0]),
        "d": fig.add_subplot(gs[1, 1]),
        "e": fig.add_subplot(gs[2, 0]),
        "f": fig.add_subplot(gs[2, 1]),
    }

    plot_mixing_state_index(axes["a"], reps)
    plot_number_decay(axes["b"], reps)
    plot_species_mass_conservation(axes["c"], reps)
    plot_composition_distribution(axes["d"], reps)

    diameter_grid, fraction_grid, initial_density, initial_norm, final_density, final_norm = (
        _size_composition_map_data(reps)
    )
    plot_size_composition_map(
        axes["e"],
        fig,
        initial_density,
        diameter_grid,
        fraction_grid,
        title=f"Initial (t = {_row_time_hours(reps, 0):g} h)",
        norm=initial_norm,
    )
    plot_size_composition_map(
        axes["f"],
        fig,
        final_density,
        diameter_grid,
        fraction_grid,
        title=f"Final (t = {_row_time_hours(reps, -1):g} h)",
        norm=final_norm,
    )

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
