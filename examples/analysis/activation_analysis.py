"""Analysis helpers for activation and activation-coagulation examples."""

from __future__ import annotations

import numpy as np

from .smoothing import kde_log_diameter
from .stochparticles_io import (
    ReplicateData,
    clean_sample_matrix,
    geometric_centers,
    require_arrays,
    stack_common_time,
)


def activation_fraction(reps: list[ReplicateData]) -> tuple[np.ndarray, np.ndarray]:
    return stack_common_time(reps, "activation_fraction")


def cloud_droplet_concentration(reps: list[ReplicateData]) -> tuple[np.ndarray, np.ndarray]:
    return stack_common_time(reps, "cloud_droplet_concentration")


def _final_volume(rep: ReplicateData) -> float:
    volume = rep.arrays.get("volume")
    if volume is not None:
        values = np.asarray(volume, dtype=float).ravel()
        if values.size:
            return float(values[-1])
    return float(rep.attrs.get("volume", 1.0))


def final_distribution(reps: list[ReplicateData], grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    rows = []
    for rep in reps:
        require_arrays(rep, ("diameter_samples",))
        samples = clean_sample_matrix(rep.arrays["diameter_samples"])[-1]
        rows.append(kde_log_diameter(samples, grid, _final_volume(rep)))
    return np.vstack(rows) if rows else np.empty((0, grid.size), dtype=float)


def _bin_fraction(
    dry_diameter: np.ndarray, activated: np.ndarray, dry_grid_edges: np.ndarray
) -> np.ndarray:
    dry = np.asarray(dry_diameter, dtype=float).ravel()
    active = np.asarray(activated, dtype=float).ravel()
    valid = np.isfinite(dry) & (dry > 0.0) & np.isfinite(active)
    dry = dry[valid]
    active = active[valid] > 0.5
    counts, _ = np.histogram(dry, bins=dry_grid_edges)
    active_counts, _ = np.histogram(dry[active], bins=dry_grid_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            active_counts,
            counts,
            out=np.full(counts.shape, np.nan, dtype=float),
            where=counts > 0,
        )


def size_resolved_activation(
    reps: list[ReplicateData], dry_grid_edges: np.ndarray, activation_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    dry_grid_edges = np.asarray(dry_grid_edges, dtype=float)
    rows = []
    for rep in reps:
        try:
            require_arrays(rep, ("dry_diameter_samples", "activation_flag_samples"))
        except KeyError as exc:
            raise KeyError(
                f"{exc}; aligned final-time dry_diameter_samples and "
                "activation_flag_samples are required for size-resolved activation"
            ) from exc

        dry_matrix = np.asarray(rep.arrays["dry_diameter_samples"], dtype=float)
        flags = np.asarray(rep.arrays["activation_flag_samples"], dtype=float)
        if dry_matrix.ndim != 2 or flags.ndim != 2:
            raise ValueError(
                f"Expected aligned 2D dry/activation sample matrices for "
                f"case={rep.case_name!r}, replicate={rep.replicate_name!r}; "
                f"got dry_diameter_samples shape {dry_matrix.shape} and "
                f"activation_flag_samples shape {flags.shape}"
            )
        dry_samples = dry_matrix[-1, :]
        final_flags = flags[-1, :]
        if dry_samples.shape != final_flags.shape:
            raise ValueError(
                f"Aligned final-time dry/activation samples are required for "
                f"case={rep.case_name!r}, replicate={rep.replicate_name!r}; "
                f"got final dry shape {dry_samples.shape} and flag shape "
                f"{final_flags.shape}"
            )
        rows.append(_bin_fraction(dry_samples, final_flags, dry_grid_edges))
    matrix = np.vstack(rows) if rows else np.empty((0, dry_grid_edges.size - 1), dtype=float)
    return geometric_centers(dry_grid_edges), matrix
