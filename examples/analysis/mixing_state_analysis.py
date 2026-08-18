"""Analysis helpers for mixing-state example plots."""

from __future__ import annotations

import numpy as np

from .stochparticles_io import (
    ReplicateData,
    require_arrays,
    stack_common_time,
)


def mixing_index(reps: list[ReplicateData]) -> tuple[np.ndarray, np.ndarray]:
    return stack_common_time(reps, "mixing_state_index")


def normalized_species_mass(reps: list[ReplicateData]) -> tuple[np.ndarray, np.ndarray]:
    if not reps:
        return np.array([], dtype=float), np.empty((0, 0, 0), dtype=float)

    require_arrays(reps[0], ("time", "species_mass_concentration"))
    common_time = np.asarray(reps[0].arrays["time"], dtype=float)
    rows = []
    for rep in reps:
        require_arrays(rep, ("time", "species_mass_concentration"))
        time = np.asarray(rep.arrays["time"], dtype=float)
        mass = np.asarray(rep.arrays["species_mass_concentration"], dtype=float)
        if mass.ndim != 2:
            raise ValueError(
                f"Expected 2D species_mass_concentration for case={rep.case_name!r}, "
                f"replicate={rep.replicate_name!r}; got shape {mass.shape}"
            )
        if not np.array_equal(time, common_time):
            interp_mass = np.empty((common_time.size, mass.shape[1]), dtype=float)
            for species_idx in range(mass.shape[1]):
                interp_mass[:, species_idx] = np.interp(
                    common_time, time, mass[:, species_idx]
                )
            mass = interp_mass
        first = mass[:1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                mass, first, out=np.full_like(mass, np.nan), where=first != 0.0
            )
        rows.append(normalized)
    return common_time, np.stack(rows, axis=0)


def _clean_fraction_row(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=float).ravel()
    return values[np.isfinite(values) & (values >= 0.0) & (values <= 1.0)]


def initial_bc_fraction(reps: list[ReplicateData]) -> list[np.ndarray]:
    arrays = []
    for rep in reps:
        require_arrays(rep, ("bc_mass_fraction_samples",))
        values = np.asarray(rep.arrays["bc_mass_fraction_samples"], dtype=float)
        row = values[0, :] if values.ndim == 2 else values
        arrays.append(_clean_fraction_row(row))
    return arrays


def final_bc_fraction(reps: list[ReplicateData]) -> list[np.ndarray]:
    arrays = []
    for rep in reps:
        require_arrays(rep, ("bc_mass_fraction_samples",))
        values = np.asarray(rep.arrays["bc_mass_fraction_samples"], dtype=float)
        row = values[-1, :] if values.ndim == 2 else values
        arrays.append(_clean_fraction_row(row))
    return arrays
