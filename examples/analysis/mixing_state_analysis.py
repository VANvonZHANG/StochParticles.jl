"""Analysis helpers for mixing-state example plots."""

from __future__ import annotations

import numpy as np
import PyMieScatt

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


SPECIES_DENSITIES = (1770.0, 1800.0)  # (SO4, BC) kg/m³ — mirrors StochParticles.AS/BC presets
SPECIES_KAPPA = (0.61, 0.0)  # Petters & Kreidenweis (2007) — mirrors StochParticles.AS/BC


def particle_species_volumes(
    diameters: np.ndarray,
    bc_fractions: np.ndarray,
    densities: tuple[float, float] = SPECIES_DENSITIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-particle (SO4, BC) dry volumes [m³] from dry diameter and BC mass fraction.

    Volume-additive inversion: m_tot = V_dry / Σ_s f_s/ρ_s, v_s = m_tot·f_s/ρ_s.
    """

    rho_so4, rho_bc = densities
    f_bc = np.asarray(bc_fractions, dtype=float)
    f_so4 = 1.0 - f_bc
    v_dry = np.pi * np.asarray(diameters, dtype=float) ** 3 / 6.0
    m_tot = v_dry / (f_so4 / rho_so4 + f_bc / rho_bc)
    return m_tot * f_so4 / rho_so4, m_tot * f_bc / rho_bc


def particle_kappas(
    diameters: np.ndarray,
    bc_fractions: np.ndarray,
    kappa: tuple[float, float] = SPECIES_KAPPA,
    densities: tuple[float, float] = SPECIES_DENSITIES,
) -> np.ndarray:
    """Volume-weighted hygroscopicity per particle (mirrors StochParticles.water_activity)."""

    v_so4, v_bc = particle_species_volumes(diameters, bc_fractions, densities)
    return (kappa[0] * v_so4 + kappa[1] * v_bc) / (v_so4 + v_bc)


KOEHLER_PARAMS = {"sigma": 0.072, "rho_w": 1000.0, "R_v": 461.5}  # mirrors ThermodynamicsParams
TEMPERATURE = 298.0


def _kohler_supersaturation(radius, v_dry, kappa_mix, temperature):
    v_w = (4.0 / 3.0) * np.pi * radius**3 - v_dry
    with np.errstate(divide="ignore", invalid="ignore"):
        a_w = v_w / (v_w + kappa_mix * v_dry)
        kelvin = np.exp(
            2.0 * KOEHLER_PARAMS["sigma"]
            / (KOEHLER_PARAMS["R_v"] * temperature * KOEHLER_PARAMS["rho_w"] * radius)
        )
        return np.where(v_w > 0.0, a_w * kelvin - 1.0, -1.0)


def critical_supersaturations(
    dry_diameters: np.ndarray,
    kappas: np.ndarray,
    temperature: float = TEMPERATURE,
) -> np.ndarray:
    """Vectorized port of StochParticles.critical_supersaturation.

    Golden-section maximization of the exact Koehler curve over all particles
    at once; same bracket and iteration budget as the Julia implementation.
    """

    dry_diameters = np.asarray(dry_diameters, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    v_dry = np.pi * dry_diameters**3 / 6.0
    r_dry = np.cbrt(3.0 * v_dry / (4.0 * np.pi))
    a_kelvin = (
        2.0 * KOEHLER_PARAMS["sigma"]
        / (KOEHLER_PARAMS["R_v"] * temperature * KOEHLER_PARAMS["rho_w"])
    )
    r_c_approx = np.sqrt(3.0 * kappas * v_dry / a_kelvin)

    a = r_dry * 1.001
    b = np.maximum(r_c_approx * 10.0, r_dry * 100.0)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        s_c = _kohler_supersaturation(c, v_dry, kappas, temperature)
        s_d = _kohler_supersaturation(d, v_dry, kappas, temperature)
        take_right = s_c < s_d
        a = np.where(take_right, c, a)
        b = np.where(take_right, b, d)
    r_opt = 0.5 * (a + b)
    return np.maximum(_kohler_supersaturation(r_opt, v_dry, kappas, temperature), 0.0)


CCN_SUPERSSATURATIONS = (0.001, 0.003, 0.01)


def _active_samples(rep: ReplicateData, row_index: int) -> tuple[np.ndarray, np.ndarray]:
    """(diameters, BC mass fractions) of active particles at one time row."""

    diameters = np.asarray(rep.arrays["diameter_samples"], dtype=float)
    fractions = np.asarray(rep.arrays["bc_mass_fraction_samples"], dtype=float)
    d_row = diameters[row_index, :] if diameters.ndim == 2 else diameters
    f_row = fractions[row_index, :] if fractions.ndim == 2 else fractions
    n = min(d_row.size, f_row.size)
    d_row = np.ravel(d_row)[:n]
    f_row = np.ravel(f_row)[:n]
    mask = (
        np.isfinite(d_row)
        & (d_row > 0.0)
        & np.isfinite(f_row)
        & (f_row >= 0.0)
        & (f_row <= 1.0)
    )
    return d_row[mask], f_row[mask]


def _n_times(reps: list[ReplicateData]) -> int:
    return np.asarray(reps[0].arrays["time"], dtype=float).size


def _series_on_common_time(
    reps: list[ReplicateData], errors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    common_time = np.asarray(reps[0].arrays["time"], dtype=float)
    for rep_idx, rep in enumerate(reps):
        time = np.asarray(rep.arrays["time"], dtype=float)
        if not np.array_equal(time, common_time):
            aligned = np.empty((errors.shape[1], common_time.size), dtype=float)
            for row in range(errors.shape[1]):
                aligned[row] = np.interp(common_time, time, errors[rep_idx, row])
            errors[rep_idx] = aligned
    return common_time, errors


def ccn_error_series(
    reps: list[ReplicateData],
    supersaturations: tuple[float, ...] = CCN_SUPERSSATURATIONS,
    temperature: float = TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray]:
    """CCN mixing-state error (Riemer et al. 2019): relative error in CCN number
    when every particle is assumed to carry the bulk volume-weighted kappa.

    Returns (time, errors) with errors shaped (n_reps, n_supersaturations, n_time);
    NaN where the actual CCN number is zero.
    """

    supersaturations = np.asarray(supersaturations, dtype=float)
    errors = np.full((len(reps), supersaturations.size, _n_times(reps)), np.nan)
    for rep_idx, rep in enumerate(reps):
        n_time = np.asarray(rep.arrays["time"], dtype=float).size
        for t_idx in range(n_time):
            diameters, fractions = _active_samples(rep, t_idx)
            if diameters.size == 0:
                continue
            kappas = particle_kappas(diameters, fractions)
            v_so4, v_bc = particle_species_volumes(diameters, fractions)
            kappa_bulk = (
                SPECIES_KAPPA[0] * v_so4.sum() + SPECIES_KAPPA[1] * v_bc.sum()
            ) / (v_so4.sum() + v_bc.sum())
            sc_actual = critical_supersaturations(diameters, kappas, temperature)
            sc_internal = critical_supersaturations(
                diameters, np.full_like(kappas, kappa_bulk), temperature
            )
            n_actual = (sc_actual[None, :] <= supersaturations[:, None]).sum(axis=1)
            n_internal = (sc_internal[None, :] <= supersaturations[:, None]).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                errors[rep_idx, :, t_idx] = np.where(
                    n_actual > 0,
                    (n_internal - n_actual) / n_actual,
                    np.nan,
                )
    return _series_on_common_time(reps, errors)


OPTICAL_WAVELENGTH = 550e-9  # m
REFRACTIVE_INDICES = {
    "so4": 1.53 + 0j,           # ammonium sulfate
    "bc": 1.95 + 0.79j,         # Bond & Bergstrom (2006)
}


def _particle_cross_sections(
    diameters: np.ndarray, core_volumes: np.ndarray, shell_volumes: np.ndarray,
    wavelength: float = OPTICAL_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-particle (C_abs, C_sca) [m²] for a BC core / SO4 shell sphere.

    Pure particles (zero core or zero shell volume) use the homogeneous-sphere
    MieQ; mixed particles use MieQCoreShell. All calls are per-particle scalars.
    """

    diameters = np.asarray(diameters, dtype=float)
    total = np.asarray(core_volumes, dtype=float) + np.asarray(shell_volumes, dtype=float)
    c_abs = np.zeros(diameters.size)
    c_sca = np.zeros(diameters.size)
    for i in range(diameters.size):
        d_total = diameters[i]
        area = np.pi * (0.5 * d_total) ** 2
        if core_volumes[i] <= 0.0:
            q = PyMieScatt.MieQ(REFRACTIVE_INDICES["so4"], wavelength, d_total, asDict=True)
        elif shell_volumes[i] <= 0.0:
            q = PyMieScatt.MieQ(REFRACTIVE_INDICES["bc"], wavelength, d_total, asDict=True)
        else:
            d_core = d_total * (core_volumes[i] / total[i]) ** (1.0 / 3.0)
            q = PyMieScatt.MieQCoreShell(
                REFRACTIVE_INDICES["bc"], REFRACTIVE_INDICES["so4"],
                wavelength, d_core, d_total, asDict=True,
            )
        c_abs[i] = q["Qabs"] * area
        c_sca[i] = q["Qsca"] * area
    return c_abs, c_sca


def optical_error_series(
    reps: list[ReplicateData],
    wavelength: float = OPTICAL_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Optical mixing-state error: relative error in bulk absorption/scattering
    when every particle is assumed to carry the bulk BC volume fraction as a
    BC core / SO4 shell (actual sizes kept).

    Returns (time, errors) with errors shaped (n_reps, 2, n_time):
    row 0 = ε_abs, row 1 = ε_sca; NaN where the actual coefficient is zero.
    """

    errors = np.full((len(reps), 2, _n_times(reps)), np.nan)
    for rep_idx, rep in enumerate(reps):
        n_time = np.asarray(rep.arrays["time"], dtype=float).size
        for t_idx in range(n_time):
            diameters, fractions = _active_samples(rep, t_idx)
            if diameters.size == 0:
                continue
            v_so4, v_bc = particle_species_volumes(diameters, fractions)
            v_total = v_so4 + v_bc
            c_abs_a, c_sca_a = _particle_cross_sections(diameters, v_bc, v_so4, wavelength)
            bc_bulk = v_bc.sum() / v_total.sum()
            c_abs_i, c_sca_i = _particle_cross_sections(
                diameters, bc_bulk * v_total, (1.0 - bc_bulk) * v_total, wavelength
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                errors[rep_idx, 0, t_idx] = (
                    (c_abs_i.sum() - c_abs_a.sum()) / c_abs_a.sum()
                    if c_abs_a.sum() > 0 else np.nan
                )
                errors[rep_idx, 1, t_idx] = (
                    (c_sca_i.sum() - c_sca_a.sum()) / c_sca_a.sum()
                    if c_sca_a.sum() > 0 else np.nan
                )
    return _series_on_common_time(reps, errors)
