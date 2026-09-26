"""Closure diagnostics for the virtual-GCVI twin experiment (blueprint contract 4)."""

from __future__ import annotations

import h5py
import numpy as np

from .stochparticles_io import read_scene


def load_observations(path) -> dict:
    """Load twin/real observations: CR/CI spectra + chemistry + chi_true.

    Schema `synthetic-obs-v1`; the real-data ingestion project will emit the
    same fields, making this reader the single swap point.
    """
    with h5py.File(path, "r") as f:
        g = f["obs"]
        return {
            "bin_edges": g["bin_edges"][()],
            "cr_dNdlogD": g["cr_dNdlogD"][()],
            "ci_dNdlogD": g["ci_dNdlogD"][()],
            "cr_chemistry": g["cr_chemistry"][()],
            "ci_chemistry": g["ci_chemistry"][()],
            "chi_true": float(g.attrs["chi_true"]),
        }


def final_virtual_instrument(rep) -> dict:
    """Last-row virtual-instrument output of one replicate record."""
    return {
        "cr": rep.arrays["cr_spectrum"][-1, :],
        "ci": rep.arrays["ci_spectrum"][-1, :],
        "cr_chemistry": rep.arrays["cr_chemistry"][-1, :],
        "ci_chemistry": rep.arrays["ci_chemistry"][-1, :],
    }


def case_statistics(scene: dict, case_name: str) -> dict:
    """Mean virtual-instrument output across replicates of one case."""
    per_rep = [final_virtual_instrument(r) for r in scene[case_name]]
    return {k: np.mean([d[k] for d in per_rep], axis=0) for k in per_rep[0]}


def chi_realized(scene: dict, case_name: str) -> float:
    return float(np.mean([r.attrs["chi_realized"] for r in scene[case_name]]))


def kld(p, q, floor: float = 1e-12) -> float:
    """KLD(p || q) over number-normalized spectra; q is floored."""
    p = np.asarray(p, dtype=float)
    q = np.maximum(np.asarray(q, dtype=float), floor)
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def chem_mse(sim, obs) -> float:
    return float(np.mean((np.asarray(sim, dtype=float) - np.asarray(obs, dtype=float)) ** 2))


def cost_curve(scene: dict, obs: dict, w_size: float = 1.0, w_chem: float = 1.0) -> list:
    """J(chi) = w_size * J_size + w_chem * J_chem per case, sorted by chi.

    J_size: mean KLD of CR and CI spectra against observations.
    J_chem: mean MSE of CR and CI dry chemistry against observations.
    """
    rows = []
    for name in scene:
        stat = case_statistics(scene, name)
        j_size = 0.5 * (kld(stat["cr"], obs["cr_dNdlogD"]) + kld(stat["ci"], obs["ci_dNdlogD"]))
        j_chem = 0.5 * (chem_mse(stat["cr_chemistry"], obs["cr_chemistry"]) +
                        chem_mse(stat["ci_chemistry"], obs["ci_chemistry"]))
        rows.append((chi_realized(scene, name), j_size, j_chem, w_size * j_size + w_chem * j_chem))
    return sorted(rows, key=lambda r: r[0])
