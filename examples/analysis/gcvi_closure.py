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


def chi_target_of(scene: dict, case_name: str) -> float:
    """Target chi of a case (mean over replicate attrs; identical by construction)."""
    return float(np.mean([r.attrs["chi_target"] for r in scene[case_name]]))


def kld(p, q, floor: float = 1e-12) -> float:
    """KLD(p || q) over number-normalized spectra; q is floored.

    Empty-spectrum guard (calibration sweeps can produce empty CR output):
    both empty counts as a match (0.0); an empty simulation against a
    non-empty observation is the maximal mismatch (inf) — the formal value
    of 0 would wrongly declare a perfect match.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0.0 and q.sum() == 0.0:
        return 0.0
    if p.sum() == 0.0:
        return float("inf")
    q = np.maximum(q, floor)
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def chem_mse(sim, obs) -> float:
    return float(np.mean((np.asarray(sim, dtype=float) - np.asarray(obs, dtype=float)) ** 2))


def _case_J_values(scene: dict, case_name: str, obs: dict,
                   w_size: float = 1.0, w_chem: float = 1.0) -> list:
    """Per-replicate total J of one case against the observations."""
    values = []
    for rep in scene[case_name]:
        vi = final_virtual_instrument(rep)
        j_size = 0.5 * (kld(vi["cr"], obs["cr_dNdlogD"]) +
                        kld(vi["ci"], obs["ci_dNdlogD"]))
        j_chem = 0.5 * (chem_mse(vi["cr_chemistry"], obs["cr_chemistry"]) +
                        chem_mse(vi["ci_chemistry"], obs["ci_chemistry"]))
        values.append(w_size * j_size + w_chem * j_chem)
    return values


def cost_curve(scene: dict, obs: dict, w_size: float = 1.0,
               w_chem: float = 1.0) -> list:
    """Pooled J per case: (chi_target, J_mean, J_std), sorted by chi_target.

    J is nonlinear, so replicates are averaged *after* evaluating J (the
    M0 mean-of-arrays pooling biased the cost); J_std is the replicate
    spread at that chi.
    """
    rows = []
    for name in scene:
        js = _case_J_values(scene, name, obs, w_size, w_chem)
        rows.append((chi_target_of(scene, name),
                     float(np.mean(js)), float(np.std(js))))
    return sorted(rows, key=lambda r: r[0])


def truth_case_name(scene: dict) -> str:
    """The single case flagged truth=true in its replicate attrs."""
    truths = [n for n in scene if bool(scene[n][0].attrs.get("truth", False))]
    if len(truths) != 1:
        raise ValueError(f"expected exactly one truth case, found {truths}")
    return truths[0]


def twin_gate(scene: dict, obs: dict, radius: float = 0.10) -> dict:
    """Twin-v1 gate: argmin J over the chi grid must bracket chi_true.

    Noise floor sigma_J: std of per-replicate J in the truth case against
    the fixed observation; the obs-source replicate is excluded because its
    J is identically 0. `separation > 2*sigma_J` is a soft check (printed,
    not gating).
    """
    truth = truth_case_name(scene)
    grid_names = [n for n in scene if n != truth]
    rows = sorted((chi_target_of(scene, n),
                   float(np.mean(_case_J_values(scene, n, obs, 1.0, 1.0))))
                  for n in grid_names)
    chis = [r[0] for r in rows]
    js = [r[1] for r in rows]
    chi_hat = chis[int(np.argmin(js))]
    chi_true = obs["chi_true"]
    ok = abs(chi_hat - chi_true) <= radius
    truth_J = _case_J_values(scene, truth, obs, 1.0, 1.0)
    sigma_j = float(np.std(truth_J[1:])) if len(truth_J) > 1 else 0.0
    separation = float(max(js) - min(js))
    verdict = {
        "pass": bool(ok), "chi_hat": chi_hat, "chi_true": chi_true,
        "sigma_J": sigma_j, "separation": separation,
        "separation_gt_2sigma": bool(separation > 2.0 * sigma_j),
        "grid": chis, "J": js,
    }
    print("Twin v1 gate:", "PASS" if ok else "FAIL")
    print(f"  chi_hat={chi_hat:.3f}  chi_true={chi_true:.3f}  "
          f"|chi_hat-chi_true|={abs(chi_hat - chi_true):.3f} <= {radius}")
    print(f"  noise floor sigma_J={sigma_j:.5f}  separation={separation:.4f}  "
          f"(> 2*sigma_J: {verdict['separation_gt_2sigma']})")
    return verdict
