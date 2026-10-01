"""Closure diagnostics for the virtual-GCVI twin experiment (blueprint contract 4)."""

from __future__ import annotations

import h5py
import numpy as np

from .stochparticles_io import read_scene


def load_observations(path) -> dict:
    """Load twin/real observations: CR/CI spectra + chemistry + chi_true.

    Current schema `synthetic-obs-v2`: truth-case replicate mean, with the
    replicate count in the `n_replicates_pooled` attr; v1 files (single
    replicate) still load but are superseded.
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


def kld(p, q, floor=None) -> float:
    """KLD(p || q) over number-normalized spectra; q is floored.

    Empty-spectrum guard (calibration sweeps can produce empty CR output):
    both empty counts as a match (0.0); an empty simulation against a
    non-empty observation is the maximal mismatch (inf) — the formal value
    of 0 would wrongly declare a perfect match.

    `floor=None` (default) uses a half-count-equivalent floor, estimated
    from q's smallest nonzero bin: for a particle-sampled spectrum that bin
    IS one particle, so an occupied-in-p / empty-in-q bin costs
    ~log(2·k) nats instead of the ~27 nats an arbitrary 1e-12 floor charges.
    A numeric floor restores the old fixed-floor behavior.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0.0 and q.sum() == 0.0:
        return 0.0
    if p.sum() == 0.0:
        return float("inf")
    if floor is None:
        positive = q[q > 0.0]
        floor = float(positive.min()) / 2.0 if positive.size else 1e-12
    q = np.maximum(q, floor)
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def rebin_spectrum(bin_edges, dNdlogD, stride: int = 3):
    """Merge `stride` adjacent bins (SMPS-style coarsening).

    With uniform log-spaced edges the merged dN/dlogD is the count-weighted
    mean of the merged bins; trailing remainder bins are dropped. Returns
    (new_edges, new_dNdlogD). Merged values preserve the input's
    normalization convention (the Julia side is dN/dlnD; only ratios are
    used downstream).
    """
    edges = np.asarray(bin_edges, dtype=float)
    vals = np.asarray(dNdlogD, dtype=float)
    log_edges = np.log10(edges)
    new_edges = edges[::stride]
    n_new = len(new_edges) - 1
    out = np.empty(n_new)
    for b in range(n_new):
        lo, hi = b * stride, min(b * stride + stride, len(vals))
        counts = vals[lo:hi] * np.diff(log_edges)[lo:hi]
        out[b] = counts.sum() / (log_edges[min(hi, len(log_edges) - 1)] - log_edges[lo])
    return new_edges, out


def chem_mse(sim, obs) -> float:
    return float(np.mean((np.asarray(sim, dtype=float) - np.asarray(obs, dtype=float)) ** 2))


# closure-bin coarsening stride shared by _case_J_values and pool_observation
CLOSURE_STRIDE = 3


def _case_J_values(scene: dict, case_name: str, obs: dict,
                   w_size: float = 1.0, w_chem: float = 1.0) -> list:
    """Per-replicate total J of one case against the observations.

    Spectra are coarsened to the closure binning (stride 3, 95 -> 31 bins)
    before the KLD: at n_sim ~ 1000 the fine grid leaves ~3 CR particles
    per bin and the KLD's empty-bin penalties measure replicate identity,
    not chi (measured 2026-09-30: replicate-vs-replicate KLD floor 1.34 ±
    0.30 exceeded every case-vs-obs J).
    """
    values = []
    for rep in scene[case_name]:
        vi = final_virtual_instrument(rep)
        cr = rebin_spectrum(obs["bin_edges"], vi["cr"], CLOSURE_STRIDE)[1]
        ci = rebin_spectrum(obs["bin_edges"], vi["ci"], CLOSURE_STRIDE)[1]
        j_size = 0.5 * (kld(cr, obs["cr_dNdlogD_rebinned"]) +
                        kld(ci, obs["ci_dNdlogD_rebinned"]))
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


def pool_observation(scene: dict, obs: dict, stride: int = CLOSURE_STRIDE) -> dict:
    """Pooled observation: truth-case replicate-mean instrument output.

    Real SMPS/ACSM observations average over a sampling window, so the twin
    observation is the replicate mean, not a single realization (a
    single-replicate obs makes the closure cost measure replicate identity:
    sigma_J ~ 0.30 at the 2026-09-30 run). Spectra are pre-rebinned to the
    closure binning so `kld` compares like with like. `chi_true` becomes
    the truth case's mean realized chi.
    """
    truth = truth_case_name(scene)
    reps = [final_virtual_instrument(r) for r in scene[truth]]
    pooled = {
        "bin_edges": obs["bin_edges"],
        "cr_dNdlogD": np.mean([r["cr"] for r in reps], axis=0),
        "ci_dNdlogD": np.mean([r["ci"] for r in reps], axis=0),
        "cr_chemistry": np.mean([r["cr_chemistry"] for r in reps], axis=0),
        "ci_chemistry": np.mean([r["ci_chemistry"] for r in reps], axis=0),
        "chi_true": float(np.mean([r.attrs["chi_realized"] for r in scene[truth]])),
        "n_replicates_pooled": len(reps),
    }
    pooled["cr_dNdlogD_rebinned"] = rebin_spectrum(obs["bin_edges"],
                                                   pooled["cr_dNdlogD"], stride)[1]
    pooled["ci_dNdlogD_rebinned"] = rebin_spectrum(obs["bin_edges"],
                                                   pooled["ci_dNdlogD"], stride)[1]
    # cross-check against a synthetic-obs-v2 file (n_replicates_pooled attr):
    # the file and the recomputed pool are two implementations of the same
    # statistic (Julia driver / Python analysis); drift must fail loudly
    if obs.get("n_replicates_pooled") not in (None, len(reps)):
        raise ValueError(f"obs file pooled over {obs['n_replicates_pooled']} "
                         f"replicates but scene truth case has {len(reps)}")
    if obs.get("n_replicates_pooled") is not None and not np.isclose(
            obs["chi_true"], pooled["chi_true"], atol=1e-9):
        raise ValueError(f"obs file chi_true {obs['chi_true']:.6f} disagrees "
                         f"with pooled scene mean {pooled['chi_true']:.6f}")
    return pooled


def twin_gate(scene: dict, obs: dict, radius: float = 0.10) -> dict:
    """Twin-v1 gate: argmin J over the chi grid must bracket chi_true.

    `obs` should be the pooled observation (`pool_observation`). Noise
    floor sigma_J: std of per-replicate J in the truth case against the
    pooled observation — with a pooled obs no replicate is the obs source,
    so all replicates contribute. `separation > 2*sigma_J` is a soft check
    (printed, not gating).
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
    sigma_j = float(np.std(truth_J)) if len(truth_J) > 1 else 0.0
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
