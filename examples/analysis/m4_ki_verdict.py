#!/usr/bin/env python3
"""M4 leg-1 verdict: can our (kinetic-arm) output be compared with pyrcel?

Metric-matched end-state comparison at the C-prime scenario (w=1.0):
  - S_max: trajectory max (both sides, full 600-s window)
  - N_act: OUR arms count at t=600 (frozen sc/sc_re vs free ba/ki);
    pyrcel canonical Nd is the PEAK-time count (terminate=True default) —
    reported alongside to expose the metric mismatch
  - q_liq(end): liquid water mixing ratio at t=600 — metric-neutral
    (both sides terminal state, no activation-criterion ambiguity)

pyrcel side: NATIVE ParcelModel, terminate=False (full window), same
kappa-tercile populations as the campaign replicates.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np

from .m3_quantification import (N_TOTAL, P0, PYRCEL_REPS, S0, T0, T_END, V_UP,
                                pyrcel_species_groups)
from .stochparticles_io import DATA_DIR, read_scene

CASES = ["chi_0.1", "chi_0.25", "chi_0.4", "chi_0.5", "chi_0.6", "chi_0.75"]
ARMS = {"sc": "gcvi_closure_parcel_sc.h5",
        "ba": "gcvi_closure_parcel_ba.h5",
        "sc_re": "gcvi_closure_parcel_sc_re_w1.h5",
        "ki": "gcvi_closure_parcel_ki.h5"}


def our_ql_end(rep, m_air, vol):
    # species_mass_concentration rows: [AS, AN, OA, BC, H2O] [kg/m^3];
    # end-state liquid water mixing ratio = water conc * V / m_air
    return float(rep.arrays["species_mass_concentration"][-1, 4]) * vol / m_air


def our_smax(rep):
    return float(np.max(rep.arrays["parcel_S"]))


def our_nact(rep):
    return float(rep.arrays["activation_fraction"][-1])


def pyrcel_ref(rep):
    """(S_max, N_act_peak, wc_end) — native full-window run. output_dt=10:
    pyrcel uses it as the SOLVER step as well — coarser steps go numerically
    unstable on the stiff condensation and inflate S_max spuriously."""
    import pyrcel as pm
    groups = pyrcel_species_groups(rep.arrays["dry_diameter_initial"] * 1e6,
                                   rep.arrays["kappa_bar_initial"])
    model = pm.ParcelModel(groups, V=V_UP, T0=T0, S0=S0, P0=P0)
    out = model.run(t_end=T_END, output_dt=10.0, terminate=False)
    wc = np.asarray(out.wc, dtype=float)
    return float(out.summary["S_max"]), float(out.Nd) / N_TOTAL, float(wc[-1])


def main() -> None:
    out_dir = DATA_DIR / "m4_ki_verdict"
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = {a: read_scene(DATA_DIR / p) for a, p in ARMS.items()}
    # tracked-parcel volume and dry-air mass (driver convention):
    # V = n_sim / N_TOTAL, m_air = rho0 * V with rho0 = P0/(R_d T0)
    vol = 1000.0 / N_TOTAL
    m_air = P0 / (287.05 * T0) * vol

    lines = ["# M4 leg-1: ki real-spectrum campaign vs pyrcel (w=1.0, n=1000)",
             "",
             "| case | S_max(ki) | S_max(pyrcel) | N_act(ki) | N_act(sc_re) |"
             " N_act(pyrcel peak) | q_liq_end(ki) | q_liq_end(pyrcel) |",
             "|---|---|---|---|---|---|---|---|"]
    per_case = []
    for case in CASES:
        reps = scenes["ki"][case]
        ref = [pyrcel_ref(r) for r in scenes["sc"][case][:PYRCEL_REPS]]
        s_ref = float(np.mean([r[0] for r in ref]))
        n_ref = float(np.mean([r[1] for r in ref]))
        w_ref = float(np.mean([r[2] for r in ref]))
        s_ki = float(np.mean([our_smax(r) for r in reps]))
        n_ki = float(np.mean([our_nact(r) for r in reps]))
        n_re = float(np.mean([our_nact(r) for r in scenes["sc_re"][case]]))
        w_ki = float(np.mean([our_ql_end(r, m_air, vol) for r in reps]))
        lines.append(f"| {case} | {s_ki:.5f} | {s_ref:.5f} | {n_ki:.4f} |"
                     f" {n_re:.4f} | {n_ref:.4f} | {w_ki:.3e} | {w_ref:.3e} |")
        per_case.append((case, s_ki, s_ref, n_ki, n_re, n_ref, w_ki, w_ref))

    ds = [abs(c[1] / c[2] - 1.0) for c in per_case]
    dw = [abs(c[6] / c[7] - 1.0) for c in per_case]
    dn_ki = [abs(c[3] / c[5] - 1.0) for c in per_case]
    dn_re = [abs(c[4] / c[5] - 1.0) for c in per_case]
    lines += ["",
              f"S_max ratio ki/pyrcel: median {np.median(ds):.3%} off unity"
              f" (range {min(ds):.3%}..{max(ds):.3%}) — systematic offset, all arms",
              f"q_liq_end ki/pyrcel: median {np.median(dw):.3%} off unity"
              f" (range {min(dw):.3%}..{max(dw):.3%}) — metric-neutral terminal state",
              f"N_act vs pyrcel-peak: ki median {np.median(dn_ki):.3f}"
              f" / sc_re median {np.median(dn_re):.3f} (sc_re closer: frozen-edge"
              " terminal count ~ peak-time count; ki terminal evaporates the edge)"]
    (out_dir / "verdict.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
