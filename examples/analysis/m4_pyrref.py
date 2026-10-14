#!/usr/bin/env python3
"""M4 model-vs-model: native pyrcel reference runner (variant-aware).

Runs the native pyrcel ParcelModel on the SAME initial populations as the
campaign replicates (read from the ki arm HDF5; populations are
seed-identical across arms and w settings — seeds key on case/replicate
only). Per case x replicate it performs BOTH runs:

  - full window (terminate=False): S(t), wc(t) trajectories for overlays
  - canonical   (terminate=True) : pyrcel's own S_max + peak-time Nd

Variants ('stock' vs 'harmonized') differ ONLY in the pyrcel source state
on disk when the process starts; this script never patches anything, so
launch each variant against the intended site-packages state (fresh
process each — JAX traces capture constants at first compile):

  python -m analysis.m4_pyrref stock      data/m4_harmonize/pyrref_stock.npz
  python -m analysis.m4_pyrref harmonized data/m4_harmonize/pyrref_harmonized.npz
"""
from __future__ import annotations

import json
import sys
import time as _time

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np

from .m3_quantification import N_TOTAL, N_TOTAL_CM3, P0, S0, T0, T_END, V_UP
from .stochparticles_io import DATA_DIR, read_scene

CASES = ["chi_0.1", "chi_0.25", "chi_0.4", "chi_0.5", "chi_0.6", "chi_0.75"]
REPS = 5
SC_ARM = DATA_DIR / "gcvi_closure_parcel_ki_postfix.h5"  # seed-identical populations (docstring: seeds key on case/replicate only); was the sc arm before four-arm retirement
COLS = ["z", "P", "T", "wv", "wc", "wi", "S"]


def species_and_weights(dry_um, kappas):
    """Kappa-tercile split -> (AerosolSpecies list, [(r_dry_um, N_cm3)]).

    Mirrors m3_quantification.pyrcel_species_groups but also returns the
    per-bin (dry radius, number) weights it fed in, for the end-spectrum
    panel."""
    import pyrcel as pm

    scale = N_TOTAL_CM3 / len(dry_um)
    edges = np.quantile(kappas, [1 / 3, 2 / 3])
    groups = [kappas <= edges[0],
              (kappas > edges[0]) & (kappas <= edges[1]),
              kappas > edges[1]]
    species, weights = [], []
    for g in groups:
        if not g.any():
            continue
        d = dry_um[g]
        be = np.quantile(d, np.linspace(0, 1, 13))
        be[0], be[-1] = be[0] * 0.5, be[-1] * 1.5
        idx = np.clip(np.digitize(d, be) - 1, 0, 11)
        cnt = np.bincount(idx, minlength=12) * scale
        radii = (np.sqrt(be[:-1] * be[1:]) / 2.0)[cnt > 0]
        cnt = cnt[cnt > 0]
        species.append(pm.AerosolSpecies(
            "m3", {"r_drys": radii, "Nis": cnt}, kappa=float(np.mean(kappas[g]))))
        weights.append((radii, cnt))
    return species, weights


def run_one(rep, terminate: bool):
    species, weights = species_and_weights(
        rep.arrays["dry_diameter_initial"] * 1e6, rep.arrays["kappa_bar_initial"])
    import pyrcel as pm
    model = pm.ParcelModel(species, V=V_UP, T0=T0, S0=S0, P0=P0)
    out = model.run(t_end=T_END, output_dt=10.0, terminate=terminate)
    return species, weights, out


def main() -> None:
    variant, out_path = sys.argv[1], Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = read_scene(SC_ARM)
    data = {}
    meta = {"variant": variant, "T0": T0, "P0": P0, "S0": S0, "V_UP": V_UP,
            "t_end": T_END, "output_dt": 10.0, "reps": REPS, "cases": CASES}
    for case in CASES:
        for rep_i, rep in enumerate(scene[case][:REPS]):
            key = f"{case}/rep{rep_i}"
            t_wall = _time.time()
            species, weights, full = run_one(rep, terminate=False)
            _, _, can = run_one(rep, terminate=True)
            state = np.asarray(full.state, dtype=float)  # (nt, 7 + nr)
            nr = state.shape[1] - 7
            data[f"{key}/time"] = np.asarray(full.time, dtype=float)
            for j, col in enumerate(COLS):
                data[f"{key}/{col}"] = state[:, j]
            data[f"{key}/r_end"] = state[-1, 7:7 + nr]
            data[f"{key}/r_dry_um"] = np.concatenate([r for r, _ in weights])
            data[f"{key}/N_cm3"] = np.concatenate([n for _, n in weights])
            data[f"{key}/S_max_full"] = float(full.summary["S_max"])
            data[f"{key}/t_smax"] = float(full.summary["t_smax"])
            data[f"{key}/S_max_canonical"] = float(can.summary["S_max"])
            data[f"{key}/Nd_peak_frac"] = float(can.Nd) / N_TOTAL
            print(f"{key}: S_max={full.summary['S_max']:.5f} "
                  f"wc_end={state[-1, 4]:.3e} "
                  f"wall={_time.time() - t_wall:.1f}s", flush=True)
    np.savez_compressed(out_path, meta_json=np.array(json.dumps(meta)), **data)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
