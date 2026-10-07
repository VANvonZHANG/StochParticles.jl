#!/usr/bin/env python3
"""Twin v2: the full 2x2 mode matrix (obs x forward) — spec §6.

Diagonal cells are the two twins (M2 open baseline + M3 closed flagship);
off-diagonal cells quantify the cost of inferring with the wrong-physics
forward model. Cross cells cost analysis only — the campaign data already
exists. Note: populations are seed-identical across modes, so
`pool_observation`'s cross-check (obs-file attrs vs scene truth) passes on
cross cells too — chi_true is a property of the shared population.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from .gcvi_closure import (cost_curve, load_observations, pool_observation,
                           twin_gate)
from .stochparticles_io import DATA_DIR, read_scene

# Flagship closed arm = the re-equilibrated (haze-buffer) physics arm.
# If Task-12 adjudication picks a different default, switch the suffix
# ("ba" | "ki" | "sc").
TAG = "sc_re"
CELLS = {
    ("open", "open"): (DATA_DIR / "gcvi_closure.h5",
                       DATA_DIR / "synthetic/twin_obs_v0.h5"),
    ("closed", "closed"): (DATA_DIR / f"gcvi_closure_parcel_{TAG}.h5",
                           DATA_DIR / f"synthetic/twin_obs_parcel_{TAG}.h5"),
    ("closed", "open"): (DATA_DIR / f"gcvi_closure_parcel_{TAG}.h5",
                         DATA_DIR / "synthetic/twin_obs_v0.h5"),
    ("open", "closed"): (DATA_DIR / "gcvi_closure.h5",
                         DATA_DIR / f"synthetic/twin_obs_parcel_{TAG}.h5"),
}


def main() -> None:
    print(f"{'obs':>8} {'fwd':>8} {'chi_hat':>8} {'chi_true':>9} "
          f"{'|err|':>7} {'sigma_J':>8} {'sep':>7} {'gate':>6}")
    results = {}
    for (obs_mode, fwd_mode), (scene_path, obs_path) in CELLS.items():
        scene = read_scene(scene_path)
        obs = pool_observation(scene, load_observations(obs_path))
        v = twin_gate(scene, obs)          # prints its own PASS/FAIL line
        results[(obs_mode, fwd_mode)] = v
        print(f"{obs_mode:>8} {fwd_mode:>8} {v['chi_hat']:>8.3f} "
              f"{v['chi_true']:>9.3f} {abs(v['chi_hat'] - v['chi_true']):>7.3f} "
              f"{v['sigma_J']:>8.5f} {v['separation']:>7.4f} "
              f"{'PASS' if v['pass'] else 'FAIL':>6}")
        rows = cost_curve(scene, obs)
        print("   J curve:", "  ".join(f"{c:.2f}:{j:.4f}" for c, j, _ in rows))
    cross_err = abs(results[("closed", "open")]["chi_hat"]
                    - results[("closed", "open")]["chi_true"])
    diag_err = abs(results[("closed", "closed")]["chi_hat"]
                   - results[("closed", "closed")]["chi_true"])
    print(f"\ncompetition cost (wrong-physics forward): cross |err|="
          f"{cross_err:.3f} vs closed-diagonal |err|={diag_err:.3f}")


if __name__ == "__main__":
    main()
