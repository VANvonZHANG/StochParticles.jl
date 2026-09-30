#!/usr/bin/env python3
"""GCVI closure M0: CR spectra across nu and the J(chi) cost curve."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.figure_style import PALETTE, apply_publication_style
from analysis.gcvi_closure import (case_statistics, chi_realized, chi_target_of,
                                   cost_curve, load_observations, truth_case_name,
                                   twin_gate)
from analysis.stochparticles_io import DATA_DIR, FIG_DIR, read_scene

# PALETTE is a name->hex dict in figure_style; scan lines need a color sequence.
SCAN_COLORS = list(PALETTE.values())


def main() -> None:
    apply_publication_style()
    scene = read_scene(DATA_DIR / "gcvi_closure.h5")
    obs = load_observations(DATA_DIR / "synthetic" / "twin_obs_v0.h5")
    rows = cost_curve(scene, obs)
    truth_target = chi_target_of(scene, truth_case_name(scene))
    rows = [r for r in rows if abs(r[0] - truth_target) > 1e-9]

    case_names = sorted(scene)
    edges = obs["bin_edges"]
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 75 / 25.4))

    ax = axes[0]
    for color, name in zip(SCAN_COLORS, case_names):
        stat = case_statistics(scene, name)
        chi = chi_realized(scene, name)
        ax.plot(centers, stat["cr"], color=color, label=f"scan chi={chi:.2f}")
    ax.plot(centers, obs["cr_dNdlogD"], color="black", linestyle="--",
            linewidth=1.2, label="truth (obs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dry diameter D [m]")
    ax.set_ylabel(r"CR dN/dlogD [m$^{-3}$]")
    ax.set_title("Virtual-SMPS CR spectra", fontsize=8)
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    chi_t = [r[0] for r in rows]
    j_mean = [r[1] for r in rows]
    j_std = [r[2] for r in rows]
    ax.errorbar(chi_t, j_mean, yerr=j_std, marker="o", capsize=2,
                color=SCAN_COLORS[0], label="J(chi) mean±std")
    best_idx = int(np.argmin(j_mean))
    ax.plot(chi_t[best_idx], j_mean[best_idx], marker="*", markersize=12,
            color=SCAN_COLORS[1], label=f"argmin J: chi={chi_t[best_idx]:.2f}")
    ax.axvline(obs["chi_true"], color="black", linestyle="--", linewidth=1.0,
               label=f"chi_true={obs['chi_true']:.2f}")
    # chi_realized overlay at the bottom: generation noise per case
    names = sorted(scene, key=lambda n: chi_target_of(scene, n))
    truth = [n for n in names if bool(scene[n][0].attrs.get("truth", False))]
    scan_names = [n for n in names if n not in truth]
    chi_r = [chi_realized(scene, n) for n in scan_names]
    chi_r_std = [float(np.std([r.attrs["chi_realized"] for r in scene[n]]))
                 for n in scan_names]
    yline = min(j_mean) - max(j_std) - 0.05 * (max(j_mean) - min(j_mean))
    ax.errorbar(chi_r, [yline] * len(chi_r), xerr=chi_r_std, fmt="|",
                color="gray", capsize=2, label="chi_realized (mean±std)")
    ax.set_xlabel("Target chi")
    ax.set_ylabel("J(chi)")
    ax.set_title("Closure cost curve (twin v1)", fontsize=8)
    ax.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    out = FIG_DIR / "gcvi_closure.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")

    print("J(chi) curve:")
    for c, j_mean_, j_std_ in rows:
        print(f"  chi_target={c:.3f}  J={j_mean_:.4f} ± {j_std_:.4f}")
    print(f"chi_true = {obs['chi_true']:.3f}")
    verdict = twin_gate(scene, obs)


if __name__ == "__main__":
    main()
