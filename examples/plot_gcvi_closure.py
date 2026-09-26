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
from analysis.gcvi_closure import case_statistics, chi_realized, cost_curve, load_observations
from analysis.stochparticles_io import DATA_DIR, FIG_DIR, read_scene

# PALETTE is a name->hex dict in figure_style; scan lines need a color sequence.
SCAN_COLORS = list(PALETTE.values())


def main() -> None:
    apply_publication_style()
    scene = read_scene(DATA_DIR / "gcvi_closure.h5")
    obs = load_observations(DATA_DIR / "synthetic" / "twin_obs_v0.h5")
    rows = cost_curve(scene, obs)

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
    chi = [r[0] for r in rows]
    j_total = [r[3] for r in rows]
    ax.plot(chi, j_total, marker="o", color=SCAN_COLORS[0], label="J(chi)")
    best_idx = int(np.argmin(j_total))
    ax.plot(chi[best_idx], j_total[best_idx], marker="*", markersize=12,
            color=SCAN_COLORS[1], label=f"argmin J: chi={chi[best_idx]:.2f}")
    ax.axvline(obs["chi_true"], color="black", linestyle="--", linewidth=1.0,
               label=f"chi_true={obs['chi_true']:.2f}")
    ax.set_xlabel("Realized chi")
    ax.set_ylabel("J(chi)")
    ax.set_title("Closure cost curve (twin v0)", fontsize=8)
    ax.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    out = FIG_DIR / "gcvi_closure.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")

    print("J(chi) curve:")
    for c, j_size, j_chem, j in rows:
        print(f"  chi={c:.3f}  J_size={j_size:.4f}  J_chem={j_chem:.5f}  J={j:.4f}")
    print(f"chi_true = {obs['chi_true']:.3f}; argmin J at chi = {chi[best_idx]:.3f}")


if __name__ == "__main__":
    main()
