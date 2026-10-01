#!/usr/bin/env python3
"""GCVI closure M2: four-panel twin-experiment figure.

(a) CR spectra across the chi grid (instrument view; the pooled truth obs is
    the black dashed line), (b) relative difference spectra vs the pooled
    truth observation — this is where the chi signal lives: the activation
    transition band separates by up to ~30% across chi while the spectrum
    plateau is chi-blind to first order, (c) the CR/CI chemistry channel
    (dry mass fractions vs chi), (d) the J(chi) closure cost curve.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.figure_style import PALETTE, apply_publication_style
from analysis.gcvi_closure import (case_statistics, chi_realized, chi_target_of,
                                   cost_curve, load_observations, pool_observation,
                                   truth_case_name, twin_gate)
from analysis.stochparticles_io import DATA_DIR, FIG_DIR, read_scene

# PALETTE is a name->hex dict in figure_style; scan lines need a color sequence.
SCAN_COLORS = list(PALETTE.values())
SPECIES_COLORS = {"AS": SCAN_COLORS[0], "AN": SCAN_COLORS[2], "OA": SCAN_COLORS[3]}
SPECIES = ("AS", "AN", "OA")
# activation transition band (mode valley / kappa-selection region), from the
# measured across-case separation of the CR spectra
TRANSITION_BAND = (7.5e-8, 1.4e-7)


def main() -> None:
    apply_publication_style()
    scene = read_scene(DATA_DIR / "gcvi_closure.h5")
    obs = load_observations(DATA_DIR / "synthetic" / "twin_obs_v0.h5")
    # pooled (time-averaged) observation: real instruments average over the
    # sampling window; a single-replicate obs measures replicate identity
    obs = pool_observation(scene, obs)
    rows = cost_curve(scene, obs)
    truth = truth_case_name(scene)
    truth_target = chi_target_of(scene, truth)
    rows = [r for r in rows if abs(r[0] - truth_target) > 1e-9]

    scan_names = sorted((n for n in scene if n != truth),
                        key=lambda n: chi_target_of(scene, n))
    edges = obs["bin_edges"]
    centers = np.sqrt(edges[:-1] * edges[1:])
    stats = {name: case_statistics(scene, name) for name in scene}

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 150 / 25.4))

    # (a) instrument view: CR spectra (log-log) + pooled truth obs
    ax = axes[0, 0]
    for color, name in zip(SCAN_COLORS, scan_names):
        ax.plot(centers, stats[name]["cr"], color=color,
                label=f"chi={chi_realized(scene, name):.2f}")
    ax.plot(centers, obs["cr_dNdlogD"], color="black", linestyle="--",
            linewidth=1.2, label="truth (obs, pooled)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dry diameter D [m]")
    ax.set_ylabel(r"CR dN/dlogD [m$^{-3}$]")
    ax.set_title("Virtual-SMPS CR spectra", fontsize=8)
    ax.legend(frameon=False, fontsize=6)

    # (b) where the chi signal lives: relative difference vs the pooled truth
    ax = axes[0, 1]
    obs_cr = obs["cr_dNdlogD"]
    populated = obs_cr > 0.02 * obs_cr.max()   # plot only where obs is measured
    for color, name in zip(SCAN_COLORS, scan_names):
        rel = np.where(populated,
                       (stats[name]["cr"] - obs_cr) / np.maximum(obs_cr, 1e-300),
                       np.nan)
        ax.plot(centers, rel, color=color, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvspan(*TRANSITION_BAND, color="gray", alpha=0.12)
    ax.annotate("activation edge", xy=(np.sqrt(TRANSITION_BAND[0] * TRANSITION_BAND[1]),
                ax.get_ylim()[1]), fontsize=6, ha="center", va="top", color="dimgray")
    ax.set_xscale("log")
    ax.set_xlabel("Dry diameter D [m]")
    ax.set_ylabel(r"$\Delta$CR / CR$_{\mathrm{truth}}$")
    ax.set_title("CR spectral separation (where chi lives)", fontsize=8)

    # (c) chemistry channel: CR/CI dry fractions vs chi + the pooled truth obs
    ax = axes[1, 0]
    chi_t_all = [chi_target_of(scene, n) for n in scan_names]
    for species, color in SPECIES_COLORS.items():
        k = SPECIES.index(species)
        cr = [stats[n]["cr_chemistry"][k] for n in scan_names]
        ci = [stats[n]["ci_chemistry"][k] for n in scan_names]
        ax.plot(chi_t_all, cr, marker="o", color=color, label=species)
        ax.plot(chi_t_all, ci, marker="s", linestyle="--", color=color,
                markerfacecolor="none")
    for channel, marker in (("cr", "*"), ("ci", "*")):
        for species, color in SPECIES_COLORS.items():
            k = SPECIES.index(species)
            ax.plot(obs["chi_true"], obs[f"{channel}_chemistry"][k], marker=marker,
                    markersize=10, color=color, markeredgecolor="black",
                    linewidth=0)
    ax.set_xlabel("Target chi")
    ax.set_ylabel("Dry mass fraction")
    ax.set_title("CR (filled) / CI (open) chemistry vs chi; stars = truth obs",
                 fontsize=8)
    ax.legend(frameon=False, fontsize=6)

    # (d) closure cost curve with replicate error bars and realized-chi overlay
    ax = axes[1, 1]
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
