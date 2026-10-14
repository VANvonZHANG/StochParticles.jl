#!/usr/bin/env python3
"""M4 model-vs-model verdict: our ki arm vs pyrcel, stock and harmonized.

Question (user directive 2026-10-14): after the thermal-resistance fix on
our side (PR #43/#44) and constant harmonization on the pyrcel side
(L_v/e_s/sigma_w/D_v/k_a/a_T matched to our campaign values), do the two
model types agree?

Inputs
------
- data/gcvi_closure_parcel_ki_postfix.h5  our ki arm, post-fix physics,
  saveat=10 (true S peak resolvable)
- data/leg1_prefix/gcvi_closure_parcel_ki.h5  pre-fix ki arm (leg-1
  archive, saveat=60 — its S_max is the post-peak relaxed value)
- data/m4_harmonize/pyrref_{stock,harmonized}.npz  native pyrcel runs

Outputs
-------
- data/m4_harmonize/verdict_v2.md  metric table
- fig/m4_harmonize.png  4-panel comparison figure
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import figure_style as fs
from .m3_quantification import N_TOTAL, P0, T0
from .stochparticles_io import DATA_DIR, read_scene

CASES = ["chi_0.1", "chi_0.25", "chi_0.4", "chi_0.5", "chi_0.6", "chi_0.75"]
CHI_LABEL = [c.replace("chi_", "") for c in CASES]
REPS = 5
VOL = 1000.0 / N_TOTAL                      # tracked-parcel volume [m^3]
M_AIR = P0 / (287.05 * T0) * VOL             # dry-air mass in that volume

OURS_POST = DATA_DIR / "gcvi_closure_parcel_ki_postfix.h5"
OURS_PRE = DATA_DIR / "leg1_prefix" / "gcvi_closure_parcel_ki.h5"
PYR = {v: DATA_DIR / "m4_harmonize" / f"pyrref_{v}.npz" for v in ("stock", "harmonized")}


def our_metrics(reps):
    """(S_max, t_peak, N_act, q_end, S(t) mean, q(t) mean, time) per rep."""
    out = []
    for rep in reps:
        S, t = rep.arrays["parcel_S"], rep.arrays["time"]
        q = rep.arrays["species_mass_concentration"][:, 4] * VOL / M_AIR
        i = int(np.argmax(S))
        out.append((float(S[i]), float(t[i]),
                    float(rep.arrays["activation_fraction"][-1]), float(q[-1])))
    m = np.mean(out, axis=0)
    tr = np.array([rep.arrays["time"] for rep in reps])
    sm = np.mean([rep.arrays["parcel_S"] for rep in reps], axis=0)
    qm = np.mean([rep.arrays["species_mass_concentration"][:, 4] * VOL / M_AIR
                  for rep in reps], axis=0)
    return m, sm, qm, tr[0]


def pyr_metrics(npz, case):
    per = []
    for i in range(REPS):
        k = f"{case}/rep{i}"
        per.append((float(npz[f"{k}/S_max_full"]), float(npz[f"{k}/t_smax"]),
                    float(npz[f"{k}/Nd_peak_frac"]),
                    float(npz[f"{k}/wc"][-1])))
    return np.mean(per, axis=0)


def spectrum_ours(rep, edges):
    """dN/dlogD of end-state wet diameters (per-particle, equal weights)."""
    d = rep.arrays["diameter_samples"][-1, :]
    w = np.full(d.size, N_TOTAL / d.size)
    h, _ = np.histogram(d, bins=edges, weights=w)
    return h / np.diff(np.log10(edges))


def spectrum_pyr(npz, case, edges):
    """dN/dlogD of pyrcel end-state wet diameters (per-bin N weights)."""
    h = np.zeros(len(edges) - 1)
    for i in range(REPS):
        k = f"{case}/rep{i}"
        d = npz[f"{k}/r_end"] * 2.0
        n = npz[f"{k}/N_cm3"] * 1e6                      # cm^-3 -> m^-3
        hh, _ = np.histogram(d, bins=edges, weights=n)
        h += hh / REPS
    return h / np.diff(np.log10(edges))


def main() -> None:
    ours = read_scene(OURS_POST)
    pre = read_scene(OURS_PRE)
    pyr = {v: np.load(p) for v, p in PYR.items()}

    # ---- metric table -------------------------------------------------
    rows = []
    for case in CASES:
        m_post, _, _, _ = our_metrics(ours[case][:REPS])
        m_pre, _, _, _ = our_metrics(pre[case][:REPS])
        s, h = pyr_metrics(pyr["stock"], case), pyr_metrics(pyr["harmonized"], case)
        rows.append((case, m_pre, m_post, s, h))

    lines = ["# M4 verdict v2: our ki (post-fix) vs pyrcel (stock|harmonized)",
             "",
             "Our side: saveat=10 (true peak); pre-fix column is the leg-1",
             "archive at saveat=60 (post-peak relaxed S — see notes).",
             "",
             "| case | Smax pre | Smax post | Smax pyr-stk | Smax pyr-hrm |"
             " post/hrm | t_pk post | t_pk hrm | Nact post | Nact hrm |"
             " q pre | q post | q hrm |",
             "|---|" + "---|" * 12]
    for case, mp, mo, s, h in rows:
        lines.append(
            f"| {case} | {mp[0]:.5f} | {mo[0]:.5f} | {s[0]:.5f} | {h[0]:.5f} "
            f"| {mo[0] / h[0]:.3f} | {mo[1]:.1f} | {h[1]:.1f} "
            f"| {mo[2]:.4f} | {h[2]:.4f} "
            f"| {mp[3]:.3e} | {mo[3]:.3e} | {h[3]:.3e} |")
    r_post = [mo[0] / h[0] for _, mp, mo, s, h in rows]
    r_q = [mo[3] / h[3] for _, mp, mo, s, h in rows]
    lines += ["",
              f"S_max(post)/S_max(harmonized): median {np.median(r_post):.3f}"
              f" (range {min(r_post):.3f}..{max(r_post):.3f})",
              f"q_liq_end(post)/q_liq_end(harmonized): median {np.median(r_q):.4f}"
              f" (range {min(r_q):.4f}..{max(r_q):.4f})"]
    out_md = DATA_DIR / "m4_harmonize" / "verdict_v2.md"
    out_md.write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- figure -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    case = "chi_0.5"
    m, sm, qm, t = our_metrics(ours[case][:REPS])
    ax = axes[0, 0]
    for rep in ours[case][:REPS]:
        ax.plot(rep.arrays["time"], rep.arrays["parcel_S"] * 100,
                color=fs.PALETTE["aerosol"], lw=0.4, alpha=0.45)
    ax.plot(t, sm * 100, color=fs.PALETTE["aerosol"], lw=1.4,
            label="StochParticles ki (post-fix)")
    ph = np.mean([pyr["harmonized"][f"{case}/rep{i}/S"] for i in range(REPS)], axis=0)
    ps = np.mean([pyr["stock"][f"{case}/rep{i}/S"] for i in range(REPS)], axis=0)
    tp = pyr["harmonized"][f"{case}/rep0/time"]
    ax.plot(tp, ps * 100, "--", color="#7f7f7f", lw=1.0, label="pyrcel stock")
    ax.plot(tp, ph * 100, "-.", color="#d62728", lw=1.0, label="pyrcel harmonized")
    ax.set(xlabel="time [s]", ylabel="supersaturation [%]",
           title=f"(a) S(t), {case}", xlim=(0, 120))

    ax = axes[0, 1]
    ax.plot(t, qm * 1e3, color=fs.PALETTE["aerosol"], lw=1.4,
            label="StochParticles ki (post-fix)")
    qh = np.mean([pyr["harmonized"][f"{case}/rep{i}/wc"] for i in range(REPS)],
                 axis=0)
    qs = np.mean([pyr["stock"][f"{case}/rep{i}/wc"] for i in range(REPS)], axis=0)
    ax.plot(tp, qs * 1e3, "--", color="#7f7f7f", lw=1.0, label="pyrcel stock")
    ax.plot(tp, qh * 1e3, "-.", color="#d62728", lw=1.0, label="pyrcel harmonized")
    ax.set(xlabel="time [s]", ylabel=r"$q_\mathrm{liq}$ [g kg$^{-1}$]",
           title=f"(b) liquid water, {case}")
    ax.legend(fontsize=6)

    ax = axes[1, 0]
    edges = np.logspace(-8.3, -4.5, 96)
    xc = np.sqrt(edges[:-1] * edges[1:])
    ax.plot(xc, spectrum_ours(ours[case][0], edges), color=fs.PALETTE["aerosol"],
            lw=1.2, label="StochParticles ki (post-fix)")
    ax.plot(xc, spectrum_pyr(pyr["harmonized"], case, edges), "-.",
            color="#d62728", lw=1.2, label="pyrcel harmonized")
    ax.plot(xc, spectrum_pyr(pyr["stock"], case, edges), "--", color="#7f7f7f",
            lw=1.0, label="pyrcel stock")
    ax.set_xscale("log")
    ax.set(xlabel="wet diameter [m]", ylabel=r"dN/dlogD [m$^{-3}$]",
           title=f"(c) end-state spectrum, {case}")
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    x = np.arange(len(CASES))
    w = 0.2
    ax.bar(x - 1.5 * w, [r[1][0] for r in rows], w, color="#c7e9c0",
           edgecolor=fs.PALETTE["aerosol"], lw=0.5, label="ki pre-fix")
    ax.bar(x - 0.5 * w, [r[2][0] for r in rows], w, color=fs.PALETTE["aerosol"],
           label="ki post-fix")
    ax.bar(x + 0.5 * w, [r[3][0] for r in rows], w, color="#bdbdbd",
           label="pyrcel stock")
    ax.bar(x + 1.5 * w, [r[4][0] for r in rows], w, color="#d62728",
           label="pyrcel harmonized")
    for i, r in enumerate(rows):
        ax.text(i + 1.5 * w, r[4][0], f"{r[2][0] / r[4][0]:.2f}",
                ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, CHI_LABEL)
    ax.set(xlabel=r"target $\chi$", ylabel="S_max [-]",
           title="(d) S_max attribution")
    ax.legend(fontsize=6, ncol=2)

    for ax in axes.flat:
        ax.tick_params(labelsize=6)
        ax.title.set_size(8)
    fig.tight_layout()
    out_fig = Path(__file__).resolve().parents[1] / "fig" / "m4_harmonize.png"
    out_fig.parent.mkdir(exist_ok=True)
    fig.savefig(out_fig)
    print(f"\nfigure -> {out_fig}")


if __name__ == "__main__":
    main()
