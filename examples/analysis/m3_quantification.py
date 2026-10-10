#!/usr/bin/env python3
"""M3 quantification: four-way comparison (sc / ba / sc_re / kinetic vs pyrcel).

Approximation 2x2 across the campaign arms:
- sc     = frozen haze + frozen edge (legacy gate)
- ba     = frozen haze + branch-aware edge
- sc_re  = re-equilibrated haze (per split step) + frozen edge
- ki     = fully kinetic (no gate) — the fundamental fix, validation arm (2 reps)

External reference: pyrcel's NATIVE parcel model (``pm.ParcelModel().run()``).
Source-verified 2026-10-05: ARG2000 lives in ``pyrcel/activation/_arg2000.py``
and is NOT imported by the parcel-model path; ``out.Nd = summary["total_Nd"]``
is a post-solve trajectory diagnostic. No parameterization is used anywhere.

Cite: Rothenberg & Wang (2016), J. Atmos. Sci., doi:10.1175/JAS-D-15-0223.1
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime

import numpy as np

from .stochparticles_io import DATA_DIR, read_scene

SCENES = {
    "sc": DATA_DIR / "gcvi_closure_parcel_sc.h5",
    "ba": DATA_DIR / "gcvi_closure_parcel_ba.h5",
    "sc_re": DATA_DIR / "gcvi_closure_parcel_sc_re.h5",
    "ki": DATA_DIR / "gcvi_closure_parcel_ki.h5",
}
PROBE = DATA_DIR / "gcvi_closure_probe.h5"
OUT_DIR = DATA_DIR / "m3_quantification"
N_TOTAL = 1.12e10          # C-prime spectrum total [m^-3]
N_TOTAL_CM3 = N_TOTAL * 1e-6
# C-prime environment (user-adjudicated 2026-10-05): /100 loading, w=1.0
T0, P0, S0, V_UP, T_END = 285.0, 90000.0, -0.002, 1.0, 600.0
PYRCEL_REPS = 5


def final_activation(rep) -> float:
    return float(rep.arrays["activation_fraction"][-1])


def s_max(rep) -> float:
    return float(np.max(rep.arrays["parcel_S"]))


def cr_flags_final(rep):
    return rep.arrays["gcvi_cr_flags"][-1, :]


def case_table(scene: dict) -> dict:
    out = {}
    for name in scene:
        reps = scene[name]
        acts = [final_activation(r) for r in reps]
        out[name] = {
            "chi_target": float(np.mean([float(r.attrs["chi_target"]) for r in reps])),
            "N_act": float(np.mean(acts)),
            "N_act_std": float(np.std(acts)),
            "S_max": float(np.mean([s_max(r) for r in reps])),
        }
    return out


def flip_rate(scene_a, scene_b, name) -> float:
    if name not in scene_a or name not in scene_b:
        return float("nan")
    rates = []
    for ra, rb in zip(scene_a[name], scene_b[name]):
        fa, fb = cr_flags_final(ra), cr_flags_final(rb)
        rates.append(float(np.mean(np.logical_xor(fa > 0.5, fb > 0.5))))
    return float(np.mean(rates))


def cr_noise(scene_a, name) -> float:
    counts = [float(np.sum(cr_flags_final(r) > 0.5)) for r in scene_a[name]]
    return float(np.std(counts) / max(np.mean(counts), 1.0))


def pyrcel_species_groups(dry_um, kappas):
    """Kappa-tercile split -> pyrcel AerosolSpecies (dict sectional input,
    radii in um / concentrations in cm^-3 — units verified against Lognorm)."""
    import pyrcel as pm

    n = len(dry_um)
    scale = N_TOTAL_CM3 / n
    edges = np.quantile(kappas, [1 / 3, 2 / 3])
    groups = [kappas <= edges[0],
              (kappas > edges[0]) & (kappas <= edges[1]),
              kappas > edges[1]]
    species = []
    for g in groups:
        if not g.any():
            continue
        d = dry_um[g]
        nb = 12
        be = np.quantile(d, np.linspace(0, 1, nb + 1))
        be[0], be[-1] = be[0] * 0.5, be[-1] * 1.5
        idx = np.clip(np.digitize(d, be) - 1, 0, nb - 1)
        cnt = np.bincount(idx, minlength=nb) * scale
        radii = (np.sqrt(be[:-1] * be[1:]) / 2.0)[cnt > 0]
        species.append(pm.AerosolSpecies(
            "m3", {"r_drys": radii, "Nis": cnt[cnt > 0]},
            kappa=float(np.mean(kappas[g]))))
    return species


def run_pyrcel(rep):
    """(S_max, activated fraction) — NATIVE pyrcel parcel run for this
    replicate's realized population."""
    import pyrcel as pm

    dry_um = rep.arrays["dry_diameter_initial"] * 1e6
    kappas = rep.arrays["kappa_bar_initial"]
    model = pm.ParcelModel(pyrcel_species_groups(dry_um, kappas),
                           V=V_UP, T0=T0, S0=S0, P0=P0)
    out = model.run(t_end=T_END, output_dt=10.0)
    return float(out.summary["S_max"]), float(out.Nd) / N_TOTAL


def chatter_verdict() -> bool:
    scene = read_scene(PROBE)
    name = next(iter(scene))
    violations = []
    for rep in scene[name]:
        s = rep.arrays["parcel_S"]
        tail = s[int(np.argmax(s)):]
        violations.append(int(np.sum(np.diff(tail) > 1e-6)))
    print(f"chatter probe (sc gate, C'): per-replicate post-argmax violations = "
          f"{violations[:10]}{'...' if len(violations) > 10 else ''}")
    return bool(max(violations) >= 3)


def kinetic_validation(scene_re, scene_ki) -> None:
    """Same-seed paired check of the campaign's key assumption:
    per-split-step re-equilibration (cheap) ~ fully kinetic haze (expensive).
    ki has 2 replicates; compare against sc_re's first 2."""
    print("\n=== kinetic vs re-equilibration (same seeds) ===")
    for name in scene_ki:
        for i, rk in enumerate(scene_ki[name]):
            rr = scene_re[name][i]
            ds = abs(s_max(rk) - s_max(rr)) / max(s_max(rr), 1e-12)
            da = abs(final_activation(rk) - final_activation(rr))
            print(f"  {name} rep{i + 1}: S_max ki={s_max(rk):.5f} "
                  f"re={s_max(rr):.5f} (rel {ds:.3%}) | act ki="
                  f"{final_activation(rk):.3f} re={final_activation(rr):.3f} "
                  f"(abs {da:.3f})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pyrcel as pm

    scenes = {}
    for k, p in SCENES.items():
        try:
            scenes[k] = read_scene(p)
        except FileNotFoundError:
            print(f"NOTE: arm '{k}' has no data file — skipping")
    tabs = {k: case_table(s) for k, s in scenes.items()}

    # pyrcel reference per case (from the sc arm's first PYRCEL_REPS replicates
    # — populations are seed-identical across arms, so any arm supplies them)
    ref = {}
    for name in scenes["sc"]:
        vals = [run_pyrcel(r) for r in scenes["sc"][name][:PYRCEL_REPS]]
        ref[name] = {"S_max": float(np.mean([v[0] for v in vals])),
                     "N_act": float(np.mean([v[1] for v in vals]))}
        print(f"pyrcel {name}: S_max={ref[name]['S_max']:.5f} "
              f"N_act={ref[name]['N_act']:.4f}")

    # arms may cover only part of the case grid (ki: ultrafine stiffness in
    # sparse-chi regimes limits coverage — validation arm, not a scan arm)
    d = {arm: {n: abs(tabs[arm][n]["N_act"] - ref[n]["N_act"]) /
                   max(ref[n]["N_act"], 1e-12)
               for n in ref if n in tabs[arm]}
         for arm in tabs}
    coverage = {arm: len(d[arm]) for arm in d}
    full_arms = [a for a in d if coverage[a] == len(ref)]
    med_d = {arm: float(np.median(list(d[arm].values()))) for arm in d}
    max_d = {arm: float(max(d[arm].values())) for arm in d}
    best = min([a for a in full_arms], key=lambda a: med_d[a]) if full_arms else min(med_d, key=med_d.get)
    t1_pass = [arm for arm in d if max_d[arm] < 0.05]
    t1 = best in t1_pass

    mean_flip = float(np.mean([flip_rate(scenes["sc"], scenes["ba"], n)
                               for n in ref]))
    typical_noise = float(np.median([cr_noise(scenes["sc"], n) for n in ref]))
    flips = [flip_rate(scenes["sc"], scenes["ba"], n) for n in ref]
    noises = [cr_noise(scenes["sc"], n) for n in ref]
    mean_flip = float(np.nanmean(flips))
    typical_noise = float(np.median(noises))
    t2 = bool(mean_flip > typical_noise)
    t3 = chatter_verdict()

    if "ki" in scenes:
        kinetic_validation(scenes["sc_re"], scenes["ki"])
    else:
        print("\nNOTE: kinetic arm absent — sc_re-vs-pyrcel closeness is the "
              "only haze validation for now")

    lines = ["# M3 quantification decision — C-prime scenario (spec §5 stage C)",
             "",
             f"generated: {datetime.datetime.now().isoformat()}",
             f"pyrcel: {pm.__version__} — NATIVE ParcelModel runs throughout",
             f"(source-verified 2026-10-05: ARG2000 not imported by the parcel",
             f" path; Nd = trajectory post-solve diagnostic)",
             f"scenario: N_total={N_TOTAL:.2e} m^-3 w={V_UP} T0={T0} P0={P0} S0={S0}",
             "",
             "| case | chi | " + " | ".join(f"N_act({a})" for a in tabs) +
             " | N_act(pyrcel) |",
             "|---|---|" + "---|" * (len(tabs) + 1)]
    for n in ref:
        lines.append(
            f"| {n} | {tabs['sc'][n]['chi_target']:.2f} | " +
            " | ".join(f"{tabs[a][n]['N_act']:.4f}" if n in tabs[a] else "—"
                      for a in tabs) +
            f" | {ref[n]['N_act']:.4f} |")
    lines += ["",
              "S_max means (per-arm over covered cases): " +
              " ".join(f"{a}={np.mean([tabs[a][n]['S_max'] for n in tabs[a]]):.5f}"
                       for a in tabs) +
              f" pyrcel={np.mean([ref[n]['S_max'] for n in ref]):.5f}",
              "",
              "d vs pyrcel (per-arm): " +
              " ".join(f"{a}: median={med_d[a]:.3f} max={max_d[a]:.3f}" for a in d),
              "",
              f"trigger 1 (correctness, best arm = {best}): "
              f"{'FIRE' if t1 else 'silent'}"
              + (f"  (arms within 5%: {t1_pass})" if t1_pass else "  (NO arm within 5%)"),
              f"trigger 2 (science impact, sc-vs-ba flip): {'FIRE' if t2 else 'silent'}"
              f"  (flip={mean_flip:.4f} vs noise={typical_noise:.4f})",
              f"trigger 3 (chatter): {'FIRE' if t3 else 'silent'}",
              "",
              "recommendation: activation_gate default = "
              + (f":kinetic ({best})" if best == "ki" else
                 f":sc_threshold + reequilibrate_haze=true (arm {best})" if best == "sc_re" else
                 f"arm {best} — see kinetic-validation block before deciding"),
              "",
              "## formulation-delta checklist (fill after comparison)",
              "- [ ] cp dry vs moist (ours: CP_DRY_AIR=1005)",
              "- [ ] qsat formula (ours: C-C anchored at 611.2 Pa, eps=0.622)",
              "- [ ] ventilation / Fuchs correction in D_v'",
              f"- [ ] measured |dS_max| best-arm-vs-pyrcel: "
              f"{abs(np.mean([tabs[best][n]['S_max'] for n in ref]) - np.mean([ref[n]['S_max'] for n in ref])):.5f}"]
    (OUT_DIR / "decision.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
