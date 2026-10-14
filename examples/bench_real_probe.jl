# Controlled timing probe for the real-spectrum :kinetic arm — WHY is it
# slow, and WHERE does the wall time go?
#
# Axes of attribution:
#   (1) window localization: record_func prints wall-elapsed at every 10-s
#       saveat — front-loaded (haze relaxation, 0-30 s) vs peak (15-25 s)
#       vs tail;
#   (2) solver: Tsit5 explicit (campaign default on main) vs QNDF / Rodas5P
#       with the sparse AD prototype (the ki-enabler path);
#   (3) same population as the campaign (5 species, chi-mixed Dirichlet
#       composition, seed-identical to shard rep 1).
#
# Usage: srun ... julia +1.10.10 --project=examples bench_real_probe.jl
# Env: N_SIM (1000) T_END (120) CHI (0.5) SOLVERS (tsit5,qndf,rodas5p)

include(joinpath(@__DIR__, "simulate_gcvi_closure.jl"))

using OrdinaryDiffEq
using SparseArrays
using Printf

const N_SIM = parse(Int, get(ENV, "N_SIM", "1000"))
const T_END = parse(Float64, get(ENV, "T_END", "120"))
const CHI = parse(Float64, get(ENV, "CHI", "0.5"))
const SOLVERS = split(get(ENV, "SOLVERS", "tsit5,qndf,rodas5p"), ',')

# campaign replica: parcel mode, kinetic gate, w=1.0, C-prime loading
cfg = GcviClosureConfig(n_sim = N_SIM, env_mode = :parcel,
    w = 1.0)
case_spec = population_spec(cfg, CHI)
case_nu = nu_for_chi(case_spec.spectrum, case_spec.fbar, CHI;
    densities = cfg.densities, chi_species = DRY_SPECIES)
all_cases = vcat(cfg.chi_grid, cfg.chi_true)
case_idx = findfirst(==(CHI), all_cases)          # campaign case indexing
seed = cfg.initial_seed_base + 100 * case_idx + 1 # replicate 1
particles0, dry0,
meta = synthesize_population(case_spec;
    seed = seed, thermo = thermo(cfg), nu = case_nu)
@printf("population: n=%d  chi_realized=%.4f  nu=%.4f  seed=%d\n",
    N_SIM, meta.chi_realized, meta.nu, seed)

function run_one(solver, tag; jac = nothing)
    pp = parcel_setup(cfg)
    cond = H2OCondensationProcess(thermo(cfg), cfg.densities;
        h2o_idx = cfg.h2o_idx, w = 0.0)
    t_wall = Ref(time())
    n_rec = Ref(0)
    rec = (t,
        u,
        sys) -> begin
        n_rec[] += 1
        if n_rec[] % 3 == 1                       # print every 30 s
            @printf("  [%s] t=%5.0f s  wall=%7.1f s\n", tag, t, time() - t_wall[])
        end
        (t = t,)
    end
    parts = deepcopy(particles0)
    t_wall[] = time()
    sol,
    recs = solve_split(parts, volume(cfg), ParcelCoupled(pp.parcel),
        (cond, pp), solver; tspan = (0.0, T_END), n_sim = N_SIM,
        dt_split = 10.0, saveat = 10.0, record_func = rec,
        abstol = 1.0e-24, reltol = 1.0e-5,
        clamp_h2o_idx = cfg.h2o_idx, jac_prototype = jac)
    wall = time() - t_wall[]
    @printf("%-14s wall=%8.1f s  ret=%s\n", tag, wall, string(sol.retcode))
    return wall
end

println("probe: n=$N_SIM  chi=$CHI  t_end=$T_END  (campaign rtol 1e-5)")
walls = Dict{String, Float64}()
for s in SOLVERS
    s = strip(s)
    if s == "tsit5"
        walls[s] = run_one(Tsit5(), "tsit5")
    elseif s == "qndf"
        walls[s] = run_one(QNDF(), "qndf-sparse"; jac = rhs_jac_prototype(N_SIM, 5))
    elseif s == "rodas5p"
        walls[s] = run_one(Rodas5P(), "rodas5p-sparse"; jac = rhs_jac_prototype(N_SIM, 5))
    else
        @warn "unknown solver tag $s skipped"
    end
end
for (k, v) in walls
    @printf("SUMMARY %-10s %8.1f s  (t_end=%.0f, extrapolated 600 s: %.0f s)\n",
        k, v, T_END, v * 600.0 / T_END)
end
