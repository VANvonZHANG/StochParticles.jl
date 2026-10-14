# M4 ki-benchmark: explicit vs implicit solvers on the :kinetic arm
# (the arm that motivated the AD-transparency stack: explicit Tsit5 is
# stability-bound by millisecond haze relaxation and crawls at n≈1000).
#
# Usage (from repo root):  julia +1.10.10 --project=examples bench_ki_implicit.jl
# Env knobs: N_SIM (default 300), T_END seconds (default 120), W (default 1.0),
#           THERMO the same T-dependent L_v family as the campaign driver.

using StochParticles
using StaticArrays
using OrdinaryDiffEq
using Printf

const N_SIM = parse(Int, get(ENV, "N_SIM", "300"))
const T_END = parse(Float64, get(ENV, "T_END", "120"))
const W_UP = parse(Float64, get(ENV, "W", "1.0"))

T0, P0, S0 = 285.0, 9.0e4, -0.002
N_TOTAL = 1.12e10                  # C-prime loading [m^-3]
V = N_SIM / N_TOTAL                # computational volume [m^3]
M_AIR = P0 / (287.05 * T0) * V     # dry-air mass of the Lagrangian sample [kg]
Tc = T0 - 273.15
thermo = ThermodynamicsParams(SVector(0.61, 0.0), 0.072, 1000.0,
    18.015e-3, 2500.8e3 - 2.36e3 * Tc, 461.5, 2.5e-5, 2.4e-2)
dens = SVector(1770.0, 1000.0)
p_sat0 = saturation_vapor_pressure(T0)
qv0 = (1.0 + S0) * (0.622 * p_sat0) / P0

# population: log-spaced dry diameters 30–500 nm, haze-equilibrated at S0
dd = exp.(range(log(30e-9), log(500e-9); length = N_SIM))
particles = [SVector(4.0 / 3.0 * π * (0.5 * d)^3 * 1770.0, 0.0) for d in dd]
pre_equilibrate!(particles, thermo, dens, T0, p_sat0 * (1.0 + S0); h2o_idx = 2)

rec = (t,
    u,
    sys) -> (
    t = t,
    S = parcel_supersaturation(extract_parcel(u, sys.n_sim, Val(2))),
    mw = sum(get_particle(u, i, Val(2))[2] for i in 1:(sys.n_active)))

function run_arm(solver, tag; jac = nothing)
    pp = ParcelProcess(W_UP, 2, M_AIR, thermo, T0, P0, qv0)
    cond = H2OCondensationProcess(thermo, dens; h2o_idx = 2, w = W_UP)
    parts = deepcopy(particles)
    t0 = time()
    sol,
    recs = solve_split(parts, V, ParcelCoupled(pp.parcel), (cond, pp),
        solver; tspan = (0.0, T_END), n_sim = N_SIM, dt_split = 5.0,
        saveat = T_END, record_func = rec, abstol = 1.0e-24, reltol = 1.0e-8,
        jac_prototype = jac)
    wall = time() - t0
    S_max = maximum(r.S for r in recs)
    @printf("%-15s %9.1f s   ret=%-8s S_max=%.4f%%  m_w_end=%.4e\n", tag, wall,
        string(sol.retcode), 100 * S_max, recs[end].mw)
    return recs[end]
end

println("bench: n=$N_SIM  t_end=$T_END s  w=$W_UP  (kinetic gate, C-prime loading)")
implicit_only = get(ENV, "ARMS", "all") == "implicit"
ref = implicit_only ? nothing : run_arm(Tsit5(), "Tsit5 (expl)")
if !implicit_only
    run_arm(Rodas5(), "Rodas5 dense")
end
r3 = run_arm(Rodas5(), "Rodas5 sparse"; jac = rhs_jac_prototype(N_SIM, 2))
r4 = run_arm(QNDF(), "QNDF sparse"; jac = rhs_jac_prototype(N_SIM, 2))
if ref !== nothing
    @printf("parity vs Tsit5:  Rodas5-sparse dS=%.2e%%  QNDF-sparse dS=%.2e%%\n",
        abs(r3.S - ref.S) * 100, abs(r4.S - ref.S) * 100)
end
