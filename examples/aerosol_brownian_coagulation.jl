# examples/aerosol_brownian_coagulation.jl
using StochParticles
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses

include("utils.jl")
using .ExampleUtils

# ---- Parameters ----
params = standard_aerosol_atmosphere()
densities = SVector(1800.0)
n_sim = 5000
volume = 1.0
tspan = (0.0, 3600.0)

# ---- Initial distribution ----
particles = lognormal_masses(n_sim, 1.0e-7, 1.5, 1800.0)

# ---- Build problem ----
kernel = BrownianKernel(params.T, params.mu_f, densities)
coag = CoagulationProcess(kernel, GlobalMajorant())
gas_fn = t -> SVector(0.0)
prob = ParticleProblem(particles, volume, gas_fn, (coag,); tspan=tspan, n_sim=n_sim)

# ---- Solve ----
println("Running aerosol Brownian coagulation simulation...")
sol = solve(prob, Tsit5())
@assert sol.retcode == ReturnCode.Success
println("Simulation complete. t_final = $(sol.t[end]) s")

# ---- Diagnostics ----
sys = prob.prob.p
A_val = Val(1)
t, N_conc, M_conc = extract_diagnostics(sol, sys, A_val)

# ---- Validation ----
mass_rel_error = abs(M_conc[end] - M_conc[1]) / M_conc[1]
println("Mass concentration relative error: $mass_rel_error")
@assert mass_rel_error < 1e-6 "Mass concentration not conserved!"

println("Aerosol example complete.")
