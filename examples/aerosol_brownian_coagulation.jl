# examples/aerosol_brownian_coagulation.jl
using StochParticles
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses
using Plots

# ---- Parameters ----
params = standard_aerosol_atmosphere()
densities = SVector(1800.0)
n_sim = 200
volume = 5.0e-10
tspan = (0.0, 3600.0)

# ---- Initial distribution ----
n_half = div(n_sim, 2)
particles_small = lognormal_masses(n_half, 5.0e-9, 1.5, 1800.0)
particles_large = lognormal_masses(n_sim - n_half, 1.0e-7, 1.5, 1800.0)
particles = vcat(particles_small, particles_large)

# ---- Build problem ----
kernel = BrownianKernel(params.T, params.p, densities)
coag = CoagulationProcess(kernel, GlobalMajorant())
gas_fn = t -> SVector(0.0)
prob = ParticleProblem(particles, volume, gas_fn, (coag,); tspan=tspan, n_sim=n_sim)

# ---- Solve ----
println("Running aerosol Brownian coagulation simulation...")
sol = solve(prob, Tsit5())
@assert sol.retcode == ReturnCode.Success
println("Simulation complete. t_final = $(sol.t[end]) s")

# ---- Validate ----
passed, rel_error = check_mass_conservation(sol, prob)
println("Mass concentration relative error: $rel_error")
@assert passed "Mass concentration not conserved! (rel_error=$rel_error)"

# ---- Plot ----
bin_edges = 10.0 .^ range(-9, -5; length=26)
pl = plot_simulation_summary(sol, prob, bin_edges, 1800.0;
                              time_unit="min", diameter_unit="μm")
savefig(pl, "aerosol_brownian_coagulation.png")
println("Plot saved to aerosol_brownian_coagulation.png")
