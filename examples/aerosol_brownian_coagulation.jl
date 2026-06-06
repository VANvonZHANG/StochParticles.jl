# examples/aerosol_brownian_coagulation.jl
using StochParticles
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses
using Plots

# ---- Output directory (this file's location) ----
outdir = @__DIR__

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
prob = ParticleProblem(particles, volume, gas_fn, (coag,); tspan = tspan, n_sim = n_sim)

# ---- Solve ----
println("Running aerosol Brownian coagulation simulation...")
sol = solve(prob, Tsit5())
@assert sol.retcode == ReturnCode.Success
println("Simulation complete. t_final = $(sol.t[end]) s")

# ---- Validate ----
passed, rel_error = check_mass_conservation(sol, prob)
println("Mass concentration relative error: $rel_error")
@assert passed "Mass concentration not conserved! (rel_error=$rel_error)"

# ---- Diagnostics export ----
bin_edges = 10.0 .^ range(-9, -5; length = 26)
sys = prob.prob.p
A = 1

h5_path = joinpath(outdir, "aerosol_brownian_coagulation.h5")
init_diagnostics_file(h5_path, A, bin_edges;
    species_names = ["SO4"], chunk_size = 64)

for (t, u) in zip(sol.t, sol.u)
    save_diagnostics(h5_path, t, u, sys, Val(A);
        bin_edges = bin_edges, rho = 1800.0)
end
println("Diagnostics saved to $h5_path")

# ---- Plot ----
pl = plot_simulation_summary(sol, prob, bin_edges, 1800.0;
    time_unit = "min", diameter_unit = "μm",
    method = :kde, bandwidth_factor = 1.0)
fig_path = joinpath(outdir, "aerosol_brownian_coagulation.png")
savefig(pl, fig_path)
println("Plot saved to $fig_path")
