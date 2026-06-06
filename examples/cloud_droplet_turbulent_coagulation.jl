# examples/cloud_droplet_turbulent_coagulation.jl
using StochParticles
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses
using Plots
using Printf

# ---- Output directory (this file's location) ----
outdir = @__DIR__

# ---- Parameters ----
params = standard_cloud_atmosphere()
densities = SVector(1000.0)
n_sim = 200
volume = 1.0e-6
tspan = (0.0, 200.0)
epsilon = 0.01
R_lambda = 50.0

# ---- Initial distribution ----
n_half = div(n_sim, 2)
particles_small = lognormal_masses(n_half, 5.0e-6, 1.3, 1000.0)
particles_large = lognormal_masses(n_sim - n_half, 2.5e-5, 1.3, 1000.0)
particles = vcat(particles_small, particles_large)

# ---- Build composite kernel ----
kernel = make_kernel(params, epsilon, R_lambda, densities)
coag = CoagulationProcess(kernel, GlobalMajorant())
gas_fn = t -> SVector(0.0)
prob = ParticleProblem(particles, volume, gas_fn, (coag,); tspan = tspan, n_sim = n_sim)

# ---- Solve ----
println("Running cloud droplet turbulent coagulation simulation...")
sol = solve(prob, Tsit5())
@assert sol.retcode == ReturnCode.Success
println("Simulation complete. t_final = $(sol.t[end]) s")

# ---- Validate ----
passed, rel_error = check_mass_conservation(sol, prob)
println("Mass concentration relative error: $rel_error")
@assert passed "Mass concentration not conserved! (rel_error=$rel_error)"

# ---- Diagnostics export ----
bin_edges = 10.0 .^ range(-6, -3; length = 21)
sys = prob.prob.p
A = 1

h5_path = joinpath(outdir, "cloud_droplet_turbulent_coagulation.h5")
init_diagnostics_file(h5_path, A, bin_edges;
    species_names = ["H2O"], chunk_size = 64)

for (t, u) in zip(sol.t, sol.u)
    save_diagnostics(h5_path, t, u, sys, Val(A);
        bin_edges = bin_edges, rho = 1000.0)
end
println("Diagnostics saved to $h5_path")

# ---- Plot summary (concentration + size distribution) ----
pl_summary = plot_simulation_summary(sol, prob, bin_edges, 1000.0;
    method = :kde, bandwidth_factor = 1.0)

# ---- Kernel contribution comparison ----
K_brown = BrownianKernel(params.T, params.p, densities)
K_grav = GravitationalKernel(params.mu_f, params.rho_f, params.rho_p, params.g, densities)
K_turb = AyalaTurbulentKernel(
    epsilon, R_lambda, params.nu, params.rho_f, params.rho_p, params.g, densities)

μ_10um = SVector((π / 6.0) * (1.0e-5)^3 * 1000.0)
μ_20um = SVector((π / 6.0) * (2.0e-5)^3 * 1000.0)

kb = K_brown(μ_10um, μ_20um)
kg = K_grav(μ_10um, μ_20um)
kt = K_turb(μ_10um, μ_20um)
ktotal = kernel(μ_10um, μ_20um)

println("\nKernel contributions (10 μm + 20 μm pair):")
println("  Brownian:   $(@sprintf("%.3e", kb)) m³/s  ($(@sprintf("%.1f", kb/ktotal*100))%)")
println("  Gravitational: $(@sprintf("%.3e", kg)) m³/s  ($(@sprintf("%.1f", kg/ktotal*100))%)")
println("  Turbulent:  $(@sprintf("%.3e", kt)) m³/s  ($(@sprintf("%.1f", kt/ktotal*100))%)")
println("  Total:      $(@sprintf("%.3e", ktotal)) m³/s")

pl_kernel = plot_kernel_contributions(
    ["Brownian", "Gravitational", "Turbulent"],
    [kb, kg, kt]
)

# ---- Combine and save ----
pl = plot(pl_summary, pl_kernel, layout = (2, 1), size = (1400, 1000))
fig_path = joinpath(outdir, "cloud_droplet_turbulent_coagulation.png")
savefig(pl, fig_path)
println("\nPlot saved to $fig_path")
