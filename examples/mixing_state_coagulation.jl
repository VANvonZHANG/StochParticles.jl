# examples/mixing_state_coagulation.jl
using StochParticles
using StaticArrays
using OrdinaryDiffEq
using Printf
using Statistics

println("Multi-species mixing state demo: external → internal mixture via coagulation")

# ---- Physical parameters ----
T = 298.0           # K
p = 1.01325e5       # Pa
V = 1.0e-6          # 1 cm³ computational volume
n_sim = 2000        # target particle count
densities = SVector(1770.0, 1800.0)  # SO4, BC [kg/m³]

# ---- Initial conditions: two pure modes (fully external) ----
d_g_so4 = 1.0e-7    # 100 nm
sigma_so4 = 1.5
so4_fractions = SVector(1.0, 0.0)
so4_particles = lognormal_masses(
    1000, d_g_so4, sigma_so4, densities; fractions = so4_fractions)

d_g_bc = 5.0e-8     # 50 nm
sigma_bc = 1.3
bc_fractions = SVector(0.0, 1.0)
bc_particles = lognormal_masses(
    1000, d_g_bc, sigma_bc, densities; fractions = bc_fractions)

particles = vcat(so4_particles, bc_particles)

# ---- Only coagulation ----
kernel = BrownianKernel(T, p, densities)
proc = CoagulationProcess(kernel, GlobalMajorant())

gas_phase(t) = SVector(0.0, 0.0)  # no gas-phase needed for pure coagulation

prob = ParticleProblem(particles, V, gas_phase, (proc,);
    tspan = (0.0, 3600.0), n_sim = n_sim)

println("Solving...")
sol = solve(prob, Tsit5(); saveat = 60.0)
println("Done. $(length(sol.t)) time steps saved.")

# ---- Extract mixing state evolution ----
chi_t = [mixing_state_index(sol.u[i], prob.prob.p) for i in eachindex(sol.t)]
mean_entropy_t = [
    mean(particle_mixing_entropy(sol.u[i], prob.prob.p))
    for i in eachindex(sol.t)]

# ---- Print summary ----
println("\nMixing state evolution:")
for (t, chi) in zip(sol.t[1:5:end], chi_t[1:5:end])
    println(@sprintf("  t = %6.0f s:  χ = %.4f", t, chi))
end
println(@sprintf("  t = %6.0f s:  χ = %.4f (final)", sol.t[end], chi_t[end]))

# ---- Save HDF5 diagnostics ----
h5_path = joinpath(@__DIR__, "mixing_state_diagnostics.h5")
bin_edges = 10.0 .^ range(-9, -6; length = 31)

init_diagnostics_file(h5_path, 2, bin_edges; species_names = ["SO4", "BC"])
for i in eachindex(sol.t)
    save_diagnostics(h5_path, sol.t[i], sol.u[i], prob.prob.p, Val(2);
        bin_edges = bin_edges, rho = 1000.0)
end
println("\nDiagnostics saved to: $(h5_path)")
