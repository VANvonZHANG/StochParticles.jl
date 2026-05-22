using StochParticles
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses
using Plots

println("Aerosol activation in updraft demo")

# ---- Parameters ----
T0 = 293.15       # Initial temperature [K]
p0 = 1.01325e5    # Initial pressure [Pa]
qv0 = 0.015       # Initial water vapor mixing ratio [kg/kg]
S0 = 0.0          # Initial supersaturation
w = 1.0           # Updraft velocity [m/s]
V = 1.0e-6        # Computational volume [m³]
n_sim = 200       # Target particle count

# ---- Species setup ----
# Species: SO4 (hygroscopic), H2O (water)
densities = SVector(1770.0, 1000.0)  # [kg/m³]
κ_values = SVector(0.61, 0.0)        # κ per species
h2o_idx = 2

# ---- Thermodynamics ----
thermo = ThermodynamicsParams(
    κ_values,
    0.072,      # σ [N/m]
    1000.0,     # ρ_w [kg/m³]
    18.015e-3,  # M_w [kg/mol]
    2.5e6,      # L_v [J/kg]
    461.5,      # R_v [J/kg/K]
    2.5e-5,     # D_v [m²/s]
    2.4e-2,     # k_a [W/m/K]
)

# ---- Initial particles ----
# Log-normal distribution of dry SO4 particles
# Mode at 50 nm, varying sizes to show size-dependent activation
n_half = div(n_sim, 2)
particles_small = lognormal_masses(n_half, 5.0e-8, 1.4, densities[1])
particles_large = lognormal_masses(n_sim - n_half, 1.0e-7, 1.3, densities[1])

# Convert to multi-species: [SO4, H2O]
particles = [SVector{2,Float64}(m[1], 0.0) for m in vcat(particles_small, particles_large)]

# ---- Build problem ----
# H2O condensation only (no coagulation for clarity)
cond = H2OCondensationProcess(thermo, densities; h2o_idx=h2o_idx, w=w)

# Gas phase: returns [T, p_v] from initial parcel state
p_sat_0 = saturation_vapor_pressure(T0)
p_v0 = qv0 * p0 / 0.622  # approximate

# Initial gas phase function
gas_fn = t -> SVector(T0, p_v0)

prob = ParticleProblem(particles, V, gas_fn, (cond,);
    tspan = (0.0, 600.0),  # 10 minutes
    n_sim = n_sim)

# ---- Solve ----
println("Solving (using TRBDF2 for stiffness during activation)...")
sol = solve(prob, TRBDF2(autodiff=false); saveat = 10.0)
println("Done. $(length(sol.t)) time steps saved.")

# ---- Extract activation evolution ----
act_frac = [activation_fraction(sol.u[i], prob.prob.p, Val(2);
    threshold = 1.0e-6, densities = densities) for i in eachindex(sol.t)]

# ---- Print summary ----
println("\nActivation evolution:")
for (t, f) in zip(sol.t[1:10:end], act_frac[1:10:end])
    println("  t = $(round(t, digits=1)) s: activation fraction = $(round(f, digits=3))")
end
println("  t = $(round(sol.t[end], digits=1)) s: activation fraction = $(round(act_frac[end], digits=3))")

# ---- Plot ----
fig = plot(sol.t, act_frac,
    xlabel = "Time (s)",
    ylabel = "Activation Fraction",
    title = "Aerosol Activation in Updraft (w = $(w) m/s)",
    linewidth = 2,
    legend = false)

savefig(fig, joinpath(@__DIR__, "activation_evolution.png"))
println("\nPlot saved to: activation_evolution.png")

# ---- Final diagnostics ----
N_d = cloud_droplet_concentration(sol.u[end], prob.prob.p, Val(2);
    threshold = 1.0e-6, densities = densities)
println("Final cloud droplet concentration: $(round(N_d, sigdigits=3)) m⁻³")
