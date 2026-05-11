# examples/cloud_droplet_turbulent_coagulation.jl
using StochParticles
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses
using Plots
using Printf

include("utils.jl")
using .ExampleUtils

# ---- Parameters ----
params = standard_cloud_atmosphere()
densities = SVector(1000.0)
n_sim = 200
volume = 1.0e-6  # high concentration for visible coagulation
tspan = (0.0, 200.0)
epsilon = 0.01
R_lambda = 50.0

# ---- Initial distribution ----
# Bimodal: 50% at 5 μm, 50% at 25 μm → activates gravitational/turbulent coagulation
n_half = div(n_sim, 2)
particles_small = lognormal_masses(n_half, 5.0e-6, 1.3, 1000.0)
particles_large = lognormal_masses(n_sim - n_half, 2.5e-5, 1.3, 1000.0)
particles = vcat(particles_small, particles_large)

# Cache initial total mass for correct volume reconstruction
A_val = Val(1)
M_total_0 = sum(sum(μ) for μ in particles)
mass_conc_0 = M_total_0 / volume

# ---- Build composite kernel ----
kernel = make_kernel(params, epsilon, R_lambda, densities)
coag = CoagulationProcess(kernel, GlobalMajorant())
gas_fn = t -> SVector(0.0)
prob = ParticleProblem(particles, volume, gas_fn, (coag,); tspan=tspan, n_sim=n_sim)

# ---- Solve ----
println("Running cloud droplet turbulent coagulation simulation...")
sol = solve(prob, Tsit5())
@assert sol.retcode == ReturnCode.Success
println("Simulation complete. t_final = $(sol.t[end]) s")

# ---- Diagnostics with correct volume reconstruction ----
sys = prob.prob.p
t = sol.t
n = length(t)
N_conc = Vector{Float64}(undef, n)
M_conc = Vector{Float64}(undef, n)
volumes = Vector{Float64}(undef, n)

for i in 1:n
    u = sol.u[i]
    M_total_t = total_mass(u, A_val, sys.n_active)
    V_t = M_total_t * volume / M_total_0
    volumes[i] = V_t
    N_conc[i] = sys.n_sim / V_t
    M_conc[i] = M_total_t / V_t
end

# ---- Validation ----
mass_rel_error = maximum(abs.(M_conc .- mass_conc_0)) / mass_conc_0
println("Mass concentration relative error: $mass_rel_error")
@assert mass_rel_error < 1e-3 "Mass concentration not conserved! (rel_error=$mass_rel_error)"

# ---- Kernel contribution comparison ----
K_brown = BrownianKernel(params.T, params.mu_f, densities)
K_grav  = GravitationalKernel(params.mu_f, params.rho_f, params.rho_p, params.g, densities)
K_turb  = AyalaTurbulentKernel(epsilon, R_lambda, params.nu, params.rho_f, params.rho_p, params.g, densities)

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

# ---- Plot 1: Concentration evolution ----
pl1 = plot(t, N_conc,
    xlabel = "Time [s]",
    ylabel = "Number concentration [m⁻³]",
    title = "Cloud Droplet Composite Coagulation",
    label = "N(t)",
    linewidth = 2,
    legend = :topright,
)
plot!(pl1, t, M_conc ./ maximum(M_conc) .* maximum(N_conc),
    label = "M(t) (normalized)",
    linewidth = 2,
    linestyle = :dash,
)

# ---- Plot 2: Size distribution heatmap ----
n_snapshots = 30
snapshot_times = range(tspan[1], tspan[2]; length=n_snapshots)
bin_edges = 10.0 .^ range(-6, -3; length=21)  # 1 μm to 1 mm
n_bins = length(bin_edges) - 1

dNdlogD_matrix = zeros(Float64, n_bins, n_snapshots)

for (j, target_t) in enumerate(snapshot_times)
    t_idx = argmin(abs.(t .- target_t))
    u = sol.u[t_idx]
    V_t = volumes[t_idx]

    diams = Float64[]
    for i in 1:sys.n_active
        μ = get_particle(u, i, A_val)
        d = (6.0 * μ[1] / (π * 1000.0))^(1.0 / 3.0)
        push!(diams, d)
    end

    counts = bin_size_distribution(diams, bin_edges)
    dlogD = diff(log10.(bin_edges))
    dNdlogD_matrix[:, j] = counts ./ dlogD ./ V_t
end

bin_centers = @. sqrt(bin_edges[1:end-1] * bin_edges[2:end])

pl2 = heatmap(snapshot_times, bin_centers .* 1.0e6, dNdlogD_matrix,
    xlabel = "Time [s]",
    ylabel = "Diameter [μm]",
    title = "Size Distribution Evolution",
    color = :viridis,
    yscale = :log10,
    yticks = ([1.0, 10.0, 100.0], ["1", "10", "100"]),
    colorbar_title = "dN/dlogD [m⁻³]",
)

# ---- Plot 3: Kernel contribution bar chart ----
pl3 = bar(["Brownian", "Gravitational", "Turbulent"],
    [kb, kg, kt],
    ylabel = "Coagulation rate [m³/s]",
    title = "Kernel Contributions (10+20 μm)",
    legend = false,
    yscale = :log10,
)

# ---- Save ----
plot(pl1, pl2, pl3, layout = (1, 3), size = (1800, 500))
savefig("cloud_droplet_turbulent_coagulation.png")
println("\nPlot saved to cloud_droplet_turbulent_coagulation.png")
