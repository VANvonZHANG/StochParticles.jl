# examples/aerosol_brownian_coagulation.jl
using StochParticles
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses
using Plots

include("utils.jl")
using .ExampleUtils

# ---- Parameters ----
params = standard_aerosol_atmosphere()
densities = SVector(1800.0)
n_sim = 200
volume = 5.0e-10  # very small volume → high concentration for visible coagulation
tspan = (0.0, 3600.0)

# ---- Initial distribution ----
# Bimodal: 50% at 5 nm, 50% at 100 nm → fast inter-modal Brownian coagulation
n_half = div(n_sim, 2)
particles_small = lognormal_masses(n_half, 5.0e-9, 1.5, 1800.0)
particles_large = lognormal_masses(n_sim - n_half, 1.0e-7, 1.5, 1800.0)
particles = vcat(particles_small, particles_large)

# Cache initial total mass for correct volume reconstruction
A_val = Val(1)
M_total_0 = sum(sum(μ) for μ in particles)
mass_conc_0 = M_total_0 / volume

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

println("Aerosol example complete.")

# ---- Plot 1: Concentration evolution ----
pl1 = plot(t ./ 60.0, N_conc,
    xlabel = "Time [min]",
    ylabel = "Number concentration [m⁻³]",
    title = "Aerosol Brownian Coagulation",
    label = "N(t)",
    linewidth = 2,
    legend = :topright,
)
plot!(pl1, t ./ 60.0, M_conc ./ maximum(M_conc) .* maximum(N_conc),
    label = "M(t) (normalized)",
    linewidth = 2,
    linestyle = :dash,
)

# ---- Plot 2: Size distribution heatmap ----
# More frequent snapshots for heatmap
n_snapshots = 30
snapshot_times = range(tspan[1], tspan[2]; length=n_snapshots)
bin_edges = 10.0 .^ range(-9, -5; length=26)  # 1 nm to 1 μm
n_bins = length(bin_edges) - 1

# Pre-allocate matrix: rows = bins, cols = time snapshots
dNdlogD_matrix = zeros(Float64, n_bins, n_snapshots)

for (j, target_t) in enumerate(snapshot_times)
    t_idx = argmin(abs.(t .- target_t))
    u = sol.u[t_idx]
    V_t = volumes[t_idx]

    diams = Float64[]
    for i in 1:sys.n_active
        μ = get_particle(u, i, A_val)
        d = (6.0 * μ[1] / (π * 1800.0))^(1.0 / 3.0)
        push!(diams, d)
    end

    counts = bin_size_distribution(diams, bin_edges)
    dlogD = diff(log10.(bin_edges))
    dNdlogD_matrix[:, j] = counts ./ dlogD ./ V_t
end

bin_centers = @. sqrt(bin_edges[1:end-1] * bin_edges[2:end])

pl2 = heatmap(snapshot_times ./ 60.0, bin_centers .* 1.0e6, dNdlogD_matrix,
    xlabel = "Time [min]",
    ylabel = "Diameter [μm]",
    title = "Size Distribution Evolution",
    color = :viridis,
    yscale = :log10,
    yticks = ([0.001, 0.01, 0.1, 1.0], ["0.001", "0.01", "0.1", "1"]),
    colorbar_title = "dN/dlogD [m⁻³]",
)

# ---- Save ----
plot(pl1, pl2, layout = (1, 2), size = (1400, 500))
savefig("aerosol_brownian_coagulation.png")
println("Plot saved to aerosol_brownian_coagulation.png")
