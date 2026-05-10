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

# ---- Plot 2: Size distribution snapshots ----
snapshot_times = [0.0, 600.0, 1800.0, 3600.0]
bin_edges = 10.0 .^ range(-8, -5; length=31)  # 10 nm to 1 μm

pl2 = plot(xlabel = "Diameter [μm]", ylabel = "dN/dlogD [m⁻³]",
    title = "Size Distribution Evolution", xscale = :log10)

for (idx, target_t) in enumerate(snapshot_times)
    # Find closest time index
    t_idx = argmin(abs.(t .- target_t))
    u = sol.u[t_idx]

    # Extract diameters
    diams = Float64[]
    for i in 1:sys.n_active
        μ = get_particle(u, i, A_val)
        d = (6.0 * μ[1] / (π * 1800.0))^(1.0 / 3.0)
        push!(diams, d)
    end

    counts = bin_size_distribution(diams, bin_edges)
    dlogD = diff(log10.(bin_edges))
    dNdlogD = counts ./ dlogD ./ sys.volume

    bin_centers = @. sqrt(bin_edges[1:end-1] * bin_edges[2:end])
    plot!(pl2, bin_centers .* 1.0e6, dNdlogD,
        label = "t = $(Int(target_t/60)) min",
        linewidth = 2,
    )
end

# ---- Save ----
plot(pl1, pl2, layout = (1, 2), size = (1200, 500))
savefig("aerosol_brownian_coagulation.png")
println("Plot saved to aerosol_brownian_coagulation.png")
