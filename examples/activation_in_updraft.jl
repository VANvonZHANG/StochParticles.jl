using StochParticles
using StaticArrays
using OrdinaryDiffEq
using Plots
using Printf

println("=== Aerosol CCN Activation: Size-Resolved Experiment ===")
println("Fixed supersaturation, comparison with κ-Köhler theory\n")

# ============================================================
# 1. Physical parameters
# ============================================================
T0 = 293.15          # Temperature [K]
p0 = 1.01325e5       # Pressure [Pa]
V = 1.0e-6           # Computational volume [m³] (1 cm³)
n_sim = 2000         # Number of particles

# Species: SO4 (ammonium sulfate proxy), H2O (water)
densities = SVector(1770.0, 1000.0)   # [kg/m³]
h2o_idx = 2

# ============================================================
# 2. Initial aerosol: two atmospheric modes (Whitby 1978)
# ============================================================
n_aitken = div(n_sim, 2)
n_accum = n_sim - n_aitken

# Aitken mode — use κ=0.30 for this sub-population
aitken_κ = SVector(0.30, 0.0)
aitken_thermo = ThermodynamicsParams(
    aitken_κ, 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)

# Accumulation mode — use κ=0.61
accum_κ = SVector(0.61, 0.0)
accum_thermo = ThermodynamicsParams(
    accum_κ, 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)

particles_aitken = lognormal_masses(n_aitken, 3.0e-8, 1.6, densities)
particles_accum = lognormal_masses(n_accum, 1.2e-7, 1.8, densities)

# Convert to multi-species: [SO4, H2O] — initially dry (no water)
particles = [SVector{2, Float64}(m[1], 0.0)
             for m in vcat(particles_aitken, particles_accum)]

# Record dry diameters at t=0 for later size-resolved analysis
dry_diams_aitken = [d[1] for d in diameters_from_masses(particles_aitken, densities)]
dry_diams_accum = [d[1] for d in diameters_from_masses(particles_accum, densities)]
dry_diams = vcat(dry_diams_aitken, dry_diams_accum)

# Assign per-particle thermo (Aitken vs accumulation) for Sc computation
particle_thermo = vcat(
    fill(aitken_thermo, n_aitken),
    fill(accum_thermo, n_accum)
)

# ============================================================
# 3. Run simulation at a fixed supersaturation
# ============================================================
S_target = 0.003   # 0.3% supersaturation
p_sat_0 = saturation_vapor_pressure(T0)
p_v0 = p_sat_0 * (1.0 + S_target)

gas_fn = t -> SVector(T0, p_v0)  # Fixed environment

# Simplification: use average κ for condensation (production code would use per-particle κ)
avg_thermo = ThermodynamicsParams(
    SVector(0.455, 0.0),  # average of 0.30 and 0.61
    0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)
cond = H2OCondensationProcess(avg_thermo, densities; h2o_idx = h2o_idx, w = 0.0)

# Pre-equilibrate non-activated particles to Köhler equilibrium (QSSA)
pre_equilibrate!(particles, avg_thermo, densities, T0, p_v0; h2o_idx = h2o_idx)

prob = ParticleProblem(particles, V, gas_fn, (cond,);
    tspan = (0.0, 600.0), n_sim = n_sim)

println("Solving at S = $(@sprintf("%.2f%%", S_target*100)) with $(n_sim) particles...")
sol = solve(prob, TRBDF2(autodiff = false); saveat = [0.0, 600.0])
println("Done.\n")

# ============================================================
# 4. Determine activation status: Köhler theory criterion
# ============================================================
# A particle is activated if S_env > Sc(d_dry, κ).
is_activated = falses(n_sim)
for i in 1:n_sim
    m_dry_i = SVector{2, Float64}(particles[i][1], 0.0)
    Sc_i = critical_supersaturation(m_dry_i, particle_thermo[i], densities, T0)
    is_activated[i] = S_target > Sc_i
end

n_activated = count(is_activated)
frac_activated = n_activated / n_sim
N_ccn = frac_activated * n_sim / V

println("--- Results at S = $(@sprintf("%.2f%%", S_target*100)) ---")
println("Activated:      $(n_activated) / $(n_sim) ($(round(frac_activated*100, digits=1))%)")
println("CCN concentration: $(@sprintf("%.2e", N_ccn)) m⁻³")

# ============================================================
# 5. Size-resolved activation fraction (simulation)
# ============================================================
bin_edges = 10.0 .^ range(-8.3, -6.3; length = 20)
n_bins = length(bin_edges) - 1
bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])

act_count = zeros(Int, n_bins)
total_count = zeros(Int, n_bins)
for i in 1:n_sim
    d_dry = dry_diams[i]
    for b in 1:n_bins
        if bin_edges[b] <= d_dry < bin_edges[b + 1]
            total_count[b] += 1
            if is_activated[i]
                act_count[b] += 1
            end
            break
        end
    end
end

act_frac_sim = zeros(n_bins)
for b in 1:n_bins
    if total_count[b] > 0
        act_frac_sim[b] = act_count[b] / total_count[b]
    end
end

# ============================================================
# 6. Theoretical Köhler curve: Sc(d_dry) at fixed κ
# ============================================================
d_theory = 10.0 .^ range(-8.5, -6.5; length = 200)

Sc_aitken = map(d_theory) do d
    m_dry = SVector{2, Float64}((π/6.0) * d^3 * densities[1], 0.0)
    critical_supersaturation(m_dry, aitken_thermo, densities, T0)
end

Sc_accum = map(d_theory) do d
    m_dry = SVector{2, Float64}((π/6.0) * d^3 * densities[1], 0.0)
    critical_supersaturation(m_dry, accum_thermo, densities, T0)
end

# ============================================================
# 7. Plot: Size-resolved activation + Köhler theory
# ============================================================
fig = plot(
    dpi = 150,
    size = (900, 700),
    layout = grid(2, 1, heights = [0.55, 0.45])
)

# Top panel: Köhler Sc(d) curves + S_target line
plot!(fig[1],
    d_theory * 1e9, Sc_aitken * 100,
    label = "Sc (κ=0.30, Aitken)",
    linewidth = 2, color = :steelblue,
    xscale = :log10, yscale = :log10,
    xlabel = "Dry diameter [nm]",
    ylabel = "Critical supersaturation Sc [%]",
    title = "κ-Köhler Theory vs Simulation (S = $(@sprintf("%.2f%%", S_target*100)))",
    legend = :topright,
    xlim = (1, 1000),
    ylim = (0.01, 10)
)

plot!(fig[1],
    d_theory * 1e9, Sc_accum * 100,
    label = "Sc (κ=0.61, Accumulation)",
    linewidth = 2, color = :darkorange
)

hline!(fig[1],
    [S_target * 100],
    label = "S_env = $(@sprintf("%.2f%%", S_target*100))",
    linewidth = 2, linestyle = :dash, color = :red
)

# Bottom panel: Size-resolved activation fraction
plot!(fig[2],
    bin_centers * 1e9, act_frac_sim,
    seriestype = :steppre,
    label = "Simulation (activated fraction)",
    linewidth = 2, color = :black,
    xscale = :log10,
    xlabel = "Dry diameter [nm]",
    ylabel = "Activation fraction",
    title = "Size-Resolved Activation",
    legend = :bottomright,
    xlim = (1, 1000),
    ylim = (-0.05, 1.1)
)

# Find theoretical cutoff diameters by interpolation
d_cut_aitken = NaN
d_cut_accum = NaN
for i in 1:(length(d_theory) - 1)
    if Sc_aitken[i] >= S_target && Sc_aitken[i + 1] < S_target
        f = log(S_target / Sc_aitken[i]) / log(Sc_aitken[i + 1] / Sc_aitken[i])
        global d_cut_aitken = d_theory[i] * (d_theory[i + 1] / d_theory[i])^f
    end
    if Sc_accum[i] >= S_target && Sc_accum[i + 1] < S_target
        f = log(S_target / Sc_accum[i]) / log(Sc_accum[i + 1] / Sc_accum[i])
        global d_cut_accum = d_theory[i] * (d_theory[i + 1] / d_theory[i])^f
    end
end

if !isnan(d_cut_aitken)
    vline!(fig[2], [d_cut_aitken * 1e9],
        label = "Theory cutoff κ=0.30 ($(@sprintf("%.0f", d_cut_aitken*1e9)) nm)",
        linestyle = :dash, color = :steelblue, linewidth = 1.5)
end
if !isnan(d_cut_accum)
    vline!(fig[2], [d_cut_accum * 1e9],
        label = "Theory cutoff κ=0.61 ($(@sprintf("%.0f", d_cut_accum*1e9)) nm)",
        linestyle = :dash, color = :darkorange, linewidth = 1.5)
end

savefig(fig, joinpath(@__DIR__, "activation_size_resolved.png"))
println("\nPlot saved to: activation_size_resolved.png")

if !isnan(d_cut_aitken)
    println("Theoretical cutoff (κ=0.30): $(@sprintf("%.1f", d_cut_aitken*1e9)) nm")
end
if !isnan(d_cut_accum)
    println("Theoretical cutoff (κ=0.61): $(@sprintf("%.1f", d_cut_accum*1e9)) nm")
end
