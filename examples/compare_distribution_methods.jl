# # Compare dN/dlogD Distribution Methods
#
# This example compares the three available methods for computing particle size
# distributions (dN/dlogD) WITHOUT running a simulation.  Synthetic diameters
# are drawn from a bimodal lognormal distribution to demonstrate:
#
#   1. How each method renders sparse vs. dense particle data
#   2. The effect of `bandwidth_factor` on KDE smoothness
#   3. The effect of `smooth_factor` on the smoothed-histogram method
#
# Run:  julia --project=examples examples/compare_distribution_methods.jl

using StochParticles
using Plots
using Random

# ---------- configuration ----------
Random.seed!(42)

bin_edges = 10.0 .^ range(-8, -5; length = 21)  # 20 bins, 10 nm – 10 μm
bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])
dlogD = diff(log10.(bin_edges))
V_t = 1.0  # unit volume so number concentration = particle count

# ---------- helpers ----------

"""Generate bimodal lognormal diameters [m]."""
function bimodal_diams(n; d1 = 50e-9, σ1 = 1.3, w1 = 0.7,
        d2 = 500e-9, σ2 = 1.5, w2 = 0.3)
    n1 = round(Int, n * w1 / (w1 + w2))
    n2 = n - n1
    d1_log = log10.(d1 .* σ1 .^ randn(n1))
    d2_log = log10.(d2 .* σ2 .^ randn(n2))
    return 10.0 .^ vcat(d1_log, d2_log)
end

"""Raw histogram dN/dlogD (same as `method = :histogram`)."""
function histogram_dNdlogD(diams, bin_edges, dlogD, V_t)
    counts = StochParticles.bin_size_distribution(diams, bin_edges)
    return Float64.(counts) ./ dlogD ./ V_t
end

# ---------- generate data ----------
diams_sparse = bimodal_diams(100)
diams_dense = bimodal_diams(5000)

# ---------- panel 1: three methods, sparse data ----------
h_sparse = histogram_dNdlogD(diams_sparse, bin_edges, dlogD, V_t)
k_sparse = StochParticles.kde_log_diameter(diams_sparse, bin_centers, V_t)
s_sparse = StochParticles.smooth_histogram_diameter(diams_sparse, bin_edges, V_t)

p1 = plot(bin_centers, h_sparse;
    label = "Histogram", lw = 1.5, ls = :dash, color = :gray,
    xlabel = "Diameter (m)", ylabel = "dN/dlogD (m⁻³)",
    title = "Sparse (n=100)", xscale = :log10,
    legend = :topright, grid = true)
plot!(p1, bin_centers, k_sparse; label = "KDE", lw = 2, color = :steelblue)
plot!(p1, bin_centers, s_sparse; label = "Smooth Histogram",
    lw = 2, ls = :dot, color = :orangered)

# ---------- panel 2: three methods, dense data ----------
h_dense = histogram_dNdlogD(diams_dense, bin_edges, dlogD, V_t)
k_dense = StochParticles.kde_log_diameter(diams_dense, bin_centers, V_t)
s_dense = StochParticles.smooth_histogram_diameter(diams_dense, bin_edges, V_t)

p2 = plot(bin_centers, h_dense;
    label = "Histogram", lw = 1.5, ls = :dash, color = :gray,
    xlabel = "Diameter (m)", ylabel = "dN/dlogD (m⁻³)",
    title = "Dense (n=5000)", xscale = :log10,
    legend = :topright, grid = true)
plot!(p2, bin_centers, k_dense; label = "KDE", lw = 2, color = :steelblue)
plot!(p2, bin_centers, s_dense; label = "Smooth Histogram",
    lw = 2, ls = :dot, color = :orangered)

# ---------- panel 3: KDE bandwidth sensitivity (sparse) ----------
bw_values = [0.3, 1.0, 3.0]
p3 = plot(bin_centers, h_sparse;
    label = "Histogram", lw = 1, ls = :dash, color = :gray,
    xlabel = "Diameter (m)", ylabel = "dN/dlogD (m⁻³)",
    title = "KDE bandwidth_factor (n=100)", xscale = :log10,
    legend = :topright, grid = true)
for bw in bw_values
    k_bw = StochParticles.kde_log_diameter(diams_sparse, bin_centers, V_t;
        bandwidth_factor = bw)
    plot!(p3, bin_centers, k_bw; label = "bf=$(bw)", lw = 2)
end

# ---------- panel 4: smooth_factor sensitivity (dense) ----------
sf_values = [1, 3, 7]
p4 = plot(bin_centers, h_dense;
    label = "Histogram", lw = 1, ls = :dash, color = :gray,
    xlabel = "Diameter (m)", ylabel = "dN/dlogD (m⁻³)",
    title = "Smooth histogram smooth_factor (n=5000)", xscale = :log10,
    legend = :topright, grid = true)
for sf in sf_values
    s_sf = StochParticles.smooth_histogram_diameter(diams_dense, bin_edges, V_t;
        smooth_factor = sf)
    plot!(p4, bin_centers, s_sf; label = "sf=$(sf)", lw = 2)
end

# ---------- combine and save ----------
pl = plot(p1, p2, p3, p4;
    layout = (2, 2), size = (1200, 800),
    plot_title = "dN/dlogD Method Comparison")
savefig(pl, joinpath(@__DIR__, "compare_distribution_methods.png"))
println("Saved: examples/compare_distribution_methods.png")
