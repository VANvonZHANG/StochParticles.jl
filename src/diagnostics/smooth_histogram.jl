# src/diagnostics/smooth_histogram.jl

"""
    smooth_histogram_diameter(diameters, bin_edges, V_t;
                              smooth_factor=3) -> Vector{Float64}

Compute smooth dN/dlogD via oversampled histogram binning + linear interpolation.

Uses fine-grained bins (`smooth_factor` subdivisions per original bin) to produce
a smoother histogram, then averages the fine-bin dN/dlogD values back into each
original bin.  This avoids interpolation artifacts (cubic-spline oscillation /
Runge's phenomenon) that arise with sparse, spiky data.

# Arguments
- `diameters`: particle diameters [m]
- `bin_edges`: diameter bin edges [m], strictly increasing
- `V_t`: computational volume [m³]
- `smooth_factor`: oversampling multiplier for fine bins (default 3)

# Returns
- `Vector{Float64}`: dN/dlogD values at each original bin center [m⁻³]
"""
function smooth_histogram_diameter(
        diameters::Vector{Float64}, bin_edges::Vector{Float64}, V_t::Float64;
        smooth_factor::Int = 3)
    n_bins = length(bin_edges) - 1

    if isempty(diameters)
        return zeros(Float64, n_bins)
    end

    x = log10.(diameters)
    log_edges = log10.(bin_edges)

    # Create oversampled fine bin edges
    n_fine = n_bins * smooth_factor
    fine_edges = collect(range(log_edges[1], log_edges[end]; length = n_fine + 1))

    # Histogram on fine grid using StatsBase (imported in module header)
    h = fit(Histogram, x, fine_edges; closed = :left)
    dlogD_fine = diff(fine_edges)

    # dN/dlogD on fine grid: counts / (log width × volume)
    dNdlogD_fine = Float64.(h.weights) ./ dlogD_fine ./ V_t

    # Average fine-bin dN/dlogD values back into each original bin
    dNdlogD = zeros(Float64, n_bins)
    for j in 1:n_bins
        lo = log_edges[j]
        hi = log_edges[j + 1]
        n_fine_in_bin = 0
        for k in 1:n_fine
            fc = 0.5 * (fine_edges[k] + fine_edges[k + 1])
            if fc >= lo && fc < hi
                dNdlogD[j] += dNdlogD_fine[k]
                n_fine_in_bin += 1
            end
        end
        if n_fine_in_bin > 0
            dNdlogD[j] /= n_fine_in_bin
        end
    end

    return dNdlogD
end
