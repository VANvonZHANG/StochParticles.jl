# src/diagnostics/smooth_histogram.jl
using Interpolations

"""
    smooth_histogram_diameter(diameters, bin_edges, V_t;
                              smooth_factor=3) -> Vector{Float64}

Compute smooth dN/dlogD via oversampled histogram binning + cubic spline interpolation.

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
    fine_centers_vec = 0.5 .* (fine_edges[1:(end - 1)] .+ fine_edges[2:end])

    # Histogram on fine grid using StatsBase (imported in module header)
    h = fit(Histogram, x, fine_edges; closed = :left)
    dlogD_fine = diff(fine_edges)

    # dN/dlogD on fine grid: counts / (log width × volume)
    dNdlogD_fine = Float64.(h.weights) ./ dlogD_fine ./ V_t

    # Cubic spline interpolation with zero extrapolation
    # cubic_spline_interpolation requires AbstractRange for knot positions
    fine_range = range(fine_centers_vec[1], fine_centers_vec[end];
        length = length(fine_centers_vec))
    itp = cubic_spline_interpolation(fine_range, dNdlogD_fine;
        extrapolation_bc = 0.0)

    # Evaluate at original bin centers (geometric mean in log space = arithmetic mean of log)
    orig_log_centers = 0.5 .* (log_edges[1:(end - 1)] .+ log_edges[2:end])
    dNdlogD = itp.(orig_log_centers)

    return max.(dNdlogD, 0.0)
end
