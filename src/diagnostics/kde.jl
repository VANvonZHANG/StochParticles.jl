# src/diagnostics/kde.jl
using KernelDensity
using Interpolations

"""
    kde_log_diameter(diameters, bin_centers, V_t;
                     bandwidth_factor=1.0, n_eval_points=200) -> Vector{Float64}

Compute smooth dN/dlogD via Gaussian KDE on log10(diameter) space.

# Arguments
- `diameters`: particle diameters [m]
- `bin_centers`: bin center diameters [m] where dN/dlogD is evaluated
- `V_t`: computational volume [m³]
- `bandwidth_factor`: multiplier for Silverman bandwidth (default 1.0)
- `n_eval_points`: number of fine-grid evaluation points (default 200)

# Returns
- `Vector{Float64}`: dN/dlogD values at each bin center [m⁻³]
"""
function kde_log_diameter(
        diameters::Vector{Float64}, bin_centers::Vector{Float64}, V_t::Float64;
        bandwidth_factor::Float64 = 1.0, n_eval_points::Int = 200)
    if isempty(diameters)
        return zeros(Float64, length(bin_centers))
    end

    x = log10.(diameters)
    n_particles = length(x)
    number_conc = n_particles / V_t

    # Silverman bandwidth: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
    sigma = std(x)
    if sigma == 0.0
        # All particles have the same diameter — use a small default bandwidth
        h = 0.1 * bandwidth_factor
    else
        iqr_val = quantile(x, 0.75) - quantile(x, 0.25)
        spread = iqr_val > 0 ? min(sigma, iqr_val / 1.34) : sigma
        h = 0.9 * spread * n_particles^(-0.2) * bandwidth_factor
    end

    # Evaluate KDE on a fine grid spanning the data range (with padding)
    eval_points = range(minimum(x) - 3h, maximum(x) + 3h; length = n_eval_points)
    kde_result = kde(x, eval_points; bandwidth = h)

    # Convert probability density to dN/dlogD
    dNdlogD_fine = kde_result.density .* number_conc

    # Interpolate to bin center positions
    itp = linear_interpolation(eval_points, dNdlogD_fine; extrapolation_bc = 0.0)
    bin_log = log10.(bin_centers)
    dNdlogD = itp.(bin_log)

    return max.(dNdlogD, 0.0)
end
