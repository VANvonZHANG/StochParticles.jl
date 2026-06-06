# src/diagnostics/distributions.jl

"""
    particle_diameters(u, sys, densities) -> Vector{Float64}

Compute sphere-equivalent diameters [m] for all active particles from their masses.

For multi-species particles, the equivalent volume is computed as:
    V_p = Σ μ_k / ρ_k

# Arguments
- `densities::SVector{A, Float64}`: per-species densities [kg/m³]
"""
function particle_diameters(
        u::Vector{Float64}, sys::ParticleSystem, densities::SVector{A, Float64}) where {A}
    A_val = species_val(sys)
    diams = Vector{Float64}(undef, sys.n_active)
    for i in 1:(sys.n_active)
        μ = get_particle(u, i, A_val)
        V_p = 0.0
        for k in 1:A
            V_p += μ[k] / densities[k]
        end
        diams[i] = (6.0 * V_p / π)^(1.0 / 3.0)
    end
    return diams
end

# Backward-compatible single-species shortcut
function particle_diameters(
        u::Vector{Float64}, sys::ParticleSystem, rho::Float64)
    return particle_diameters(u, sys, SVector{1, Float64}(rho))
end

"""
    compute_size_distribution(sol, prob, bin_edges, rho;
                              n_snapshots=30, method=:kde,
                              bandwidth_factor=1.0, n_eval_points=200,
                              smooth_factor=3)
        -> (snapshot_times, bin_centers, dNdlogD_matrix)

Compute size distribution matrix dN/dlogD over time.

# Arguments
- `sol`: SciML solution object
- `prob`: ParticleProblem (JumpProblem)
- `bin_edges`: diameter bin edges [m], strictly increasing
- `rho`: particle density [kg/m³]
- `n_snapshots`: number of time snapshots to evaluate (default 30)
- `method`: `:histogram`, `:histogram_smooth`, or `:kde` (default `:kde`)
- `bandwidth_factor`: KDE bandwidth multiplier (default 1.0, only for `:kde`)
- `n_eval_points`: KDE evaluation grid size (default 200, only for `:kde`)
- `smooth_factor`: histogram oversampling factor (default 3, only for `:histogram_smooth`)

# Returns
- `snapshot_times::Vector{Float64}`: time points
- `bin_centers::Vector{Float64}`: geometric mean of each bin edge pair [m]
- `dNdlogD_matrix::Matrix{Float64}`: matrix of shape (n_bins, n_snapshots)
"""
function compute_size_distribution(
        sol, prob, bin_edges::Vector{Float64}, rho::Float64;
        n_snapshots::Int = 30,
        method::Symbol = :kde,
        bandwidth_factor::Float64 = 1.0,
        n_eval_points::Int = 200,
        smooth_factor::Int = 3)
    if length(bin_edges) < 2
        throw(ArgumentError("bin_edges must have at least 2 elements"))
    end
    for i in 2:length(bin_edges)
        if bin_edges[i] <= bin_edges[i - 1]
            throw(ArgumentError("bin_edges must be strictly increasing"))
        end
    end

    if method ∉ (:histogram, :histogram_smooth, :kde)
        throw(ArgumentError(
            "method must be :histogram, :histogram_smooth, or :kde, got :$method"))
    end

    sys = prob.prob.p
    A = species_val(sys)
    tspan = (sol.t[1], sol.t[end])
    snapshot_times = range(tspan[1], tspan[2]; length = n_snapshots)
    n_bins = length(bin_edges) - 1

    dNdlogD_matrix = zeros(Float64, n_bins, n_snapshots)
    volumes = reconstruct_volumes(sol, prob)
    dlogD = diff(log10.(bin_edges))
    bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])

    for (j, target_t) in enumerate(snapshot_times)
        t_idx = argmin(abs.(sol.t .- target_t))
        u = sol.u[t_idx]
        V_t = volumes[t_idx]

        if sys.n_active == 0
            continue
        end

        diams = particle_diameters(u, sys, rho)

        if method == :histogram
            counts = bin_size_distribution(diams, bin_edges)
            dNdlogD_matrix[:, j] = Float64.(counts) ./ dlogD ./ V_t

        elseif method == :kde
            dNdlogD_matrix[:, j] = kde_log_diameter(
                diams, bin_centers, V_t;
                bandwidth_factor = bandwidth_factor,
                n_eval_points = n_eval_points)

        elseif method == :histogram_smooth
            dNdlogD_matrix[:, j] = smooth_histogram_diameter(
                diams, bin_edges, V_t;
                smooth_factor = smooth_factor)
        end
    end

    return collect(snapshot_times), bin_centers, dNdlogD_matrix
end
