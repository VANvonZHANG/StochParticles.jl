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
    for i in 1:sys.n_active
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
    return particle_diameters(u, sys, SVector(rho))
end

"""
    compute_size_distribution(sol, prob, bin_edges, rho; n_snapshots=30)
        -> (snapshot_times, bin_centers, dNdlogD_matrix)

Compute size distribution matrix dN/dlogD over time for heatmap plotting.

# Arguments
- `sol`: SciML solution object
- `prob`: ParticleProblem (JumpProblem)
- `bin_edges`: diameter bin edges [m], strictly increasing
- `rho`: particle density [kg/m³]
- `n_snapshots`: number of time snapshots to evaluate

# Returns
- `snapshot_times::Vector{Float64}`: time points
- `bin_centers::Vector{Float64}`: geometric mean of each bin edge pair [m]
- `dNdlogD_matrix::Matrix{Float64}`: matrix of shape (n_bins, n_snapshots)
"""
function compute_size_distribution(
        sol, prob, bin_edges::Vector{Float64}, rho::Float64; n_snapshots::Int = 30)
    if length(bin_edges) < 2
        throw(ArgumentError("bin_edges must have at least 2 elements"))
    end
    for i in 2:length(bin_edges)
        if bin_edges[i] <= bin_edges[i - 1]
            throw(ArgumentError("bin_edges must be strictly increasing"))
        end
    end

    sys = prob.prob.p
    A = species_val(sys)
    tspan = (sol.t[1], sol.t[end])
    snapshot_times = range(tspan[1], tspan[2]; length = n_snapshots)
    n_bins = length(bin_edges) - 1

    dNdlogD_matrix = zeros(Float64, n_bins, n_snapshots)
    volumes = reconstruct_volumes(sol, prob)
    dlogD = diff(log10.(bin_edges))

    for (j, target_t) in enumerate(snapshot_times)
        t_idx = argmin(abs.(sol.t .- target_t))
        u = sol.u[t_idx]
        V_t = volumes[t_idx]

        diams = particle_diameters(u, sys, rho)
        counts = bin_size_distribution(diams, bin_edges)
        dNdlogD_matrix[:, j] = counts ./ dlogD ./ V_t
    end

    bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])
    return collect(snapshot_times), bin_centers, dNdlogD_matrix
end
