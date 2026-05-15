# src/utils/initial_conditions.jl

using StaticArrays

"""
    lognormal_masses(N, d_g, sigma_g, densities::SVector{A};
                     fractions::SVector{A}=SVector(ntuple(k -> k == 1 ? 1.0 : 0.0, Val(A))))

Generate `N` particle masses drawn from a log-normal diameter distribution.
Each particle has the same species mass fractions.

For multi-modal initial conditions, call separately for each mode and `vcat`.
"""
function lognormal_masses(
        N::Int, d_g::Float64, sigma_g::Float64,
        densities::SVector{A, Float64};
        fractions::SVector{A, Float64} = SVector{A, Float64}(
            ntuple(k -> k == 1 ? 1.0 : 0.0, Val(A)))) where {A}
    ln_dg = log(d_g)
    ln_sigma = log(sigma_g)
    diameters = exp.(ln_dg .+ ln_sigma .* randn(N))
    # Effective density for volume-additive mixing: ρ_eff = 1 / Σ(f_k / ρ_k)
    rho_eff = 1.0 / sum(fractions[k] / densities[k] for k in 1:A)
    masses = @. (π / 6.0) * diameters^3 * rho_eff
    return [masses[i] * fractions for i in 1:N]
end

# Backward-compatible single-species version
function lognormal_masses(
        N::Int, d_g::Float64, sigma_g::Float64, rho::Float64)
    return lognormal_masses(N, d_g, sigma_g, SVector{1, Float64}(rho))
end

"""
    diameters_from_masses(masses, rho) -> Vector{Float64}

Convert particle mass vectors back to diameters [m].
"""
function diameters_from_masses(masses::Vector{SVector{1, Float64}}, rho::Float64)
    return [(6.0 * m[1] / (π * rho))^(1.0 / 3.0) for m in masses]
end
