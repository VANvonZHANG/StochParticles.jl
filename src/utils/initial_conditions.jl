# src/utils/initial_conditions.jl

using StaticArrays

"""
    lognormal_masses(N, d_g, sigma_g, rho) -> Vector{SVector{1,Float64}}

Generate `N` particle masses drawn from a log-normal diameter distribution.
"""
function lognormal_masses(N::Int, d_g::Float64, sigma_g::Float64, rho::Float64)
    ln_dg = log(d_g)
    ln_sigma = log(sigma_g)
    diameters = exp.(ln_dg .+ ln_sigma .* randn(N))
    masses = @. (π / 6.0) * diameters^3 * rho
    return [SVector{1,Float64}(m) for m in masses]
end

"""
    diameters_from_masses(masses, rho) -> Vector{Float64}

Convert particle mass vectors back to diameters [m].
"""
function diameters_from_masses(masses::Vector{SVector{1,Float64}}, rho::Float64)
    return [(6.0 * m[1] / (π * rho))^(1.0 / 3.0) for m in masses]
end
