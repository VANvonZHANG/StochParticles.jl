# examples/utils.jl
module ExampleUtils

using StochParticles
using StaticArrays

# Placeholder exports
export lognormal_masses, diameters_from_masses
export bin_size_distribution, extract_diagnostics
export standard_aerosol_atmosphere, standard_cloud_atmosphere

"""
    lognormal_masses(N, d_g, sigma_g, rho) -> Vector{SVector{1,Float64}}

Generate `N` particle masses drawn from a log-normal diameter distribution.

- `d_g`: geometric mean diameter [m]
- `sigma_g`: geometric standard deviation
- `rho`: particle density [kg/m³]

Returns a vector of single-species mass vectors in kg.
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

function bin_size_distribution(diams, bin_edges)
    error("not implemented")
end

function extract_diagnostics(sol, sys, A)
    error("not implemented")
end

"""
    standard_aerosol_atmosphere() -> AtmosphericParameters

Standard atmospheric conditions for near-surface aerosol simulations.
- T = 293.15 K
- P = 101325 Pa
- rho_p = 1800 kg/m³ (sulfate-like aerosol)
"""
function standard_aerosol_atmosphere()
    return AtmosphericParameters(
        293.15,      # T [K]
        101325.0,    # P [Pa]
        1.225,       # rho_f [kg/m³]
        1.81e-5,     # mu_f [Pa·s]
        1.48e-5,     # nu [m²/s]
        1800.0,      # rho_p [kg/m³]
        9.81         # g [m/s²]
    )
end

"""
    standard_cloud_atmosphere() -> AtmosphericParameters

Standard atmospheric conditions for cumulus cloud simulations (~2 km altitude).
- T = 288.15 K
- P = 80000 Pa
- rho_p = 1000 kg/m³ (liquid water)
"""
function standard_cloud_atmosphere()
    return AtmosphericParameters(
        288.15,      # T [K]
        80000.0,     # P [Pa]
        1.06,        # rho_f [kg/m³] (lower pressure)
        1.75e-5,     # mu_f [Pa·s]
        1.65e-5,     # nu [m²/s]
        1000.0,      # rho_p [kg/m³]
        9.81         # g [m/s²]
    )
end

end
