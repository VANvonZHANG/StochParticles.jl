# src/processes/condensation.jl

"""
    CondensationProcess{I} <: PhysicsProcess

ODE drift process: dμ/dt = flux(μ, g(t), t).
Provides drift only (no jumps).

# Fields
- `flux::I` — function (μ::SVector, g, t) -> SVector, returning condensation flux
"""
struct CondensationProcess{I <: Function} <: PhysicsProcess
    flux::I
end

provides_drift(::CondensationProcess) = true

"""
    apply_drift(proc::CondensationProcess, μ, sys, t) -> SVector

Compute the condensation drift for particle state μ.
"""
function apply_drift(proc::CondensationProcess, μ::SVector{A, Float64}, sys::ParticleSystem{A}, t) where {A}
    g = sys.gas_phase(t)
    return proc.flux(μ, g, t)
end

"""
    SpeciesDependentCondensation{A} <: PhysicsProcess

Constant per-species condensation rates.

# Constructor
    SpeciesDependentCondensation(rates::SVector{A, Float64})

Each species grows at its own constant rate [kg/s], independent of particle state.
For state-dependent or gas-phase-dependent rates, use `CondensationProcess` directly.
"""
struct SpeciesDependentCondensation{A} <: PhysicsProcess
    rates::SVector{A, Float64}
end

provides_drift(::SpeciesDependentCondensation) = true

function apply_drift(
        proc::SpeciesDependentCondensation{A}, μ::SVector{A, Float64},
        sys::ParticleSystem{A}, t) where {A}
    return proc.rates
end

# ---- H2O Condensation Flux (new) ----

"""
    H2OCondensationFlux

Physically rigorous H2O condensation flux implementing κ-Köhler theory.

# Constructor
    H2OCondensationFlux(thermo, h2o_idx, densities, w)

# Fields
- `thermo::ThermodynamicsParams` — thermodynamic parameters
- `h2o_idx::Int` — index of H2O in the species vector
- `densities::SVector{A,Float64}` — per-species densities
- `w::Float64` — updraft velocity [m/s] (used when parcel model is active)
"""
struct H2OCondensationFlux{A}
    thermo::ThermodynamicsParams{A}
    h2o_idx::Int
    densities::SVector{A, Float64}
    w::Float64
end

"""
    (flux::H2OCondensationFlux)(μ, env, sys, t) -> SVector{A,Float64}

Compute H2O condensation flux for particle with composition μ.

# Arguments
- `μ::SVector{A}` — particle masses [kg]
- `env::SVector{2}` — environment [T, p_v] (temperature [K], vapor pressure [Pa])
- `sys` — ParticleSystem (unused in basic flux, reserved for future parcel coupling)
- `t` — time [s]

# Returns
- `dμ/dt::SVector{A}` — mass change rates [kg/s], only H2O species is non-zero
"""
function (flux::H2OCondensationFlux{A})(
        μ::SVector{A, Float64},
        env::SVector{2, Float64},
        sys,
        t
) where {A}
    h2o_idx = flux.h2o_idx
    thermo = flux.thermo
    densities = flux.densities

    T = env[1]
    p_v = env[2]

    # Extract dry masses and water mass
    m_dry = zero(SVector{A, Float64})
    for k in 1:(A - 1)
        if k != h2o_idx
            m_dry = setindex(m_dry, μ[k], k)
        end
    end
    m_w = μ[h2o_idx]

    # Equilibrium vapor pressure over droplet
    p_eq = equilibrium_vapor_pressure(m_dry, m_w, thermo, densities, T)

    # Modified diffusion coefficient
    p_sat = saturation_vapor_pressure(T)
    D_v_prime = modified_diffusion_coefficient(thermo, T, p_sat)

    # Wet particle radius
    R = particle_wet_radius(m_dry, m_w, densities)

    # Condensation rate (moles/s)
    # dm_w/dt = 4πR · D_v' · (p_v - p_eq) / (R_v · T)
    dNw_dt = 4.0 * π * R * D_v_prime * (p_v - p_eq) / (thermo.R_v * T)

    # Convert to mass rate
    dm_w_dt = dNw_dt * thermo.M_w

    # Build dμ/dt: only H2O changes
    dμ = zero(SVector{A, Float64})
    dμ = setindex(dμ, dm_w_dt, h2o_idx)

    return dμ
end

"""
    H2OCondensationProcess(thermo, densities; h2o_idx, w)

Convenience constructor for a `CondensationProcess` with physically rigorous H2O flux.

# Arguments
- `thermo::ThermodynamicsParams` — thermodynamic parameters
- `densities::SVector{A,Float64}` — per-species densities
- `h2o_idx::Int` — index of H2O in species vector (default: last species)
- `w::Float64` — updraft velocity [m/s] (default: 1.0)

# Returns
- `CondensationProcess` with `H2OCondensationFlux`

# Example
```julia
thermo = ThermodynamicsParams(κ_values, 0.072, 1000.0, ...)
densities = SVector(1770.0, 1000.0)  # SO4, H2O
proc = H2OCondensationProcess(thermo, densities; h2o_idx=2, w=1.0)
```
"""
function H2OCondensationProcess(
        thermo::ThermodynamicsParams{A},
        densities::SVector{A, Float64};
        h2o_idx::Int = A,
        w::Float64 = 1.0
) where {A}
    flux = H2OCondensationFlux(thermo, h2o_idx, densities, w)
    return CondensationProcess((μ, g, t) -> flux(μ, g, nothing, t))
end
