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
    apply_drift(proc::CondensationProcess, μ, g, t) -> SVector

Compute the condensation drift for particle state μ given the precomputed
environment `g` (see `interface.jl` — the RHS evaluates `g` once per
evaluation and passes it through).
"""
function apply_drift(proc::CondensationProcess, μ::SVector{A, <:Real}, g, t) where {A}
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
        proc::SpeciesDependentCondensation{A}, μ::SVector{A, <:Real},
        g, t) where {A}
    return proc.rates
end

# ---- H2O Condensation Flux (new) ----

"""
    H2OCondensationFlux

Physically rigorous H2O condensation flux implementing κ-Köhler theory.
Gate-free (kinetic): every particle, haze or droplet, relaxes toward its
Köhler equilibrium via the Mason flux — the pyrcel-equivalent physics
class. Sc/D_crit are diagnostics-only (`critical_point`).

# Constructor
    H2OCondensationFlux(thermo, h2o_idx, densities, w, accom_c, accom_T)

# Fields
- `thermo::ThermodynamicsParams` — thermodynamic parameters
- `h2o_idx::Int` — index of H2O in the species vector
- `densities::SVector{A,Float64}` — per-species densities
- `w::Float64` — updraft velocity `m/s` (used when parcel model is active)
- `accom_c::Float64` — mass condensation coefficient (vapor-side Fuchs, Ghan A2)
- `accom_T::Float64` — thermal accommodation coefficient (heat-side Fuchs, Ghan A3)
"""
struct H2OCondensationFlux{A}
    thermo::ThermodynamicsParams{A}
    h2o_idx::Int
    densities::SVector{A, Float64}
    w::Float64
    accom_c::Float64   # mass condensation coefficient (vapor-side Fuchs, Ghan A2)
    accom_T::Float64   # thermal accommodation coefficient (heat-side Fuchs, Ghan A3)

    # explicit inner constructor with defaults: defining it suppresses the
    # auto-generated constructors, so an outer convenience method would not
    # collide during precompilation; validation lives at this single point
    function H2OCondensationFlux(thermo::ThermodynamicsParams{A}, h2o_idx::Int,
            densities::SVector{A, Float64}, w::Float64 = 0.0,
            accom_c::Float64 = 1.0, accom_T::Float64 = 1.0) where {A}
        0.0 < accom_c <= 1.0 ||
            throw(ArgumentError("accom_c must be in (0, 1], got $accom_c"))
        0.0 < accom_T <= 1.0 ||
            throw(ArgumentError("accom_T must be in (0, 1], got $accom_T"))
        return new{A}(thermo, h2o_idx, densities, w, accom_c, accom_T)
    end
end

"""
    (flux::H2OCondensationFlux)(μ, env, sys, t) -> SVector{A,Float64}

Compute H2O condensation flux for particle with composition μ.

# Arguments
- `μ::SVector{A}` — particle masses [kg]
- `env::SVector{3}` — environment [T, p_v, p_air] (temperature [K], vapor
  pressure [Pa], total air pressure [Pa]; the third slot feeds the air
  density of the heat-side transition-regime correction)
- `sys` — ParticleSystem (unused in basic flux, reserved for future parcel coupling)
- `t` — time [s]

# Returns
- `dμ/dt::SVector{A}` — mass change rates [kg/s], only H2O species is non-zero
"""
function (flux::H2OCondensationFlux{A})(
        μ::SVector{A, <:Real},
        env::SVector{3, <:Real},
        sys,
        t
) where {A}
    h2o_idx = flux.h2o_idx
    thermo = flux.thermo
    densities = flux.densities

    T = env[1]
    p_v = env[2]
    p_air = env[3]

    # output eltype: μ and env can carry independent Dual seeds (∂/∂state vs
    # ∂/∂env probing); every output path must use the promoted type or the
    # final setindex silently truncates derivatives (Float64(Dual) error)
    zero_out = zero(SVector{A, promote_type(eltype(μ), eltype(env))})

    # non-physical trial environment: with the corrected (55x stronger)
    # latent-heat feedback, a solver trial step can drive the coupled parcel
    # temperature below zero; a negative T makes A_kelvin (and thus the
    # critical-point search domain) negative. Zero drift lets the solver
    # reject the step.
    if T <= 0.0 || p_v < 0.0
        return zero_out
    end

    # Extract dry masses and water mass
    m_dry = zero(μ)
    for k in 1:A
        if k != h2o_idx
            m_dry = setindex(m_dry, μ[k], k)
        end
    end
    m_w = μ[h2o_idx]

    # Non-physical solver TRIAL state (negative mass): adaptive solvers
    # evaluate the RHS at rejected candidate points; return zero drift there
    # instead of throwing in the Kohler search (sqrt of a negative argument).
    # Accepted states are physical — the water-conservation invariant guards
    # this at record time.
    if m_w < 0.0 || any(k != h2o_idx && μ[k] <= 0.0 for k in 1:A)
        return zero_out
    end

    # numerical-ghost floor: Dirichlet corner draws in sparse-mixing regimes
    # can carry dry cores down to ~1e-229 kg (D_dry << 1 nm). Their Kelvin
    # term exp(A/R) overflows in p_eq and they can never hold meaningful
    # water or activate — treat sub-nanometer cores as inert in all modes
    # (the legacy gate happened to zero them anyway; :kinetic must too).
    V_dry_min = 5.2e-28                  # m^3, sphere of D = 1 nm
    V_dry = 0.0
    for k in 1:A
        if k != h2o_idx
            V_dry += μ[k] / densities[k]
        end
    end
    if V_dry < V_dry_min
        return zero_out
    end

    # Kinetic (spec §3.1, four-arm retirement 2026-10-14): no gate — every
    # particle, haze or droplet, relaxes toward its Köhler equilibrium; the
    # haze population participates in the vapor budget. Flux is continuous
    # in state (no discontinuity surface for the solver).
    # Equilibrium vapor pressure over droplet
    p_eq = equilibrium_vapor_pressure(m_dry, m_w, thermo, densities, T)

    p_sat = saturation_vapor_pressure(T)
    # Wet particle radius
    R = particle_wet_radius(m_dry, m_w, densities)

    # Effective diffusivity: Ghan (2011) split transition-regime assembly —
    # vapor branch corrected with accom_c, heat branch with accom_T (Eqs.
    # A2-A3). Supersedes the single-factor form (continuum × one Fuchs),
    # which the 2026-10-08 primer documented; at accom_c = accom_T = 1 the
    # two agree to ≲3% (ℓ_v ≈ 0.17 µm vs ℓ_t ≈ 0.19 µm at 285 K).
    D_v_prime = modified_diffusion_coefficient(thermo, T, p_sat, R, p_air;
        accom_c = flux.accom_c, accom_T = flux.accom_T)

    # Condensation mass rate [kg/s] — Mason flux with the modified
    # (latent-heat-corrected) diffusivity:
    #   dm_w/dt = 4πR · D_v' · (p_v - p_eq) / (R_v · T)
    # Δp/(R_v·T) is already a vapor MASS density difference [kg/m³] because
    # R_v is per-kg — do NOT multiply by M_w again (that bug, present since
    # inception, weakened the flux 55.5x; masked under prescribed-S open
    # loops, exposed by the closed-loop pyrcel comparison, 2026-10-07)
    dm_w_dt = 4.0 * π * R * D_v_prime * (p_v - p_eq) / (thermo.R_v * T)

    # Build dμ/dt: only H2O changes
    dμ = zero_out
    dμ = setindex(dμ, dm_w_dt, h2o_idx)

    return dμ
end

"""
    H2OCondensationProcess(thermo, densities; h2o_idx, w, accom_c, accom_T)

Convenience constructor for a `CondensationProcess` with physically rigorous H2O flux.

# Arguments
- `thermo::ThermodynamicsParams` — thermodynamic parameters
- `densities::SVector{A,Float64}` — per-species densities
- `h2o_idx::Int` — index of H2O in species vector (default: last species)
- `w::Float64` — updraft velocity `m/s` (default: 1.0)

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
        w::Float64 = 1.0,
        accom_c::Float64 = 1.0,
        accom_T::Float64 = 1.0
) where {A}
    flux = H2OCondensationFlux(thermo, h2o_idx, densities, w, accom_c, accom_T)
    return CondensationProcess((μ, g, t) -> flux(μ, g, nothing, t))
end

"""
    pre_equilibrate!(particles, thermo, densities, T, p_v; h2o_idx)

Pre-equilibrate non-activated particles to their Köhler equilibrium.

Modifies `particles` in-place: non-activated particles (S_env <= Sc) get
m_w set to equilibrium water mass. Activated particles are unchanged.

# Arguments
- `particles::Vector{SVector{A,Float64}}` — particle states (modified in-place)
- `thermo` — ThermodynamicsParams
- `densities` — per-species densities
- `T` — temperature [K]
- `p_v` — ambient vapor pressure [Pa]
- `h2o_idx` — index of H2O in species vector
"""
function pre_equilibrate!(
        particles::Vector{SVector{A, Float64}},
        thermo::ThermodynamicsParams{A},
        densities::SVector{A, Float64},
        T::Float64,
        p_v::Float64;
        h2o_idx::Int = A
) where {A}
    for i in eachindex(particles)
        μ = particles[i]
        m_dry = zero(μ)
        for k in 1:A
            if k != h2o_idx
                m_dry = setindex(m_dry, μ[k], k)
            end
        end
        Sc = critical_supersaturation(m_dry, thermo, densities, T)
        S_env = p_v / saturation_vapor_pressure(T) - 1.0
        if S_env <= Sc  # Non-activated: equilibrate
            m_eq = equilibrium_water_mass(m_dry, thermo, densities, T, p_v)
            particles[i] = setindex(μ, m_eq, h2o_idx)
        end
    end
    return particles
end
