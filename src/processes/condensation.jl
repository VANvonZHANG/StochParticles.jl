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
    H2OCondensationFlux(thermo, h2o_idx, densities, w, activation_gate)

# Fields
- `thermo::ThermodynamicsParams` — thermodynamic parameters
- `h2o_idx::Int` — index of H2O in the species vector
- `densities::SVector{A,Float64}` — per-species densities
- `w::Float64` — updraft velocity `m/s` (used when parcel model is active)
- `activation_gate::Symbol` — `:sc_threshold` (default, legacy: zero flux
  whenever S_env ≤ Sc — branch-blind) or `:branch_aware` (past-peak droplets
  follow their branch equilibrium instead of freezing when S falls back
  below Sc; haze branch unchanged)
"""
struct H2OCondensationFlux{A}
    thermo::ThermodynamicsParams{A}
    h2o_idx::Int
    densities::SVector{A, Float64}
    w::Float64
    activation_gate::Symbol

    # explicit inner constructor with defaults: defining it suppresses the
    # auto-generated constructors, so an outer convenience method would not
    # collide during precompilation; validation lives at this single point
    function H2OCondensationFlux(thermo::ThermodynamicsParams{A}, h2o_idx::Int,
            densities::SVector{A, Float64}, w::Float64 = 0.0,
            activation_gate::Symbol = :sc_threshold) where {A}
        activation_gate in (:sc_threshold, :branch_aware) ||
            throw(ArgumentError("activation_gate must be :sc_threshold or :branch_aware, got $activation_gate"))
        return new{A}(thermo, h2o_idx, densities, w, activation_gate)
    end
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
    for k in 1:A
        if k != h2o_idx
            m_dry = setindex(m_dry, μ[k], k)
        end
    end
    m_w = μ[h2o_idx]

    # Activation gate (spec §3.1). The legacy gate is branch-blind: once a
    # droplet is past the Köhler peak (D_wet > D_crit) its branch equilibrium
    # sits BELOW Sc, so when S_env falls back below Sc the gate wrongly
    # freezes it. :branch_aware keeps the haze-branch freeze (D_wet <= D_crit)
    # but lets the p_eq-based flux take over on the droplet branch — flux is
    # continuous across D_crit because p_eq(D_crit) = p_sat·(1+Sc).
    Sc, D_crit = critical_point(m_dry, thermo, densities, T)
    S_env = p_v / saturation_vapor_pressure(T) - 1.0
    if S_env <= Sc
        if flux.activation_gate === :sc_threshold
            return zero(SVector{A, Float64})
        end
        D_wet = 2.0 * particle_wet_radius(m_dry, m_w, densities)
        if D_wet <= D_crit
            return zero(SVector{A, Float64})
        end
    end

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
    H2OCondensationProcess(thermo, densities; h2o_idx, w, activation_gate)

Convenience constructor for a `CondensationProcess` with physically rigorous H2O flux.

# Arguments
- `thermo::ThermodynamicsParams` — thermodynamic parameters
- `densities::SVector{A,Float64}` — per-species densities
- `h2o_idx::Int` — index of H2O in species vector (default: last species)
- `w::Float64` — updraft velocity `m/s` (default: 1.0)
- `activation_gate::Symbol` — `:sc_threshold` (default) or `:branch_aware`

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
        activation_gate::Symbol = :sc_threshold
) where {A}
    flux = H2OCondensationFlux(thermo, h2o_idx, densities, w, activation_gate)
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
        m_dry = zero(SVector{A, Float64})
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

"""
    reequilibrate_haze!(u, sys, thermo, densities; h2o_idx, T, S, m_air) -> ΔΣm_w

Split-step haze re-equilibration (spec §3.2, blueprint §4.3 precision
switch, default OFF). For every NON-ACTIVATED particle — haze branch under
BOTH gate modes, i.e. `S <= Sc && D_wet <= D_crit` — reset `m_w` to the
Köhler equilibrium at the supplied `(T, S)`. Particles with `S > Sc` are
skipped (they activate and grow via the ODE; their lower-branch root does
not exist at S > Sc). Returns the total particle-water change ΔΣm_w [kg]
and enforces water conservation by adjusting the parcel tail:
`qv -= ΔΣ/m_air`.
"""
function reequilibrate_haze!(u::Vector{Float64}, sys::ParticleSystem{A},
        thermo::ThermodynamicsParams{A}, densities::SVector{A, Float64};
        h2o_idx::Int, T::Float64, S::Float64, m_air::Float64) where {A}
    p_v = saturation_vapor_pressure(T) * (1.0 + S)
    delta = 0.0
    for i in 1:(sys.n_active)
        μ = get_particle(u, i, Val(A))
        m_dry = zero(SVector{A, Float64})
        for k in 1:A
            if k != h2o_idx
                m_dry = setindex(m_dry, μ[k], k)
            end
        end
        Sc, D_crit = critical_point(m_dry, thermo, densities, T)
        D_wet = 2.0 * particle_wet_radius(m_dry, μ[h2o_idx], densities)
        if S <= Sc && D_wet <= D_crit
            m_eq = equilibrium_water_mass(m_dry, thermo, densities, T, p_v)
            delta += m_eq - μ[h2o_idx]
            set_particle!(u, i, Val(A), setindex(μ, m_eq, h2o_idx))
        end
    end
    u[sys.n_sim * A + 3] -= delta / m_air    # parcel qv tail: conserve water
    return delta
end
