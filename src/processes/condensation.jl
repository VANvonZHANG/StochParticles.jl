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
