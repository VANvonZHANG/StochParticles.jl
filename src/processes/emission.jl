# src/processes/emission.jl

"""
    EmissionProcess{S<:Function} <: PhysicsProcess

Particle emission via Poisson point process.

# Constructor
    EmissionProcess(total_rate, sampler)
- `total_rate::Float64` — Λ_emit = V × ∫ ṅ_emit(μ, t) dμ (user-precomputed or time-varying)
- `sampler::S` — function sampler(t) -> SVector{A, Float64}, returns a new particle state

Contributes a ConstantRateJump (V1 simplification: constant total rate).
"""
struct EmissionProcess{S<:Function} <: PhysicsProcess
    total_rate::Float64
    sampler::S
end

provides_drift(::EmissionProcess) = false

"""
    make_emission_jump(proc::EmissionProcess) -> ConstantRateJump

Create a jump that adds a new particle sampled from the emission distribution.
"""
function make_emission_jump(proc::EmissionProcess)
    rate = (u, p, t) -> begin
        p.n_active >= p.n_sim ? 0.0 : proc.total_rate
    end

    affect! = (integrator) -> begin
        p = integrator.p
        if p.n_active >= p.n_sim
            return nothing
        end
        p.n_active += 1
        μ_new = proc.sampler(integrator.t)
        set_particle!(integrator.u, p.n_active, species_val(p), μ_new)
        nothing
    end

    return ConstantRateJump(rate, affect!)
end
