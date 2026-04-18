# src/processes/dilution.jl

"""
    DilutionProcess{L<:Function, B<:Function} <: PhysicsProcess

Dilution via independent death (particle loss) and birth (background entrainment).

# Constructor
    DilutionProcess(dilution_rate_fn, background_sampler_fn)
- `dilution_rate_fn::L` — λ_dil(t) -> Float64
- `background_sampler_fn::B` — sampler(t) -> SVector{A, Float64}, returns a background particle

Contributes two ConstantRateJumps: death and birth.
"""
struct DilutionProcess{L<:Function, B<:Function} <: PhysicsProcess
    dilution_rate::L
    background_sampler::B
end

provides_drift(::DilutionProcess) = false

"""
    dilution_death_affect!(integrator, proc)

Remove one random active particle via swap-delete.
Rate: λ_dil(t) × n_active
"""
function dilution_death_affect!(integrator, proc::DilutionProcess)
    p = integrator.p
    if p.n_active <= 1
        return nothing
    end
    A_val = species_val(p)
    # Swap-delete: replace random particle with last active
    target = rand(1:p.n_active)
    if target < p.n_active
        μ_last = get_particle(integrator.u, p.n_active, A_val)
        set_particle!(integrator.u, target, A_val, μ_last)
    end
    p.n_active -= 1
    nothing
end

"""
    dilution_birth_affect!(integrator, proc)

Add a background particle sampled from the background distribution.
Rate: λ_dil(t) × V(t) × ∫ n_back(μ) dμ  (simplified to user-provided rate)
"""
function dilution_birth_affect!(integrator, proc::DilutionProcess)
    p = integrator.p
    if p.n_active >= p.n_sim
        return nothing
    end
    p.n_active += 1
    μ_new = proc.background_sampler(integrator.t)
    set_particle!(integrator.u, p.n_active, species_val(p), μ_new)
    nothing
end

"""
    make_dilution_jumps(proc::DilutionProcess) -> (ConstantRateJump, ConstantRateJump)

Create death and birth jumps for dilution.
"""
function make_dilution_jumps(proc::DilutionProcess)
    death_rate = (u, p, t) -> begin
        N = p.n_active
        N <= 1 ? 0.0 : proc.dilution_rate(t) * N
    end
    death_jump = ConstantRateJump(death_rate,
        (integrator) -> dilution_death_affect!(integrator, proc))

    birth_rate = (u, p, t) -> begin
        p.n_active >= p.n_sim ? 0.0 : proc.dilution_rate(t) * p.volume
    end
    birth_jump = ConstantRateJump(birth_rate,
        (integrator) -> dilution_birth_affect!(integrator, proc))

    return (death_jump, birth_jump)
end
