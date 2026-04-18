# src/core/assembly.jl

"""
    apply_all_drifts!(dμ, μ, sys, t, processes::Tuple)

Apply drift from all drift-providing processes. Type-stable because tuple
element types are known at compile time (concrete NTuple).
"""
function apply_all_drifts!(dμ, μ, sys, t, processes::Tuple)
    for proc in processes
        if provides_drift(proc)
            dμ = dμ + apply_drift(proc, μ, sys, t)
        end
    end
    return dμ
end

"""
    make_ode_func(processes::Tuple)

Create the ODE right-hand side function for SciML's ODEProblem.
Iterates over active particles, applies drift from all drift-providing processes.
The returned closure captures the concrete process tuple type for full specialization.
"""
function make_ode_func(processes::Tuple)
    return function (du, u, p, t)
        A = species_val(p)
        n = p.n_active
        for i in 1:n
            μ = get_particle(u, i, A)
            dμ = apply_all_drifts!(zero(μ), μ, p, t, processes)
            set_particle!(du, i, A, dμ)
        end
        # Zero out inactive particle slots
        if n < p.n_sim
            zero_μ = zero(get_particle(u, 1, A))
            for i in (n + 1):p.n_sim
                set_particle!(du, i, A, zero_μ)
            end
        end
        nothing
    end
end

# ---- Per-process jump collection ----

_process_jumps(::PhysicsProcess, sys) = ()
_process_jumps(proc::CoagulationProcess, sys) = (make_coagulation_jump(proc.kernel, proc.sampling),)
_process_jumps(proc::EmissionProcess, sys) = (make_emission_jump(proc),)
_process_jumps(proc::DilutionProcess, sys) = make_dilution_jumps(proc)

"""
    collect_all_jumps(processes::Tuple, sys) -> Tuple

Collect all jumps from all processes into a tuple for JumpProblem construction.
"""
function collect_all_jumps(processes::Tuple, sys)
    jumps = []
    for proc in processes
        append!(jumps, _process_jumps(proc, sys))
    end
    return Tuple(jumps)
end

"""
    ParticleProblem(particles, volume, gas_phase_fn, processes; tspan, n_sim)

Construct a SciML JumpProblem representing the PDMP for particle simulation.

# Arguments
- `particles::Vector{SVector{A, Float64}}` — initial particle states
- `volume::Float64` — initial computational volume
- `gas_phase_fn` — external gas concentration function g(t)
- `processes::NTuple` — tuple of PhysicsProcess instances
- `tspan` — (t_start, t_end)
- `n_sim` — target particle count (CNMC). Default: length(particles)

# Returns
- `JumpProblem` ready for `solve(prob, Tsit5())`
"""
function ParticleProblem(
    particles::Vector{SVector{A, Float64}},
    volume::Float64,
    gas_phase_fn,
    processes::Tuple{Vararg{PhysicsProcess}};
    tspan = (0.0, 3600.0),
    n_sim = length(particles),
) where {A}

    sys = ParticleSystem(Val(A), n_sim, volume, gas_phase_fn)
    u0 = make_u0(particles)

    # Pad u0 to n_sim if needed
    if length(u0) < n_sim * A
        u0_extended = zeros(Float64, n_sim * A)
        u0_extended[1:length(u0)] .= u0
        u0 = u0_extended
    end

    ode_func! = make_ode_func(processes)
    jumps = collect_all_jumps(processes, sys)

    oprob = ODEProblem(ode_func!, u0, tspan, sys)
    jprob = JumpProblem(oprob, Direct(), jumps...)

    return jprob
end
