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
        if n < p.n_sim && n > 0
            zero_μ = zero(get_particle(u, 1, A))
            for i in (n + 1):(p.n_sim)
                set_particle!(du, i, A, zero_μ)
            end
        end
        nothing
    end
end

# ---- Per-process jump collection ----

_process_jumps(::PhysicsProcess, sys) = ()
function _process_jumps(proc::CoagulationProcess, sys)
    (make_coagulation_jump(proc.kernel, proc.sampling),)
end
function _process_jumps(proc::NonCNMCCoagulationProcess, sys)
    (make_non_cnmc_coagulation_jump(proc.kernel, proc.sampling),)
end
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
- `n_sim` — particle-slot capacity; for CNMC this is also the target count.
  Default: length(particles)

# Returns
- `JumpProblem` ready for `solve(prob, Tsit5())`
"""
function ParticleProblem(
        particles::Vector{SVector{A, Float64}},
        volume::Float64,
        gas_phase_fn,
        processes::Tuple{Vararg{PhysicsProcess}};
        tspan = (0.0, 3600.0),
        n_sim = length(particles)
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

# ---- Parcel state extraction (NEW) ----

"""
    extract_parcel(u::Vector{Float64}, n_sim, A) -> ParcelState

Extract parcel variables from the end of the ODE state vector.
"""
function extract_parcel(u::Vector{Float64}, n_sim::Int, A::Int)
    offset = n_sim * A
    return ParcelState(
        u[offset + 1],  # T
        u[offset + 2],  # p
        u[offset + 3],  # qv
        u[offset + 4]  # S
    )
end

"""
    set_parcel!(u::Vector{Float64}, parcel::ParcelState, n_sim, A)

Write parcel variables into the end of the ODE state vector.
"""
function set_parcel!(
        u::Vector{Float64}, parcel::ParcelState, n_sim::Int, A::Int)
    offset = n_sim * A
    u[offset + 1] = parcel.T
    u[offset + 2] = parcel.p
    u[offset + 3] = parcel.qv
    u[offset + 4] = parcel.S
    nothing
end

"""
    solve_split(particles, volume, gas_phase_fn, processes, solver;
                tspan, n_sim, dt_split, saveat, record_func) -> (sol, records)

Lie-Trotter operator-splitting driver: per sub-step of length `dt_split`,
solve the drift-only ODE on the interval, then advance the frozen-state
coagulation SSA by the same interval (`step_coagulation!`). Requires
`saveat` to be an integer multiple of `dt_split`; supports drift processes
plus at most one coagulation process (`CoagulationProcess` /
`NonCNMCCoagulationProcess`); Emission/Dilution processes are rejected.
Records are taken at `tspan[1]` and at every `tspan[1] + k*saveat` boundary,
after that sub-step's jump phase. Returns the final ODE solution and the
records vector built from `record_func(t, u, sys)`.
"""
function solve_split(particles::Vector{SVector{A, Float64}},
        volume::Float64, gas_phase_fn, processes::Tuple{Vararg{PhysicsProcess}},
        solver;
        tspan = (0.0, 3600.0), n_sim = length(particles),
        dt_split::Real, saveat::Real, record_func) where {A}
    dt_split > 0.0 || throw(ArgumentError("dt_split must be positive, got $dt_split"))
    saveat > 0.0 || throw(ArgumentError("saveat must be positive, got $saveat"))
    isapprox(rem(saveat, dt_split), 0.0; atol = 1.0e-9 * dt_split) ||
        throw(ArgumentError("saveat ($saveat) must be an integer multiple of dt_split ($dt_split)"))
    drift_processes = Tuple(p for p in processes if provides_drift(p))
    coag = nothing
    for p in processes
        if p isa CoagulationProcess || p isa NonCNMCCoagulationProcess
            coag === nothing ||
                throw(ArgumentError("solve_split supports at most one coagulation process"))
            coag = p
        elseif !(provides_drift(p))
            throw(ArgumentError("solve_split does not support process $(typeof(p))"))
        end
    end
    sys = ParticleSystem(Val(A), n_sim, volume, gas_phase_fn)
    u = make_u0(particles)
    if length(u) < n_sim * A
        u_extended = zeros(Float64, n_sim * A)
        u_extended[1:length(u)] .= u
        u = u_extended
    end
    ode_func! = make_ode_func(drift_processes)
    records = Any[record_func(tspan[1], u, sys)]
    t0, t_end = tspan
    n_steps = ceil(Int, (t_end - t0) / dt_split - 1.0e-12)
    sol = nothing
    t_prev = t0
    for k in 1:n_steps
        t_next = min(t0 + k * dt_split, t_end)
        oprob = ODEProblem(ode_func!, u, (t_prev, t_next), sys)
        sol = solve(oprob, solver)
        u = copy(sol.u[end])
        if coag !== nothing
            step_coagulation!(u, sys, coag, t_next - t_prev)
        end
        if isapprox(rem(t_next - t0, saveat), 0.0; atol = 1.0e-9 * saveat) ||
           t_next >= t_end - 1.0e-12
            push!(records, record_func(t_next, u, sys))
        end
        t_prev = t_next
    end
    return sol, records
end
