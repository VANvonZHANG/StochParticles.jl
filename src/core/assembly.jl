# src/core/assembly.jl

using SparseArrays: sparse, SparseMatrixCSC

"""
    apply_all_drifts!(dμ, μ, g, t, processes::Tuple)

Apply drift from all drift-providing processes given the precomputed
environment `g`. Type-stable because tuple element types are known at
compile time (concrete NTuple).
"""
function apply_all_drifts!(dμ, μ, g, t, processes::Tuple)
    for proc in processes
        if provides_drift(proc)
            dμ = dμ + apply_drift(proc, μ, g, t)
        end
    end
    return dμ
end

"""
    make_ode_func(processes::Tuple)

RHS for SciML's ODEProblem. If the tuple contains a `ParcelProcess`, the
state vector carries a 3-slot parcel tail `[T, p, qv]`: the environment
`g` is diagnosed from the tail via `parcel_env` ONCE per evaluation and
`parcel_drift` is fed the tail scalars directly after the particle loop —
no intermediate state, so the whole RHS is transparent to automatic
differentiation (implicit solvers with ForwardDiff Jacobians work).
Σ dm_w is accumulated at `h2o_idx`. Without a parcel process,
`g = sys.gas_phase(t)` is evaluated once per evaluation.
"""
function make_ode_func(processes::Tuple)
    idx = findfirst(p -> p isa ParcelProcess, processes)
    parcel_proc = idx === nothing ? nothing : processes[idx]
    return function (du, u, p, t)
        A = species_val(p)
        n = p.n_active
        # parcel tail placeholders (overwritten in the parcel branch — keeps
        # the locals type-stable in both the Float64 and the Dual world)
        T_p = p_p = qv_p = zero(eltype(u))
        total_cond = zero(eltype(u))
        if parcel_proc !== nothing
            T_p, p_p, qv_p = u[end - 2], u[end - 1], u[end]   # last 3 slots
            g = parcel_env(T_p, p_p, qv_p)
        else
            g = p.gas_phase(t)
        end
        for i in 1:n
            μ = get_particle(u, i, A)
            dμ = apply_all_drifts!(zero(μ), μ, g, t, processes)
            set_particle!(du, i, A, dμ)
            if parcel_proc !== nothing
                total_cond += dμ[parcel_proc.h2o_idx]
            end
        end
        # Zero out inactive particle slots (A-length writes; tail untouched)
        if n < p.n_sim && n > 0
            zero_μ = zero(get_particle(u, 1, A))
            for i in (n + 1):(p.n_sim)
                set_particle!(du, i, A, zero_μ)
            end
        end
        if parcel_proc !== nothing
            drift3 = parcel_drift(T_p, p_p, total_cond, parcel_proc.w,
                parcel_proc.m_air; thermo = parcel_proc.thermo)
            set_parcel_drift!(du, drift3, p.n_sim, A)
        end
        nothing
    end
end

"""
    rhs_jac_prototype(n_sim, A) -> SparseMatrixCSC{Bool}

Sparsity prototype of the parcel-coupled RHS Jacobian ∂(du)/∂u for
`solve_split(...; jac_prototype = rhs_jac_prototype(n_sim, A))` with an
implicit solver (Rodas5/QNDF/FBDF): particle `i`'s drift depends only on
its own A state slots and the 3-slot parcel tail; the tail rows depend on
everything (Σ dm_w). Block-diagonal + arrowhead — with the prototype,
sparse-coloring AD needs ~(A+3) RHS evaluations per Jacobian instead of
`n_sim·A+3` (dense), the difference between minutes and days at n=1000.
"""
function rhs_jac_prototype(n_sim::Int, A::Int)
    N = n_sim * A + 3
    rows = Int[]
    cols = Int[]
    cap = n_sim * A * (A + 3) + 3 * N
    sizehint!(rows, cap)
    sizehint!(cols, cap)
    for i in 1:n_sim
        base = (i - 1) * A
        tail = (N - 2):N
        for r in (base + 1):(base + A)
            for c in (base + 1):(base + A)
                push!(rows, r)
                push!(cols, c)
            end
            for c in tail
                push!(rows, r)
                push!(cols, c)
            end
        end
    end
    for r in (N - 2):N, c in 1:N

        push!(rows, r)
        push!(cols, c)
    end
    # store `true` at structural positions — the sparsity PATTERN is the
    # deliverable (values are never used by the solver)
    return sparse(rows, cols, trues(length(rows)), N, N)
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

If the tuple contains a `ParcelProcess`, the state vector gains the 3-slot
parcel tail `[T, p, qv]` (jump `affect!` functions only touch particle slots).
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

    # M3: append the parcel tail when a ParcelProcess is present
    p_idx = findfirst(p -> p isa ParcelProcess, processes)
    if p_idx !== nothing
        pc = processes[p_idx]
        append!(u0, (pc.parcel[].T, pc.parcel[].p, pc.parcel[].qv))
    end

    ode_func! = make_ode_func(processes)
    jumps = collect_all_jumps(processes, sys)

    oprob = ODEProblem(ode_func!, u0, tspan, sys)
    jprob = JumpProblem(oprob, Direct(), jumps...)

    return jprob
end

"""
    solve_split(particles, volume, gas_phase_fn, processes, solver;
                tspan, n_sim, dt_split, saveat, record_func,
                abstol = nothing, reltol = nothing)
            -> (sol, records)

Lie-Trotter operator-splitting driver: per sub-step of length `dt_split`,
solve the drift-only ODE on the interval, then advance the frozen-state
coagulation SSA by the same interval (`step_coagulation!`). Requires
`saveat` to be an integer multiple of `dt_split`; supports drift processes
(including at most one `ParcelProcess`, which appends the 3-slot parcel tail
`[T, p, qv]` to the state vector) plus at most one coagulation process
(`CoagulationProcess` / `NonCNMCCoagulationProcess`); Emission/Dilution
processes are rejected. Records are taken at `tspan[1]` and at every
`tspan[1] + k*saveat` boundary, after that sub-step's jump phase. Returns
the final ODE solution and the records vector built from
`record_func(t, u, sys)`.

Optional `abstol`/`reltol` are forwarded to each internal `solve` call. Set
them explicitly for kilogram-scale particle states, where the solver defaults
(abstol = 1e-6) exceed the state magnitudes by many orders of magnitude.

`jac_prototype` (e.g. `rhs_jac_prototype(n_sim, A)`), when supplied, is
attached to the internal `ODEFunction` — REQUIRED for implicit solvers
(Rodas5/QNDF/FBDF) at realistic `n_sim`: with the sparse prototype,
coloring-aware AD costs ~(A+3) RHS evaluations per Jacobian instead of
the dense N = n_sim·A+3.
"""
function solve_split(particles::Vector{SVector{A, Float64}},
        volume::Float64, gas_phase_fn, processes::Tuple{Vararg{PhysicsProcess}},
        solver;
        tspan = (0.0, 3600.0), n_sim = length(particles),
        dt_split::Real, saveat::Real, record_func,
        abstol = nothing, reltol = nothing,
        clamp_h2o_idx::Union{Nothing, Int} = nothing,
        jac_prototype = nothing) where {A}
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
    # M3: append the parcel tail when a ParcelProcess is present
    p_idx = findfirst(p -> p isa ParcelProcess, processes)
    if p_idx !== nothing
        pc = processes[p_idx]
        append!(u, (pc.parcel[].T, pc.parcel[].p, pc.parcel[].qv))
    end
    ode_func! = make_ode_func(drift_processes)
    ode_rhs = jac_prototype === nothing ? ode_func! :
              ODEFunction(ode_func!; jac_prototype = jac_prototype)
    records = Any[record_func(tspan[1], u, sys)]
    t0, t_end = tspan
    n_steps = ceil(Int, (t_end - t0) / dt_split - 1.0e-12)
    solver_opts = Pair{Symbol, Any}[]
    abstol === nothing || push!(solver_opts, :abstol => abstol)
    reltol === nothing || push!(solver_opts, :reltol => reltol)
    sol = nothing
    t_prev = t0
    for k in 1:n_steps
        t_next = min(t0 + k * dt_split, t_end)
        oprob = ODEProblem(ode_rhs, u, (t_prev, t_next), sys)
        sol = solve(oprob, solver; solver_opts...)
        u = copy(sol.u[end])
        # mirror the accepted parcel state into the driver-facing box
        # (the RHS itself reads the tail directly — no in-solver box access)
        if p_idx !== nothing
            processes[p_idx].parcel[] = extract_parcel(u, sys.n_sim, Val(A))
        end
        # clamp evaporation overshoots at the ACCEPTED sub-step boundary:
        # any mode with fast condensation/evaporation dynamics can integrate
        # a particle slightly past zero water (magnitude ~1e-20 kg); the RHS
        # trial-guard cannot see accepted states, and diagnostics would
        # DomainError on them. Index comes from the ParcelProcess when
        # present, else the explicit clamp_h2o_idx argument (open-loop runs)
        hi = clamp_h2o_idx !== nothing ? clamp_h2o_idx :
             (p_idx !== nothing ? processes[p_idx].h2o_idx : nothing)
        if hi !== nothing
            for i in 1:(sys.n_active)
                μ = get_particle(u, i, Val(A))
                if μ[hi] < 0.0
                    set_particle!(u, i, Val(A), setindex(μ, 0.0, hi))
                end
            end
        end
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
