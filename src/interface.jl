# src/interface.jl
# Formal interface definitions for StochParticles.jl
# All abstract types must be defined before this file is included
# (via processes/process_trait.jl and processes/coagulation.jl).
#
# Pattern: each interface function has a fallback on the abstract supertype
# that raises a descriptive error. Concrete subtypes override with specific
# implementations.

# ---- PhysicsProcess interface ----

"""
    apply_drift(proc::PhysicsProcess, μ, sys, t) -> SVector

Compute the ODE drift contribution from `proc` for particle state `μ`
at time `t`, given the particle system `sys`.

Only called when `provides_drift(proc) == true`.

# Implementation contract
Concrete subtypes of `PhysicsProcess` that declare `provides_drift`
must implement this method.
"""
function apply_drift(proc::PhysicsProcess, μ, sys, t)
    error("$(typeof(proc)) must implement `apply_drift`")
end

# ---- CoagulationKernel interface ----

"""
    (kernel::CoagulationKernel)(μ_i, μ_j) -> Float64

Evaluate the coagulation rate between two particles with compositions
`μ_i` and `μ_j`. Returns a rate in [m³/s].

# Implementation contract
All concrete subtypes of `CoagulationKernel` must be callable as
`kernel(μ_i, μ_j)`.
"""
function (kernel::CoagulationKernel)(μ_i, μ_j)
    error("$(typeof(kernel)) must implement the callable interface `(kernel)(μ_i, μ_j)`")
end

# ---- CoagulationSampling interface ----

"""
    compute_majorant(sampling::CoagulationSampling, kernel, u, sys) -> Float64

Compute the majorant kernel value K_max ≥ K(μ_i, μ_j) for all active
particle pairs in state `u`.

Used by the Majorant/Null-event method to upper-bound the coagulation
rate and drive the acceptance/rejection step.

# Implementation contract
Concrete subtypes of `CoagulationSampling` must implement this method.
"""
function compute_majorant(sampling::CoagulationSampling, kernel, u, sys)
    error("$(typeof(sampling)) must implement `compute_majorant`")
end

"""
    majorant_rate(sampling::CoagulationSampling, kernel, u, sys) -> Float64

Compute the total coagulation event rate using the majorant method:

    Λ = (K_max / V) × N × (N − 1) / 2

where K_max = `compute_majorant(sampling, kernel, u, sys)`,
V is the computational volume, and N is the number of active particles.

Default implementation delegates to `compute_majorant`.
"""
function majorant_rate(sampling::CoagulationSampling, kernel, u, sys)
    K_max = compute_majorant(sampling, kernel, u, sys)
    N = sys.n_active
    return K_max / sys.volume * N * (N - 1) / 2
end
