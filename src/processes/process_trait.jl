# src/processes/process_trait.jl

"""
    abstract type PhysicsProcess

Base trait for all physics processes (coagulation, condensation, etc.).
Each concrete subtype declares what it contributes to the PDMP via
`provides_drift` and `provides_jumps`.
"""
abstract type PhysicsProcess end

"""
    provides_drift(proc::PhysicsProcess) -> Bool

Does this process contribute an ODE drift term? Default: false.
"""
provides_drift(::PhysicsProcess) = false

# ---- Coagulation traits ----

"""
    abstract type CoagulationKernel

Supertype for coagulation rate kernels. A kernel computes the coagulation rate
between two particles: `K(μ_i, μ_j) -> Float64` [m³/s].

Subtypes must implement the callable interface `(kernel)(μ_i, μ_j)`.
"""
abstract type CoagulationKernel end

"""
    abstract type CoagulationSampling

Supertype for coagulation pair-selection sampling strategies.
Different strategies trade off between per-event cost and acceptance rate.

Subtypes must implement `compute_majorant(sampling, kernel, u, sys)`.
"""
abstract type CoagulationSampling end