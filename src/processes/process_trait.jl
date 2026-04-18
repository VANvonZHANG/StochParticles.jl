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
