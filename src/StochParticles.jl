# src/StochParticles.jl
module StochParticles

using StaticArrays
using JumpProcesses
using OrdinaryDiffEq
using Random

# Core
include("core/particle_system.jl")
include("core/cnmc.jl")

# Process trait system
include("processes/process_trait.jl")

# Individual processes
include("processes/coagulation.jl")
include("processes/condensation.jl")
include("processes/emission.jl")
include("processes/dilution.jl")

# Assembly (depends on all above)
include("core/assembly.jl")

# Diagnostics
include("diagnostics/moments.jl")

# Public API
export ParticleSystem, ParticleProblem, species_val
export get_particle, set_particle!, make_u0, total_mass
export PhysicsProcess, provides_drift
export BrownianKernel, GlobalMajorant, CompositeKernel
export CoagulationProcess, CondensationProcess, EmissionProcess, DilutionProcess
export apply_drift, make_ode_func
export compute_majorant, majorant_rate

end
