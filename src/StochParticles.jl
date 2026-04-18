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
export get_particle, set_particle!
export BrownianKernel, GlobalMajorant
export CoagulationProcess, CondensationProcess, EmissionProcess, DilutionProcess

end
