# src/StochParticles.jl
module StochParticles

using StaticArrays
using JumpProcesses
using OrdinaryDiffEq
using Random

# ---- Trait layer (abstract types) ----
include("processes/process_trait.jl")

# ---- Interface definitions (error-stub fallbacks) ----
include("interface.jl")

# ---- Public API exports ----
export ParticleSystem, ParticleProblem, species_val
export get_particle, set_particle!, make_u0, total_mass
export PhysicsProcess, provides_drift
export CoagulationKernel, CoagulationSampling
export BrownianKernel, GlobalMajorant, CompositeKernel
export CoagulationProcess, CondensationProcess, EmissionProcess, DilutionProcess
export apply_drift, make_ode_func
export compute_majorant, majorant_rate
export cnmc_merge!, cnmc_clone!, cnmc_volume_rescale!, cnmc_coagulate!
export make_coagulation_jump, make_emission_jump
export dilution_death_affect!, dilution_birth_affect!, make_dilution_jumps
export number_concentration, mass_concentration, species_mass_concentration

# ---- Core implementations ----
include("core/particle_system.jl")
include("core/cnmc.jl")

# ---- Process implementations ----
include("processes/coagulation.jl")
include("processes/condensation.jl")
include("processes/emission.jl")
include("processes/dilution.jl")

# ---- Assembly (depends on all above) ----
include("core/assembly.jl")

# ---- Diagnostics ----
include("diagnostics/moments.jl")

end
