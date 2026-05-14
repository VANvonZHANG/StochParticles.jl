# src/StochParticles.jl
module StochParticles

using StaticArrays
using JumpProcesses
using OrdinaryDiffEq
using Random
using SpecialFunctions

# ---- Trait layer (abstract types) ----
include("processes/process_trait.jl")

# ---- Interface definitions (error-stub fallbacks) ----
include("interface.jl")

# ---- Public API exports ----
export ParticleSystem, ParticleProblem, species_val
export get_particle, set_particle!, make_u0, total_mass
export PhysicsProcess, provides_drift
export CoagulationKernel, CoagulationSampling
export BrownianKernel, GlobalMajorant, CompositeKernel, GravitationalKernel,
       AyalaTurbulentKernel, AtmosphericParameters, make_kernel
export CoagulationProcess, CondensationProcess, EmissionProcess, DilutionProcess
export apply_drift, make_ode_func
export compute_majorant, majorant_rate
export cnmc_merge!, cnmc_clone!, cnmc_volume_rescale!, cnmc_coagulate!
export make_coagulation_jump, make_emission_jump
export dilution_death_affect!, dilution_birth_affect!, make_dilution_jumps
export number_concentration, mass_concentration, species_mass_concentration
export reconstruct_volumes, extract_concentrations
export check_mass_conservation
export particle_diameters, compute_size_distribution
export bin_size_distribution
export standard_aerosol_atmosphere, standard_cloud_atmosphere
export lognormal_masses, diameters_from_masses
export plot_concentration_evolution, plot_size_distribution_heatmap,
       plot_kernel_contributions, plot_simulation_summary

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
include("diagnostics/reconstruction.jl")
include("diagnostics/distributions.jl")
include("diagnostics/validation.jl")

# ---- Plotting ----
include("plotting/core.jl")
include("plotting/recipes.jl")

# ---- Utils ----
include("utils/binning.jl")
include("utils/parameters.jl")
include("utils/initial_conditions.jl")

# ---- I/O ----
include("io/hdf5.jl")

end
