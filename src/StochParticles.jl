# src/StochParticles.jl
module StochParticles

using StaticArrays
using JumpProcesses
using OrdinaryDiffEq
using Random
using SpecialFunctions
using StatsBase

# ---- Trait layer (abstract types) ----
include("processes/process_trait.jl")

# ---- Interface definitions (error-stub fallbacks) ----
include("interface.jl")

# ---- Public API exports ----
export ParticleSystem, ParticleProblem, species_val
export get_particle, set_particle!, make_u0, total_mass
export PhysicsProcess, provides_drift
export CoagulationKernel, CoagulationSampling
export BrownianKernel, GlobalMajorant, LocalMajorant, CompositeKernel, GravitationalKernel,
       AyalaTurbulentKernel, AtmosphericParameters, make_kernel
export CoagulationProcess, NonCNMCCoagulationProcess,
       CondensationProcess, SpeciesDependentCondensation,
       H2OCondensationFlux,
       H2OCondensationProcess, pre_equilibrate!, EmissionProcess, DilutionProcess
export apply_drift, make_ode_func
export compute_majorant, majorant_rate
export cnmc_merge!, cnmc_clone!, cnmc_volume_rescale!, cnmc_coagulate!
export make_coagulation_jump, make_non_cnmc_coagulation_jump, make_emission_jump
export dilution_death_affect!, dilution_birth_affect!, make_dilution_jumps
export water_activity, ThermodynamicsParams, saturation_vapor_pressure,
       modified_diffusion_coefficient, particle_wet_radius, equilibrium_vapor_pressure,
       critical_supersaturation, equilibrium_water_mass
export ParcelState, parcel_drift, extract_parcel, set_parcel!
export number_concentration, mass_concentration, species_mass_concentration,
       species_fractions,
       mixing_state_index, particle_mixing_entropy, shannon_entropy
export reconstruct_volumes, extract_concentrations
export activation_fraction, cloud_droplet_concentration
export check_mass_conservation
export particle_diameters, compute_size_distribution
export kde_log_diameter, smooth_histogram_diameter
export bin_size_distribution
export standard_aerosol_atmosphere, standard_cloud_atmosphere
export Species, species_vectors
export AS, AN, BC, OA, H2O
export lognormal_masses, diameters_from_masses
export plot_concentration_evolution, plot_size_distribution_heatmap,
       plot_kernel_contributions, plot_simulation_summary
export save_checkpoint, load_checkpoint, list_checkpoints, restore_rng
export save_checkpoint_jld2, load_checkpoint_jld2
export init_diagnostics_file, save_diagnostics, export_diagnostics_to_csv

# ---- Core implementations ----
include("core/particle_system.jl")
include("core/cnmc.jl")

# ---- Process implementations ----
include("processes/coagulation.jl")
include("processes/h2o_thermodynamics.jl")
include("processes/condensation.jl")
include("processes/parcel_model.jl")
include("processes/emission.jl")
include("processes/dilution.jl")

# ---- Assembly (depends on all above) ----
include("core/assembly.jl")

# ---- Diagnostics ----
include("diagnostics/moments.jl")
include("diagnostics/mixing_state.jl")
include("diagnostics/reconstruction.jl")
include("diagnostics/distributions.jl")
include("diagnostics/kde.jl")
include("diagnostics/smooth_histogram.jl")
include("diagnostics/validation.jl")
include("diagnostics/activation.jl")

# ---- Plotting ----
include("plotting/core.jl")
include("plotting/recipes.jl")

# ---- Utils ----
include("utils/binning.jl")
include("utils/parameters.jl")
include("utils/initial_conditions.jl")
include("utils/preset_species.jl")

# ---- I/O ----
include("io/hdf5.jl")
include("io/checkpoint.jl")
include("io/diagnostics.jl")
include("io/jld2.jl")

end
