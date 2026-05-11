# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-11

### Added

#### Diagnostics module
- `reconstruct_volumes(sol, prob)`: Reconstruct volume history using mass conservation
- `extract_concentrations(sol, prob)`: Extract number and mass concentration over time
- `particle_diameters(u, sys, rho)`: Compute sphere-equivalent diameters from particle masses
- `compute_size_distribution(sol, prob, bin_edges, rho; n_snapshots)`: Compute dN/dlogD size distribution matrix for heatmap visualization
- `check_mass_conservation(sol, prob; tolerance)`: Validate mass concentration conservation, returns `(passed, rel_error)`

#### Plotting module
- `plot_concentration_evolution(t, N_conc, M_conc; time_unit)`: Low-level concentration evolution plot
- `plot_size_distribution_heatmap(snapshot_times, bin_centers, dNdlogD_matrix; time_unit, diameter_unit)`: Low-level size distribution heatmap
- `plot_kernel_contributions(labels, values)`: Bar chart for kernel contribution comparison
- `plot_simulation_summary(sol, prob, bin_edges, rho; time_unit, diameter_unit, n_snapshots, layout, size)`: High-level convenience function combining concentration evolution and size distribution heatmap

#### Utils module
- `bin_size_distribution(diams, bin_edges)`: Histogram particle diameters into bins
- `standard_aerosol_atmosphere()`: Standard atmospheric parameters for near-surface aerosol simulations
- `standard_cloud_atmosphere()`: Standard atmospheric parameters for cumulus cloud simulations
- `lognormal_masses(N, d_g, sigma_g, rho)`: Generate particle masses from log-normal diameter distribution
- `diameters_from_masses(masses, rho)`: Convert mass vectors back to diameters

#### Coagulation kernels
- `GravitationalKernel`: Gravitational settling coagulation kernel
- `AyalaTurbulentKernel`: Turbulent coagulation kernel (Ayala et al.)
- `make_kernel(params, epsilon, R_lambda, densities)`: Convenience constructor for composite kernel (Brownian + gravitational + turbulent)

#### Examples
- Aerosol Brownian coagulation example with bimodal initial distribution and heatmap visualization
- Cloud droplet turbulent coagulation example with composite kernel and kernel contribution comparison

### Changed
- Examples simplified from ~250 lines of inline code to ~100 lines using new public API
- Removed `examples/utils.jl` (all functions migrated to package)

## [0.1.0] - 2026-04-18

### Added
- `ParticleSystem` mutable struct: PDMP parameter container with compile-time species count
- `ParticleProblem` constructor: assembles SciML `JumpProblem` from particle states and physics processes
- Particle access utilities: `get_particle`, `set_particle!`, `make_u0`, `total_mass`
- `CondensationProcess`: ODE drift process for condensational growth
- `CoagulationProcess`: stochastic jump process with Majorant/Null-event sampling
- Coagulation kernels: `BrownianKernel`, `CompositeKernel`
- `EmissionProcess`: Poisson point process for particle injection
- `DilutionProcess`: death/birth jumps for entrainment and dilution
- CNMC (Constant Number Monte Carlo): merge, clone, volume rescale, and full coagulate step
- Diagnostic moments: `number_concentration`, `mass_concentration`, `species_mass_concentration`
- `PhysicsProcess` trait system with `provides_drift` and `apply_drift`
- 49 tests across 9 test files

[0.1.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.1.0
