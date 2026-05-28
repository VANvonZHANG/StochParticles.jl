# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-28

### Added

#### QSSA (Quasi-Steady State Approximation) for H2O condensation
- `equilibrium_water_mass(m_dry, thermo, densities, T, p_v)` — binary search for Köhler equilibrium water mass.
- `pre_equilibrate!(particles, thermo, densities, T, p_v; h2o_idx)` — in-place initialization of non-activated particles to Köhler equilibrium before ODE solve.
- `H2OCondensationFlux` now applies QSSA flux freezing: non-activated particles receive zero condensation flux during integration, preventing unphysical negative water masses.

#### Thermodynamics API documentation
- New `docs/src/api/thermodynamics.md` reference page documenting all exported thermodynamics functions: `ThermodynamicsParams`, `saturation_vapor_pressure`, `equilibrium_vapor_pressure`, `modified_diffusion_coefficient`, `water_activity`, `particle_wet_radius`, `critical_supersaturation`, `equilibrium_water_mass`.

#### Documentation updates
- `api/processes.md` now documents `pre_equilibrate!` and `H2OCondensationProcess`.
- `index.md` features list includes QSSA pre-equilibration.
- `tutorial.md` includes a complete QSSA condensation simulation walkthrough.

#### Testing
- Integration test verifying no-negative-mass invariant for 200 particles over 60-second QSSA condensation simulation.
- Unit tests for `equilibrium_water_mass` (equilibrium verification, size ordering).
- Unit tests for `pre_equilibrate!` (in-place modification, activated vs. non-activated behavior).
- Unit tests for QSSA flux freezing (zero flux at equilibrium, positive flux for activated particles).

### Fixed
- `[m/s]` docstring brackets in `condensation.jl` were incorrectly parsed as Markdown links by Documenter.
- `H2OCondensationFlux` and `pre_equilibrate!` loop bounds: changed `1:(A - 1)` to `1:A` to correctly handle `h2o_idx != A` cases.
- JuliaFormatter v2.5.0 formatting applied to all files (parentheses around range endpoints).

## [0.2.0] - 2026-05-16

### Added

#### HDF5/JLD2 I/O subsystem
- `save_checkpoint` / `load_checkpoint` — HDF5 checkpoint with schema v1.0.0.
- `save_checkpoint_jld2` / `load_checkpoint_jld2` — JLD2 fallback for Julia-native workflows.
- `init_diagnostics_file` / `save_diagnostics` — chunked HDF5 append for time-series data.
- `export_diagnostics_to_csv` — convert HDF5 diagnostics to CSV.
- `restore_rng` / `list_checkpoints` — RNG restoration and checkpoint enumeration.
- Diagnostics datasets: `time`, `number_concentration`, `mass_concentration`, `species_mass_concentration`, `mean_diameter`, `volume`, `size_distribution`.
- HDF5 output for mixing state diagnostics (`species_fractions`, `mixing_state_index`, `particle_entropy`).
- Dependencies: `HDF5.jl` and `JLD2.jl`.

#### Multi-species mixing state
- `species_fractions(u, Val(A), sys)` — per-particle species mass fraction matrix.
- `mixing_state_index(u, Val(A), sys)` — diversity-based mixing state index (0 = fully internal, 1 = fully external).
- `particle_entropy(u, Val(A), sys)` — Shannon entropy of particle composition distribution.
- Multi-species density support in `particle_diameters` and `diameters_from_masses`.
- Multi-species initial conditions in `lognormal_masses` via `fractions` keyword.
- `SpeciesDependentCondensation` — per-species condensation growth rates with linear mixing parameterization.
- External-to-internal mixing state coagulation example (`mixing_state_coagulation.jl`).

#### Post-processing and examples
- Python post-processing scripts in `examples/`:
  - `analyze_aerosol_brownian_coagulation.py` — single combined figure from HDF5 diagnostics.
  - `analyze_cloud_droplet_turbulent_coagulation.py` — single combined figure from HDF5 diagnostics.
- Example outputs (`.h5`, `.png`) are now written to the `examples/` folder via `@__DIR__`.

### Fixed

- Edge cases in `mixing_state_index` for single-species and uniform-composition particles.
- Validate that `lognormal_masses` fractions sum to 1.
- HDF5 2D dataset dimension ordering for cross-language compatibility with h5py.
- Missing imports and mass conservation check in mixing state coagulation example.

## [0.1.0] - 2026-05-12

### Added

#### Core simulation framework
- `ParticleSystem` mutable struct: PDMP parameter container with compile-time species count
- `ParticleProblem` constructor: assembles SciML `JumpProblem` from particle states and physics processes
- Particle access utilities: `get_particle`, `set_particle!`, `make_u0`, `total_mass`
- `PhysicsProcess` trait system with `provides_drift` and `apply_drift`

#### Physics processes
- `CondensationProcess`: ODE drift process for condensational growth
- `CoagulationProcess`: stochastic jump process with Majorant/Null-event sampling
- `EmissionProcess`: Poisson point process for particle injection
- `DilutionProcess`: death/birth jumps for entrainment and dilution

#### Coagulation kernels
- `BrownianKernel`: full transition-regime Brownian coagulation kernel (Jacobson 2005 Eq. 15.33)
  - Computes air properties (viscosity, mean free path) from temperature and pressure
  - Cunningham slip correction for Knudsen number regime
  - Covers full particle size range from 1 nm to 100 um
- `GravitationalKernel`: Gravitational settling coagulation kernel
- `AyalaTurbulentKernel`: Turbulent coagulation kernel (Ayala et al.)
- `CompositeKernel`: combine multiple kernels multiplicatively
- `make_kernel(params, epsilon, R_lambda, densities)`: convenience constructor for composite kernel

#### CNMC (Constant Number Monte Carlo)
- Merge, clone, volume rescale, and full coagulate step
- Maintains constant particle count during stochastic coagulation

#### Diagnostics module
- `reconstruct_volumes(sol, prob)`: Reconstruct volume history using mass conservation
- `extract_concentrations(sol, prob)`: Extract number and mass concentration over time
- `particle_diameters(u, sys, rho)`: Compute sphere-equivalent diameters from particle masses
- `compute_size_distribution(sol, prob, bin_edges, rho; n_snapshots)`: Compute dN/dlogD size distribution matrix for heatmap visualization
- `check_mass_conservation(sol, prob; tolerance)`: Validate mass concentration conservation
- `number_concentration(sys)`: Zeroth moment (particles per m^3)
- `mass_concentration(u, Val(A), sys)`: First moment (kg per m^3)
- `species_mass_concentration(u, idx, Val(A), sys)`: Single-species mass concentration

#### Plotting module
- `plot_concentration_evolution(t, N_conc, M_conc; time_unit)`: Low-level concentration evolution plot
- `plot_size_distribution_heatmap(snapshot_times, bin_centers, dNdlogD_matrix; time_unit, diameter_unit)`: Low-level size distribution heatmap
- `plot_kernel_contributions(labels, values)`: Bar chart for kernel contribution comparison
- `plot_simulation_summary(sol, prob, bin_edges, rho; ...)`: High-level convenience function combining concentration evolution and size distribution heatmap

#### Utils module
- `bin_size_distribution(diams, bin_edges)`: Histogram particle diameters into bins
- `standard_aerosol_atmosphere()`: Standard atmospheric parameters for near-surface aerosol simulations
- `standard_cloud_atmosphere()`: Standard atmospheric parameters for cumulus cloud simulations
- `lognormal_masses(N, d_g, sigma_g, rho)`: Generate particle masses from log-normal diameter distribution
- `diameters_from_masses(masses, rho)`: Convert mass vectors back to diameters

#### Examples
- Aerosol Brownian coagulation example with bimodal initial distribution and heatmap visualization
- Cloud droplet turbulent coagulation example with composite kernel and kernel contribution comparison

#### Testing and quality
- 98 tests across 11 test files
- Aqua.jl code quality checks (unbound type params, undefined exports, stale deps, piracy)
- Brownian kernel transition-regime precision tests

[0.1.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.1.0
[0.2.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.2.0
[0.3.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.3.0
