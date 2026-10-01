# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.0] - 2026-09-30

Milestone release (M2 of the GCVI closure blueprint). No breaking API
changes — the minor bump marks the milestone.

### Added

- Synthetic-population initialization subsystem
  (`src/utils/synthetic_population.jl`): `TabulatedSpectrum` (tabulated
  dN/dlogD with exact inverse-CDF sampling), `lognormal_table` (analytic
  CDF-difference discretization), `SizeResolvedComposition`
  (size-resolved mean dry composition with anchor interpolation),
  `constant_fbar`, `SyntheticPopulationSpec`, `synthesize_population`
  (spectrum sampling + Dirichlet(ν·f̄(D)) composition + haze
  pre-equilibration), `nu_for_chi` (Monte-Carlo ν(χ) calibration on the
  exact generation path, cached), and `reachable_chi_max` (χ∞
  reachability bound for size-resolved f̄).
- `Distributions` dependency (Dirichlet sampling).
- Twin experiment v1 in `examples/`: calibrated χ grid
  {0.10, 0.25, 0.40, 0.60, 0.75} with off-grid truth χ = 0.50,
  per-replicate population resampling, σ_χ monitoring, KLD empty-spectrum
  guard, per-replicate pooled J with error bars, and the twin-v1 gate with
  J(χ_true) noise floor.

### Changed

- `cost_curve` now averages J over replicates *after* evaluating J (J is
  nonlinear; the previous mean-of-inputs pooling biased the cost) and
  reports mean ± std per case.

### Fixed

- M2 driver χ gate is now case-level (mean realized χ within tolerance): realized-χ sd ≈ 0.011 at n_sim = 1000 in the sparse-mixing regime, so single-replicate excursions are the resampling noise the twin experiment quantifies, not synthesis errors.
- Twin-v1 gate metric repair: pooled observation (`synthetic-obs-v2` schema, truth-case replicate mean — matching real instruments' sampling-window averaging), half-count KLD floor (one-particle equivalent instead of 1e-12), and J_size on a coarsened 31-bin closure grid; J(χ_true) noise floor σ_J 0.30 → 0.0049 — twin gate now PASS with U-shaped J and argmin inside the truth bracket.
- Pooled-observation Python path cross-checked against the exported `synthetic-obs-v2` file; closure-grid stride unified in a shared `CLOSURE_STRIDE` constant across Julia and Python.

## [0.9.1] - 2026-09-29

### Fixed

- Activation-comparison example unified to sulfate κ = 0.61 (Petters & Kreidenweis, 2007) across config, tests, and setup notes (#23).

### Changed

- CHANGELOG now carries Keep-a-Changelog version links for 0.7.0–0.9.1 (they stopped at 0.6.0); TagBot release-notes prepend step made idempotent.

## [0.9.0] - 2026-09-27
### Added

#### Virtual-GCVI closure toolkit (M0+M1 of the GCVI closure blueprint)
- `mixing_state_index(...; species)` — dry-composition χ over a species mask (water/tracers excluded); water-addition invariance under test.
- `PrescribedProfile` — open-loop environment source (linear S(t)/T(t) interpolation, flat extrapolation); `p_v` derived via `saturation_vapor_pressure` single-source invariant.
- `GCVIResponse` / `transmission` / `classify_cr_ci` — virtual GCVI ideal-classifier semantics (logistic response on endpoint wet diameter, Bernoulli CR/CI assignment).
- `virtual_smps` / `virtual_acsm` — CR/CI subpopulation aggregation to instrument dimensions, reusing `dNdlogD_from_diameters` (promoted from examples to `src/utils/binning.jl`).
- `examples/simulate_gcvi_closure.jl` — M0 walking-skeleton driver: ν scan × replicates under open-loop S(t) ramp, endpoint virtual instrument, twin-observation export (`synthetic-obs-v1` schema).
- `examples/analysis/gcvi_closure.py` + `plot_gcvi_closure.py` — KLD/MSE cost curve J(χ) and twin-recovery figure.

### Changed

- `mixing_state_index` default path (no `species` keyword): particles with zero total mass are now skipped (zero mass weight, no entropy contribution) instead of contributing uniform-fraction entropy; single-species systems still return 1.0.
- `solve_split` now forwards optional `abstol`/`reltol` to the inner ODE solve (defaults unchanged); the activation driver passes mass-scaled tolerances because the OrdinaryDiffEq default `abstol` is ~9 orders above particle masses.

## [0.8.0] - 2026-08-25
### Added

#### Activation-coagulation example redesign (`examples/`)
- Two lognormal modes allocated by number concentration (Aitken 20 nm x 800 / accumulation 200 nm x 200 at n_sim = 1000, V = 1000/1.05e12 m^3) with supersaturation S = 0.005 chosen from the measured Köhler activation window of the realized population.
- Coagulation runs `NonCNMCCoagulationProcess` + `LocalMajorant` through `solve_split` (dt_split = 10 s); per-mode number/activation records and 9-series Brownian/gravitational/turbulent kernel attribution per pair class (A-A, D-D, A-D).
- The configured concentrations drive a coalescence-sweep cascade (droplet number collapses, merged droplets grow to ~39 um and sweep the interstitial mode); full 20-replicate run ~2 h.
- Composite figure redesigned to 8 panels (per-mode N/N0, mean wet diameter, attribution triptych, size-resolved activation, dual KDE heatmaps); all three example figures unified to the Nature 89/183-mm geometry standard with PNG-only export.

#### Mixing-state impact metrics (`examples/`)
- `ccn_error_series` in `examples/analysis/mixing_state_analysis.py`: ε-type CCN mixing-state metric (Riemer et al. 2019) — relative error in CCN number under a fully-internal-mixture assumption, using a vectorized numpy port of the exact κ-Köhler `critical_supersaturation` maximization.
- `optical_error_series` in `examples/analysis/mixing_state_analysis.py`: relative error in bulk absorption/scattering at 550 nm under the same assumption, computed with PyMieScatt core-shell Mie (BC core / sulfate shell; new analysis dependency).
- Mixing-state figure: mass-conservation and composition-PDF panels replaced by ε_CCN (S = 0.1/0.3/1.0%) and ε_optical (ε_abs, ε_sca) panels.

## [0.7.0] - 2026-08-25
### Added

- `LocalMajorant` sampling with exact per-particle bounds, `step_coagulation!`
  frozen-state SSA advancer, and `solve_split` Lie-Trotter operator-splitting
  driver — makes n_sim=1000 coagulation tractable (O(N^2) bound rebuild once
  per split sub-step instead of once per virtual event).

## [0.6.0] - 2026-08-18

### Added

#### Non-CNMC coagulation
- `NonCNMCCoagulationProcess` — stochastic coagulation process without constant-number Monte Carlo resampling: accepted events merge two particles and decrement `n_active`, leaving the computational volume unchanged.
- `make_non_cnmc_coagulation_jump(kernel, sampling)` — `ConstantRateJump` for the non-CNMC path, using a majorant rate with acceptance-rejection thinning.
- Process wiring in `ParticleSystem` assembly and moment diagnostics for the variable-`n_active` regime.
- Tests for the non-CNMC coagulation path and mixing-state example configuration.

#### Example suite overhaul (`examples/`)
- `simulation_io.jl` — HDF5-based simulation record export (schema `examples-v1`), data/figure directory helpers, and replicate control via `STOCHPARTICLES_EXAMPLE_REPLICATES`.
- New simulation scripts: `simulate_single_component_coagulation.jl`, `simulate_activation_coagulation_comparison.jl`, `simulate_mixing_state_coagulation.jl`; coagulation runs via direct SSA (`direct_ssa_non_cnmc`) with dense-then-regular save schedules and configurable particle counts.
- Python analysis package `examples/analysis/` — HDF5 reader, per-study analysis modules (coagulation, activation, mixing state), KDE smoothing, shared figure style — plus top-level plot scripts producing high-DPI PNG figures, including KDE heatmaps with an aerosol-regime zoom panel.

### Changed

- `number_concentration` and zeroth-moment diagnostics now use `n_active` instead of `n_sim`: identical for CNMC processes, correct number decay under non-CNMC coagulation.
- Example HDF5 exports store time-major datasets with a compatibility group for downstream readers.

### Fixed

- Example exports now include the final simulation record (deduplicated coincident timestamps).
- Replicates use fixed initial particles so replicate spread reflects stochasticity only.

### Removed

- Legacy example scripts (`activation_in_updraft.jl`, `aerosol_brownian_coagulation.jl`, `cloud_droplet_turbulent_coagulation.jl`, `compare_distribution_methods.jl`, `mixing_state_coagulation.jl`, and the mixed PDMP demonstration) together with their ad-hoc Python analysis scripts, replaced by the new example suite.

## [0.5.0] - 2026-06-08

### Added

#### Smooth distribution plotting
- `kde_log_diameter(diams; npoints=200)` — kernel density estimation for smooth dN/dlogD curves over log-spaced diameter grids.
- `smooth_histogram_diameter(diams; nbins=100, npoints=200)` — oversampled-bin histogram with linear interpolation for smooth distribution curves.
- Two new `compute_size_distribution` methods: `:kde` and `:histogram_smooth`, adding smooth alternatives to the existing `:histogram` method.
- Method parameter forwarding through `plot_simulation_summary` for seamless smooth-plot integration.

#### Dependencies
- KernelDensity.jl — KDE-based smooth distribution estimation.
- Interpolations.jl — linear interpolation for smooth histogram curves.
- StatsBase.jl — `fit(Histogram, ...)` replacing hand-written binning logic.

#### Examples
- `compare_distribution_methods.jl` — side-by-side comparison of `:histogram`, `:kde`, and `:histogram_smooth` distribution methods.

#### Documentation
- New `docs/src/api/diagnostics.md` — reference for all diagnostics functions.
- New `docs/src/api/plotting.md` — reference for all plotting functions.

### Changed

- `bin_size_distribution` refactored to use `StatsBase.fit(Histogram, ...)` internally.
- All source files reformatted with JuliaFormatter (SciML style, indent=4, margin=92).
- Examples updated to use KDE for smooth distribution plots by default.

### Fixed

- Threshold-based spread check for KDE bandwidth to avoid oversmoothing with concentrated distributions.
- Linear interpolation in `smooth_histogram` for sparse data stability.

## [0.4.0] - 2026-05-31

### Added

#### Preset Species Library
- `Species` struct — `@kwdef` struct holding `name`, `density`, `kappa`, `molar_mass` for any aerosol species. Supports both positional and keyword constructors.
- Preset species constants with literature values:
  - `AS` — Ammonium sulfate (NH₄)₂SO₄, κ = 0.61
  - `AN` — Ammonium nitrate NH₄NO₃, κ = 0.67
  - `BC` — Black carbon (elemental carbon), κ = 0.0
  - `OA` — Organic aerosol (bulk surrogate), κ = 0.1
  - `H2O` — Water
- `species_vectors(species::Species...)` — vararg combiner returning a `NamedTuple` with fully type-stable `SVector{A}` parameter vectors (`densities`, `kappas`, `molar_masses`, `names`) plus auto-detected `h2o_idx`.
- Custom species support: users can define `Species(:CUSTOM, ...)` and mix with presets.

#### Testing
- 44 new tests for preset species: struct construction, preset values, combiner correctness, `h2o_idx` positioning, type stability (`@inferred`), error handling, and custom species mixing.

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
[0.4.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.4.0
[0.5.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.5.0
[0.6.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.6.0
[0.7.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.7.0
[0.8.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.8.0
[0.9.0]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.9.0
[0.9.1]: https://github.com/VANvonZHANG/StochParticles.jl/releases/tag/v0.9.1
[0.10.0]: https://github.com/VANvonZHANG/StochParticles.jl/compare/v0.9.1...v0.10.0
[unreleased]: https://github.com/VANvonZHANG/StochParticles.jl/compare/v0.10.0...HEAD
