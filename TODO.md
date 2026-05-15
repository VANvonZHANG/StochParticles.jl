# TODO

This document tracks completed features and planned improvements for StochParticles.jl.

## Completed (as of 2026-05-11)

### Core Framework
- [x] PDMP framework with `ParticleSystem` and `ParticleProblem`
- [x] Trait-based process architecture (`PhysicsProcess`, `provides_drift`)
- [x] CNMC (Constant Number Monte Carlo) particle management
- [x] Particle access utilities (`get_particle`, `set_particle!`, `make_u0`, `total_mass`)

### Physics Processes
- [x] `CondensationProcess` — ODE drift for condensational growth
- [x] `CoagulationProcess` — stochastic jump with Majorant/Null-event sampling
- [x] `EmissionProcess` — Poisson point process injection
- [x] `DilutionProcess` — death/birth for entrainment and dilution

### Coagulation Kernels
- [x] `BrownianKernel` — Brownian diffusion
- [x] `GravitationalKernel` — gravitational settling
- [x] `AyalaTurbulentKernel` — turbulent coagulation
- [x] `CompositeKernel` — sum of multiple kernels
- [x] `make_kernel` convenience constructor for cloud droplet composite kernel

### Diagnostics
- [x] `number_concentration`, `mass_concentration`, `species_mass_concentration`
- [x] `reconstruct_volumes` — volume history from mass conservation
- [x] `extract_concentrations` — N(t) and M(t) extraction
- [x] `particle_diameters` — mass-to-diameter conversion
- [x] `compute_size_distribution` — dN/dlogD matrix for heatmaps
- [x] `check_mass_conservation` — conservation validation

### Plotting
- [x] `plot_concentration_evolution` — low-level concentration plot
- [x] `plot_size_distribution_heatmap` — low-level heatmap
- [x] `plot_kernel_contributions` — bar chart for kernel comparison
- [x] `plot_simulation_summary` — high-level combined summary plot

### Utils
- [x] `bin_size_distribution` — histogram binning
- [x] `standard_aerosol_atmosphere` / `standard_cloud_atmosphere`
- [x] `lognormal_masses` / `diameters_from_masses`

### Examples
- [x] Aerosol Brownian coagulation with bimodal distribution
- [x] Cloud droplet turbulent coagulation with composite kernel

## Completed (2026-05-12 to 2026-05-15)

### Data I/O
- [x] HDF5 checkpoint/restart (`save_checkpoint`, `load_checkpoint`) with schema v1.0.0
- [x] JLD2 fallback for Julia-native workflows (`save_checkpoint_jld2`, `load_checkpoint_jld2`)
- [x] Chunked HDF5 append for time-series diagnostics (`init_diagnostics_file`, `save_diagnostics`)
- [x] Export diagnostics to CSV (`export_diagnostics_to_csv`)
- [x] RNG restoration and checkpoint enumeration (`restore_rng`, `list_checkpoints`)
- [x] HDF5 2D dataset dimension ordering fix for cross-language compatibility with h5py
- [x] Python post-processing scripts for HDF5 diagnostics (single combined figure per example)
- [x] Example outputs written to `examples/` folder via `@__DIR__`

### CI / Automation
- [x] JuliaFormatter integration and format check in CI
- [x] Code coverage reporting via Codecov
- [x] Aqua.jl code quality checks (unbound type params, undefined exports, stale deps, piracy)
- [x] TagBot for automatic release tagging
- [x] CompatHelper for dependency updates

---

## Planned Improvements

### Documentation
- [ ] Update Documenter.jl docs to include new diagnostics, plotting, and utils APIs
- [ ] Add API reference pages for each module
- [ ] Add tutorial: "Writing a custom coagulation kernel"
- [ ] Add tutorial: "Post-processing workflow"

### Diagnostics Extensions
- [ ] Higher-order moments: variance, skewness, kurtosis of size distribution
- [ ] Effective diameter and mean volume diameter
- [ ] Geometric standard deviation of size distribution
- [ ] Collision efficiency factor (for charged particles)
- [ ] Growth rate analysis
- [ ] Residence time distribution

### Initial Conditions
- [ ] Multi-modal lognormal (arbitrary number of modes)
- [ ] Monodisperse initial conditions
- [ ] Arbitrary distribution sampling (specify PDF/CDF)
- [ ] Read initial conditions from measurement data (CSV/NetCDF)

### Additional Coagulation Kernels
- [ ] Shear kernel (laminar and turbulent shear)
- [ ] Acoustic/ultrasonic coagulation kernel
- [ ] Electric charge effects (Coulomb + image charge)
- [ ] van der Waals correction factor
- [ ] Fractal aggregate kernel

### Multi-Species Support
- [ ] Multi-component particles with different species densities
- [ ] Species-dependent condensation rates
- [ ] Partial coagulation (species redistribution on merge)

### Performance
- [ ] GPU acceleration for coagulation rate matrix computation
- [ ] Parallel majorant computation
- [ ] Adaptive time-stepping for jump processes
- [ ] Benchmark suite with performance regression detection

### Data I/O
- [ ] Import initial conditions from observational data (CSV/NetCDF)

### Uncertainty Quantification
- [ ] Ensemble simulation runner
- [ ] Sensitivity analysis (Sobol indices, Morris screening)
- [ ] Parameter estimation / inverse problem framework

### Testing & CI
- [ ] Example scripts run in CI (smoke tests)

### Registry & Distribution
- [ ] Register v0.2.0 to Julia General Registry
- [ ] Add CITATION.cff

## Long-term Vision

- [ ] Coupled gas-particle chemistry module
- [ ] 3D spatial transport (advection + diffusion)
- [ ] Adaptive particle number (variable n_sim)
- [ ] Machine learning surrogate for coagulation rates
