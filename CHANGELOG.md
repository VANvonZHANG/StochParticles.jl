# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
