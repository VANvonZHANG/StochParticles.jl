# StochParticles.jl

[![CI](https://github.com/VANvonZHANG/StochParticles.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VANvonZHANG/StochParticles.jl/actions/workflows/CI.yml)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://VANvonZHANG.github.io/StochParticles.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://VANvonZHANG.github.io/StochParticles.jl/dev/)
[![Codecov](https://codecov.io/gh/VANvonZHANG/StochParticles.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/VANvonZHANG/StochParticles.jl)

**StochParticles.jl** is a SciML-native Julia framework for particle-resolved Monte Carlo simulation of aerosol and colloidal population dynamics. It models particle populations as Piecewise Deterministic Markov Processes (PDMPs), combining ODE drift (condensation) with stochastic jumps (coagulation, emission, dilution) via the [SciML](https://sciml.ai/) ecosystem.

## Features

- **Trait-based process architecture** — define physics processes as composable modules that declare their contributions (ODE drift and/or jumps)
- **Compile-time specialization** — species count is a type parameter; all inner loops are fully type-stable
- **CNMC particle management** — Constant Number Monte Carlo with mass-conserving merge, clone, and volume rescale
- **Pluggable coagulation kernels** — Brownian diffusion, composable kernels, and extensible sampling strategies
- **SciML integration** — produces standard `JumpProblem` objects solvable with `Tsit5()`, `SSC1()`, or any SciML algorithm

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/VANvonZHANG/StochParticles.jl")
```

## Quick Start

Simulate a population of particles undergoing condensation and Brownian coagulation:

```julia
using StochParticles
using StaticArrays
using OrdinaryDiffEq

# Define physics processes
cond = CondensationProcess((μ, g, t) -> -0.01 .* μ)
kernel = BrownianKernel(293.15, 1.81e-5, SVector(1800.0))  # T, μ_f, ρ
coag = CoagulationProcess(kernel, GlobalMajorant())

# Initial conditions: 50 particles, each 1 fg of species 1
particles = fill(SVector(1.0e-15), 50)
gas_fn = t -> SVector(0.0)

# Assemble and solve the PDMP
prob = ParticleProblem(particles, 1.0, gas_fn, (cond, coag);
                       tspan=(0.0, 10.0), n_sim=50)
sol = solve(prob, Tsit5())

# Post-process: compute diagnostic moments
sys = prob.prob.p
A_val = species_val(sys)
u_final = sol.u[end]
println("Number concentration: ", number_concentration(sys), " particles/m³")
println("Mass concentration:   ", mass_concentration(u_final, A_val, sys), " kg/m³")
```

## API Overview

### Core Types

| Type | Description |
|------|-------------|
| `ParticleSystem{A}` | Mutable PDMP parameter container (the `p` in `f(u, p, t)`) |
| `ParticleProblem(...)` | Constructor that assembles a SciML `JumpProblem` |

### Physics Processes

| Process | Type | Mechanism |
|---------|------|-----------|
| `CondensationProcess` | ODE drift | Condensational growth/evaporation |
| `CoagulationProcess` | Stochastic jump | Particle-particle coagulation |
| `EmissionProcess` | Stochastic jump | Particle emission/injection |
| `DilutionProcess` | Stochastic jump | Dilution (death) + entrainment (birth) |

### CNMC Operations

| Function | Description |
|----------|-------------|
| `cnmc_merge!(u, sys, i, j)` | Merge particle j into i |
| `cnmc_clone!(u, sys, slot, source)` | Clone particle into vacated slot |
| `cnmc_volume_rescale!(sys, μ)` | Rescale volume to conserve mass concentration |
| `cnmc_coagulate!(u, sys, Val(A), i, j)` | Full merge + clone + rescale step |

### Diagnostics

| Function | Returns |
|----------|---------|
| `number_concentration(sys)` | N_sim / V [particles/m³] |
| `mass_concentration(u, Val(A), sys)` | Total mass / V [kg/m³] |
| `species_mass_concentration(u, idx, Val(A), sys)` | Single-species mass / V |

## Unit Convention

All quantities use SI units: masses in **kg**, volumes in **m³**, temperatures in **K**, viscosities in **Pa·s**, densities in **kg/m³**.

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).
