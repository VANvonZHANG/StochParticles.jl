# StochParticles.jl

```@setup
using StochParticles
using StaticArrays
using OrdinaryDiffEq
```

A SciML-native Julia framework for particle-resolved Monte Carlo simulation of aerosol and colloidal population dynamics.

## Overview

StochParticles.jl models particle populations as Piecewise Deterministic Markov Processes (PDMPs), seamlessly combining continuous ODE dynamics (condensational growth) with discrete stochastic events (coagulation, emission, dilution) through the [SciML](https://sciml.ai/) ecosystem.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  User API: ParticleProblem, solve, diagnostics      │
├─────────────────────────────────────────────────────┤
│  Physics Processes (pluggable traits)               │
│  Condensation · Coagulation · Emission · Dilution    │
├─────────────────────────────────────────────────────┤
│  Core: ParticleSystem, CNMC, JumpProblem assembly   │
└─────────────────────────────────────────────────────┘
```

### Features

- **Compile-time specialization** — species count `A` is a type parameter; all inner loops are fully type-stable
- **Trait-based process system** — each physics process declares whether it provides ODE drift and/or jumps
- **CNMC particle management** — Constant Number Monte Carlo with mass-conserving merge, clone, and volume rescale
- **SciML integration** — produces standard `JumpProblem` objects solvable with `Tsit5()`, `SSC1()`, or any SciML algorithm

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/VANvonZHANG/StochParticles.jl")
```

## Quick Start

```@repl
cond = CondensationProcess((μ, g, t) -> -0.01 .* μ)
kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
coag = CoagulationProcess(kernel, GlobalMajorant())

particles = fill(SVector(1.0e-15), 50)
gas_fn = t -> SVector(0.0)

prob = ParticleProblem(particles, 1.0, gas_fn, (cond, coag);
                       tspan=(0.0, 10.0), n_sim=50)
sol = solve(prob, Tsit5())
```

## Unit Convention

All quantities use SI units: masses in **kg**, volumes in **m³**, temperatures in **K**, viscosities in **Pa·s**, densities in **kg/m³**.

## Contents

- [Tutorial](tutorial.md) — step-by-step walkthrough of setting up a simulation
- [API Reference](api/core.md) — complete documentation of all exported types and functions
