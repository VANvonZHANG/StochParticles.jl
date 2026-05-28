# Tutorial

```@setup tutorial
using StochParticles
using StaticArrays
using OrdinaryDiffEq
```

This tutorial walks through building a complete particle population simulation with all four physics processes.

## Setting Up the Particle System

Each particle is represented as an `SVector{A, Float64}` where `A` is the number of chemical species. For a single-species system:

```@repl tutorial
n_sim = 50

# Initial particles: each with 1 fg of mass
particles = fill(SVector(1.0e-15), n_sim)
```

## Defining Physics Processes

### Condensation (ODE Drift)

Condensation provides continuous growth via an ODE drift term:

```@repl tutorial
cond = CondensationProcess((μ, g, t) -> -0.01 .* μ)
```

The flux function `(μ, g, t) -> SVector` receives the particle state `μ`, gas-phase concentration `g(t)`, and time `t`.

### Coagulation (Stochastic Jump)

Brownian coagulation uses the diffusion kernel with majorant sampling:

```@repl tutorial
kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))  # T, p, ρ
coag = CoagulationProcess(kernel, GlobalMajorant())
```

When two particles coagulate, the package uses CNMC (Constant Number Monte Carlo) to maintain the particle count: merge + clone + volume rescale.

### Emission (Stochastic Jump)

New particles are injected via a Poisson process:

```@repl tutorial
emit = EmissionProcess(0.1, t -> SVector(1.0e-15))
```

The first argument is the total emission rate; the second is a sampler function `t -> SVector{A}`.

### Dilution (Stochastic Jump)

Dilution removes random particles and injects background particles:

```@repl tutorial
dil = DilutionProcess(t -> 0.05, t -> SVector(0.5e-15))
```

First argument: dilution rate `λ(t)`. Second argument: background particle sampler.

## Assembling and Solving

Combine all processes into a `ParticleProblem` and solve:

```@repl tutorial
gas_fn = t -> SVector(0.0)

prob = ParticleProblem(particles, 1.0, gas_fn, (cond, coag, emit, dil);
                       tspan=(0.0, 5.0), n_sim=n_sim)

sol = solve(prob, Tsit5())
```

The returned `sol` is a standard SciML solution object.

## Computing Diagnostics

```@repl tutorial
sys = prob.prob.p
A_val = species_val(sys)
u_final = sol.u[end]

n0 = number_concentration(sys)
M = mass_concentration(u_final, A_val, sys)
```

- `number_concentration(sys)` — zeroth moment [particles/m³]
- `mass_concentration(u, Val(A), sys)` — first moment [kg/m³]
- `species_mass_concentration(u, idx, Val(A), sys)` — single-species mass [kg/m³]

## QSSA Condensation Simulation

This section demonstrates a multi-species condensation simulation with QSSA (Quasi-Steady State Approximation) pre-equilibration to prevent unphysical negative water masses.

### Physical Setup

```@repl tutorial
T0 = 293.15
p_sat = saturation_vapor_pressure(T0)
S_target = 0.003
p_v0 = p_sat * (1.0 + S_target)

densities = SVector(1770.0, 1000.0)
h2o_idx = 2

avg_thermo = ThermodynamicsParams(
    SVector(0.455, 0.0), 0.072, 1000.0, 18.015e-3,
    2.5e6, 461.5, 2.5e-5, 2.4e-2)
```

### Create Particles

Generate dry particles with no initial water:

```@repl tutorial
particles_dry = lognormal_masses(100, 5.0e-8, 1.6, densities)
particles = [SVector{2, Float64}(m[1], 0.0) for m in particles_dry]
```

### Pre-Equilibrate (QSSA)

Set non-activated particles to their Köhler equilibrium before the ODE solve:

```@repl tutorial
pre_equilibrate!(particles, avg_thermo, densities, T0, p_v0; h2o_idx = h2o_idx)
```

### Solve with H2OCondensationProcess

```@repl tutorial
gas_fn = t -> SVector(T0, p_v0)
cond = H2OCondensationProcess(avg_thermo, densities; h2o_idx = h2o_idx, w = 0.0)

prob = ParticleProblem(particles, 1.0e-6, gas_fn, (cond,);
    tspan = (0.0, 600.0), n_sim = 100)

sol = solve(prob, Tsit5(); saveat = 0.0:60.0:600.0)
```

During integration, `H2OCondensationProcess` automatically freezes non-activated particles (zero flux) while allowing activated particles to condense normally.
