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
kernel = BrownianKernel(293.15, 1.81e-5, SVector(1800.0))  # T, μ_f, ρ
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
