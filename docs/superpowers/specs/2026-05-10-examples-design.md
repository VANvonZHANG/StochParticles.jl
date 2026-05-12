# StochParticles.jl Examples Design

**Date:** 2026-05-10
**Topic:** examples/ directory with aerosol and cloud droplet coagulation demonstrations
**Approach:** Shared utility module + lean example scripts (Option B)

---

## 1. Goals

Provide two runnable example scripts that demonstrate:
- **Aerosol-scale simulation:** Brownian coagulation of sub-micron particles
- **Cloud droplet-scale simulation:** Composite coagulation (Brownian + gravitational + turbulent) of 10-100 μm droplets

Both examples produce diagnostic plots and validate physical invariants (mass conservation, constant-N constraint).

---

## 2. Directory Structure

```
examples/
├── Project.toml                          # examples-only environment
├── utils.jl                              # shared utility module
├── aerosol_brownian_coagulation.jl       # Example 1
└── cloud_droplet_turbulent_coagulation.jl # Example 2
```

**Rationale:** A shared `utils.jl` avoids duplicating lognormal initialization, size-binning, and plotting helpers across both examples. Each example script focuses on the physics unique to its scale.

---

## 3. Shared Utilities (`utils.jl`)

### 3.1 Initial Distribution Generation

```julia
lognormal_masses(N, d_g, sigma_g, rho) -> Vector{SVector{1,Float64}}
```

Generate `N` particle masses (kg) drawn from a log-normal size distribution.  
- `d_g`: geometric mean diameter [m]  
- `sigma_g`: geometric standard deviation  
- `rho`: particle density [kg/m³]

```julia
diameters_from_masses(masses, rho) -> Vector{Float64}
```

Convert mass vectors back to diameters for diagnostics.

### 3.2 Diagnostics & Binning

```julia
bin_size_distribution(diams, bin_edges) -> Vector{Int}
```

Histogram particle diameters into bins. Bin edges span the relevant range for each example.

```julia
extract_diagnostics(sol, sys, A) -> (t, N_conc, M_conc, diameters_t)
```

Extract time-series data from the SciML solution:
- `t`: time points
- `N_conc`: number concentration [particles/m³]
- `M_conc`: mass concentration [kg/m³]
- `diameters_t`: diameter snapshots at select times

### 3.3 Atmospheric Parameter Presets

```julia
standard_aerosol_atmosphere() -> AtmosphericParameters
# T=293.15 K, P=101325 Pa, rho_p=1800 kg/m³ (sulfate-like)

standard_cloud_atmosphere() -> AtmosphericParameters
# T=288.15 K, P=80000 Pa (~2 km), rho_p=1000 kg/m³ (liquid water)
```

---

## 4. Example 1: Aerosol Brownian Coagulation

**File:** `examples/aerosol_brownian_coagulation.jl`

### 4.1 Physical Scenario

Closed-box simulation of sulfate-like aerosol particles undergoing Brownian diffusion coagulation only.

| Parameter | Value |
|-----------|-------|
| Temperature | 293.15 K |
| Pressure | 101325 Pa |
| Particle density | 1800 kg/m³ |
| Geometric mean diameter | 0.1 μm |
| Geometric std. dev. | 1.5 |
| Initial particle count | 5000 |
| Computational volume | 1.0 m³ |
| Time span | 0 – 3600 s |

### 4.2 Processes

Only `CoagulationProcess` with `BrownianKernel`:
```julia
kernel = BrownianKernel(params.T, params.mu_f, densities)
coag = CoagulationProcess(kernel, GlobalMajorant())
```

No condensation, emission, or dilution.

### 4.3 Diagnostics & Validation

- **Number concentration** `N(t) = n_active / V` — should decrease as particles coagulate.
- **Mass concentration** `M(t)` — must be conserved (CNMC volume rescale).
- **Post-simulation assertions:**
  - `sol.retcode == ReturnCode.Success`
  - `|M(t_end) - M(0)| / M(0) < 1e-6`
  - `n_active == n_sim` (CNMC invariant)

### 4.4 Plots

1. **Concentration evolution** (dual y-axis):  
   Left: `N(t)` [particles/m³]  
   Right: `M(t)` [kg/m³]

2. **Size distribution snapshots** (4 panels):  
   Histograms at t = 0, 600, 1800, 3600 s.  
   X-axis: diameter [μm], logarithmic scale.  
   Y-axis: dN/dlogD [particles/m³].

---

## 5. Example 2: Cloud Droplet Turbulent Coagulation

**File:** `examples/cloud_droplet_turbulent_coagulation.jl`

### 5.1 Physical Scenario

Cumulus cloud environment where droplets grow by the composite of Brownian, gravitational, and turbulent coagulation.

| Parameter | Value |
|-----------|-------|
| Temperature | 288.15 K |
| Pressure | 80000 Pa (~2 km altitude) |
| Particle density | 1000 kg/m³ (liquid water) |
| Geometric mean diameter | 10 μm |
| Geometric std. dev. | 1.3 |
| Initial particle count | 5000 |
| Computational volume | 1.0 m³ |
| Time span | 0 – 600 s |
| Turbulent dissipation ε | 0.01 m²/s³ |
| Taylor Reynolds number R_λ | 50 |

### 5.2 Processes

`CoagulationProcess` with the three-kernel composite from `make_kernel`:
```julia
kernel = make_kernel(params, epsilon, R_lambda, densities)
coag = CoagulationProcess(kernel, GlobalMajorant())
```

### 5.3 Diagnostics & Validation

Same invariants as Example 1, plus:

**Kernel contribution comparison:** For a representative pair (e.g. 10 μm + 20 μm), print the relative magnitude of:
- `K_brown(μ_i, μ_j)`
- `K_grav(μ_i, μ_j)`
- `K_turb(μ_i, μ_j)`
- `K_total(μ_i, μ_j)`

This demonstrates that at cloud-droplet scales, Brownian is negligible while gravitational and turbulent dominate.

### 5.4 Plots

1. **Concentration evolution** (dual y-axis): same style as Example 1.
2. **Size distribution snapshots** (4 panels): at t = 0, 100, 300, 600 s.
3. **Kernel contribution bar chart:** relative contribution of the three mechanisms for a bidisperse pair.

---

## 6. Dependency Management

### 6.1 `examples/Project.toml`

Independent environment; Plots.jl is **not** added to the root package.

```toml
[deps]
JumpProcesses = "..."
OrdinaryDiffEq = "..."
Plots = "..."
StaticArrays = "..."
StochParticles = "..."

[compat]
julia = "1.10"
```

### 6.2 Local Development Setup

```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
```

### 6.3 Running Examples

```bash
julia --project=. aerosol_brownian_coagulation.jl
julia --project=. cloud_droplet_turbulent_coagulation.jl
```

---

## 7. README Integration

Add an "Examples" section to the root README:
- One-sentence description of each example
- Copy-paste commands for running
- Sample output figures (added after examples are implemented)

---

## 8. Future Extensions (Out of Scope)

- Multi-species particles (A > 1)
- Condensation + coagulation coupled example
- Emission / dilution examples
- Literate.jl notebooks for interactive documentation

---

## 9. Summary of Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Example format | Plain `.jl` scripts | User explicitly chose over notebooks |
| Shared code | `utils.jl` module | DRY; common lognormal/binning/plotting logic |
| Number of examples | 2 | Aerosol + cloud, as requested |
| Particles per simulation | 5000 | User request; better statistics |
| Plotting library | Plots.jl | User request; lightweight enough for scripts |
| Dependency isolation | `examples/Project.toml` | Avoids bloating root package with Plots |
