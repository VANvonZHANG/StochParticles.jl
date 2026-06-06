# Smooth Distribution Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade dN/dlogD computation with KDE and smoothed histogram methods to produce smooth distribution curves.

**Architecture:** Three methods dispatched via `method` keyword in `compute_size_distribution`: `:histogram` (StatsBase rewrite), `:histogram_smooth` (oversampled bins + cubic spline), `:kde` (Gaussian KDE via KernelDensity.jl). Replace hand-written `bin_size_distribution` with `StatsBase.fit(Histogram, ...)`. New files `kde.jl` and `smooth_histogram.jl` in `src/diagnostics/`.

**Tech Stack:** Julia, KernelDensity.jl, Interpolations.jl, StatsBase.jl, Plots.jl

---

### Task 1: Add Dependencies

**Files:**
- Modify: `Project.toml`

- [ ] **Step 1: Add KernelDensity, Interpolations, StatsBase to Project.toml**

Add after the `SpecialFunctions` line in `[deps]`:

```toml
Interpolations = "a2c51105-..."
KernelDensity = "5ab10a40-..."
StatsBase = "2913bbd2-..."
```

Add in `[compat]` section:

```toml
Interpolations = "0.15"
KernelDensity = "0.6"
StatsBase = "0.34"
```

The full `[deps]` becomes:

```toml
[deps]
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
Interpolations = "a2c51105-7a91-4a01-9e4b-beb2a4e1e8fe"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
JumpProcesses = "ccbc3e58-028d-4f4c-8cd5-9ae44345cda5"
KernelDensity = "5ab10a40-5f51-46c3-a5b5-634e9104b6e5"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
StatsBase = "2913bbd2-ae8a-5f71-85c6-3e8f3aeaa6e1"
```

- [ ] **Step 2: Resolve dependencies**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.resolve()'`

Expected: dependency resolution succeeds without errors.

- [ ] **Step 3: Commit**

```bash
git add Project.toml
git commit -m "chore: add KernelDensity, Interpolations, StatsBase dependencies"
```

---

### Task 2: Rewrite `bin_size_distribution` Using StatsBase

**Files:**
- Modify: `src/utils/binning.jl`
- Modify: `test/test_utils.jl`

Replace the hand-written O(n×m) linear scan with `StatsBase.fit(Histogram, ...)`. Keep the same function signature and export for backward compatibility.

- [ ] **Step 1: Update `bin_size_distribution` implementation**

Replace the full contents of `src/utils/binning.jl` with:

```julia
# src/utils/binning.jl
using StatsBase

"""
    bin_size_distribution(diams, bin_edges) -> Vector{Int}

Histogram particle diameters into bins defined by `bin_edges`.
Each diameter `d` is counted in the first bin where `bin_edges[i] <= d < bin_edges[i+1]`.

Uses `StatsBase.fit(Histogram, ...)` internally.
"""
function bin_size_distribution(diams::Vector{Float64}, bin_edges::Vector{Float64})
    if length(bin_edges) < 2
        throw(ArgumentError("bin_edges must have at least 2 elements"))
    end
    for i in 2:length(bin_edges)
        if bin_edges[i] <= bin_edges[i - 1]
            throw(ArgumentError("bin_edges must be strictly increasing"))
        end
    end
    h = fit(Histogram, diams, bin_edges; closed = :left)
    return h.weights
end
```

Note: `closed = :left` matches the existing behavior (`bin_edges[i] <= d < bin_edges[i+1]`).

- [ ] **Step 2: Add `using StatsBase` to module**

In `src/StochParticles.jl`, add this line after `using SpecialFunctions` (line 8):

```julia
using StatsBase
```

This makes StatsBase available throughout the module so `binning.jl` can use `fit(Histogram, ...)`.

- [ ] **Step 3: Run existing binning tests to verify backward compat**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Utils"])'`

Expected: all `bin_size_distribution` tests pass with identical results.

- [ ] **Step 4: Commit**

```bash
git add src/utils/binning.jl src/StochParticles.jl
git commit -m "refactor: replace hand-written binning with StatsBase.fit(Histogram)"
```

---

### Task 3: Create `kde.jl` — KDE on log10(d)

**Files:**
- Create: `src/diagnostics/kde.jl`
- Modify: `test/test_plotting.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/test_plotting.jl`, inside the `@testset "Plotting" begin` block, after the existing test sets:

```julia
    @testset "kde_log_diameter" begin
        # Generate lognormal-like diameters for testing
        using Random
        Random.seed!(42)
        # 1000 particles centered around d ≈ 1e-6 m (1 μm)
        diams = 1e-6 .* exp.(0.3 .* randn(1000))

        bin_edges = 10.0 .^ range(-7, -5; length=11)  # 10 bins
        bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])
        V_t = 1.0  # unit volume

        result = StochParticles.kde_log_diameter(diams, bin_centers, V_t)

        # Result has one value per bin center
        @test length(result) == length(bin_centers)

        # All values non-negative
        @test all(result .>= 0.0)

        # Peak should be near 1e-6 (the center of our distribution)
        peak_bin = argmax(result)
        @test bin_centers[peak_bin] ≈ 1e-6 atol = 5e-6

        # With bandwidth_factor = 0.1 (very narrow), result should still be valid
        result_narrow = StochParticles.kde_log_diameter(
            diams, bin_centers, V_t; bandwidth_factor = 0.1)
        @test length(result_narrow) == length(bin_centers)
        @test all(result_narrow .>= 0.0)

        # With bandwidth_factor = 5.0 (very wide), result should be smoother
        result_wide = StochParticles.kde_log_diameter(
            diams, bin_centers, V_t; bandwidth_factor = 5.0)
        @test length(result_wide) == length(bin_centers)
        @test all(result_wide .>= 0.0)

        # Wider bandwidth should produce smaller max value (more spread out)
        @test maximum(result_wide) < maximum(result_narrow)
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Plotting"])'`

Expected: FAIL with `UndefVarError: kde_log_diameter not defined`

- [ ] **Step 3: Create `src/diagnostics/kde.jl`**

```julia
# src/diagnostics/kde.jl
using KernelDensity
using Interpolations
using Statistics

"""
    kde_log_diameter(diameters, bin_centers, V_t;
                     bandwidth_factor=1.0, n_eval_points=200) -> Vector{Float64}

Compute smooth dN/dlogD via Gaussian KDE on log10(diameter) space.

# Arguments
- `diameters`: particle diameters [m]
- `bin_centers`: bin center diameters [m] where dN/dlogD is evaluated
- `V_t`: computational volume [m³]
- `bandwidth_factor`: multiplier for Silverman bandwidth (default 1.0)
- `n_eval_points`: number of fine-grid evaluation points (default 200)

# Returns
- `Vector{Float64}`: dN/dlogD values at each bin center [m⁻³]
"""
function kde_log_diameter(
        diameters::Vector{Float64}, bin_centers::Vector{Float64}, V_t::Float64;
        bandwidth_factor::Float64 = 1.0, n_eval_points::Int = 200)
    if isempty(diameters)
        return zeros(Float64, length(bin_centers))
    end

    x = log10.(diameters)
    n_particles = length(x)
    number_conc = n_particles / V_t

    # Silverman bandwidth: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
    sigma = std(x)
    if sigma == 0.0
        # All particles have the same diameter — use a small default bandwidth
        h = 0.1 * bandwidth_factor
    else
        iqr_val = quantile(x, 0.75) - quantile(x, 0.25)
        h = 0.9 * min(sigma, iqr_val / 1.34) * n_particles^(-0.2) * bandwidth_factor
    end

    # Evaluate KDE on a fine grid spanning the data range
    eval_points = range(minimum(x) - 3h, maximum(x) + 3h; length = n_eval_points)
    kde_result = kde(x, eval_points; bandwidth = h)

    # Convert probability density to dN/dlogD
    dNdlogD_fine = kde_result.density .* number_conc

    # Interpolate to bin center positions
    itp = linear_interpolation(eval_points, dNdlogD_fine; extrapolation = 0.0)
    bin_log = log10.(bin_centers)
    dNdlogD = itp.(bin_log)

    return max.(dNdlogD, 0.0)
end
```

- [ ] **Step 4: Add include to module**

In `src/StochParticles.jl`, add after line 72 (`include("diagnostics/distributions.jl")`):

```julia
include("diagnostics/kde.jl")
include("diagnostics/smooth_histogram.jl")
```

(Note: `smooth_histogram.jl` doesn't exist yet — it will be created in Task 4. If this causes an error, you can add the `kde.jl` include only and add the other line in Task 4.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Plotting"])'`

Expected: PASS (the new `kde_log_diameter` test and all existing tests)

- [ ] **Step 6: Commit**

```bash
git add src/diagnostics/kde.jl src/StochParticles.jl test/test_plotting.jl
git commit -m "feat: add kde_log_diameter for smooth KDE-based dN/dlogD"
```

---

### Task 4: Create `smooth_histogram.jl` — Oversampled Bins + Spline

**Files:**
- Create: `src/diagnostics/smooth_histogram.jl`
- Modify: `test/test_plotting.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/test_plotting.jl`, inside the `@testset "Plotting" begin` block, after the `kde_log_diameter` test set:

```julia
    @testset "smooth_histogram_diameter" begin
        using Random
        Random.seed!(42)
        diams = 1e-6 .* exp.(0.3 .* randn(1000))

        bin_edges = 10.0 .^ range(-7, -5; length=11)  # 10 bins
        V_t = 1.0

        result = StochParticles.smooth_histogram_diameter(diams, bin_edges, V_t)

        # Result has one value per bin (length(edges) - 1)
        @test length(result) == length(bin_edges) - 1

        # All values non-negative
        @test all(result .>= 0.0)

        # With smooth_factor=1, should behave like raw histogram (but via spline)
        result_sf1 = StochParticles.smooth_histogram_diameter(
            diams, bin_edges, V_t; smooth_factor = 1)
        @test length(result_sf1) == length(bin_edges) - 1
        @test all(result_sf1 .>= 0.0)

        # With smooth_factor=5, still produces valid results
        result_sf5 = StochParticles.smooth_histogram_diameter(
            diams, bin_edges, V_t; smooth_factor = 5)
        @test length(result_sf5) == length(bin_edges) - 1
        @test all(result_sf5 .>= 0.0)

        # Empty input returns zeros
        result_empty = StochParticles.smooth_histogram_diameter(
            Float64[], bin_edges, V_t)
        @test all(result_empty .== 0.0)
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Plotting"])'`

Expected: FAIL with `UndefVarError: smooth_histogram_diameter not defined`

- [ ] **Step 3: Create `src/diagnostics/smooth_histogram.jl`**

```julia
# src/diagnostics/smooth_histogram.jl
using StatsBase
using Interpolations

"""
    smooth_histogram_diameter(diameters, bin_edges, V_t;
                              smooth_factor=3) -> Vector{Float64}

Compute smooth dN/dlogD via oversampled histogram binning + cubic spline interpolation.

# Arguments
- `diameters`: particle diameters [m]
- `bin_edges`: diameter bin edges [m], strictly increasing
- `V_t`: computational volume [m³]
- `smooth_factor`: oversampling multiplier for fine bins (default 3)

# Returns
- `Vector{Float64}`: dN/dlogD values at each original bin center [m⁻³]
"""
function smooth_histogram_diameter(
        diameters::Vector{Float64}, bin_edges::Vector{Float64}, V_t::Float64;
        smooth_factor::Int = 3)
    n_bins = length(bin_edges) - 1

    if isempty(diameters)
        return zeros(Float64, n_bins)
    end

    x = log10.(diameters)
    log_edges = log10.(bin_edges)

    # Create oversampled fine bin edges
    n_fine = n_bins * smooth_factor
    fine_edges = collect(range(log_edges[1], log_edges[end]; length = n_fine + 1))
    fine_centers = 0.5 .* (fine_edges[1:(end - 1)] .+ fine_edges[2:end])

    # Histogram on fine grid using StatsBase
    h = fit(Histogram, x, fine_edges; closed = :left)
    dlogD_fine = diff(fine_edges)

    # dN/dlogD on fine grid: counts / (log width × volume)
    dNdlogD_fine = Float64.(h.weights) ./ dlogD_fine ./ V_t

    # Cubic spline interpolation
    itp = cubic_spline_interpolation(fine_centers, dNdlogD_fine;
        extrapolation = 0.0)

    # Evaluate at original bin centers (geometric mean in log space = arithmetic mean of log)
    orig_log_centers = 0.5 .* (log_edges[1:(end - 1)] .+ log_edges[2:end])
    dNdlogD = itp.(orig_log_centers)

    return max.(dNdlogD, 0.0)
end
```

- [ ] **Step 4: Add include to module (if not already added in Task 3)**

If `include("diagnostics/smooth_histogram.jl")` was not added in Task 3 Step 4, add it now in `src/StochParticles.jl` after the `kde.jl` include.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Plotting"])'`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/diagnostics/smooth_histogram.jl src/StochParticles.jl test/test_plotting.jl
git commit -m "feat: add smooth_histogram_diameter with oversampled bins + spline"
```

---

### Task 5: Update `compute_size_distribution` with Method Dispatch

**Files:**
- Modify: `src/diagnostics/distributions.jl`
- Modify: `test/test_plotting.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/test_plotting.jl`, inside the `@testset "Plotting" begin` block, after the existing `plot_simulation_summary` test set:

```julia
    @testset "compute_size_distribution methods" begin
        n_sim = 200
        kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())
        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        prob = ParticleProblem(
            particles, 1.0, gas_fn, (coag,); tspan = (0.0, 1.0), n_sim = n_sim)
        sol = solve(prob, Tsit5(); saveat = 0.1)
        bin_edges = 10.0 .^ range(-9, -5; length = 26)

        # Test :histogram method (existing behavior)
        t1, c1, m1 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0; n_snapshots = 5, method = :histogram)
        @test length(t1) == 5
        @test length(c1) == 25
        @test size(m1) == (25, 5)
        @test all(m1 .>= 0.0)

        # Test :kde method (new default)
        t2, c2, m2 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0; n_snapshots = 5, method = :kde)
        @test length(t2) == 5
        @test length(c2) == 25
        @test size(m2) == (25, 5)
        @test all(m2 .>= 0.0)

        # Test :histogram_smooth method
        t3, c3, m3 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0; n_snapshots = 5, method = :histogram_smooth)
        @test length(t3) == 5
        @test length(c3) == 25
        @test size(m3) == (25, 5)
        @test all(m3 .>= 0.0)

        # All methods should return the same bin centers and snapshot times
        @test c1 ≈ c2
        @test c1 ≈ c3
        @test t1 ≈ t2
        @test t1 ≈ t3

        # Test invalid method throws
        @test_throws ArgumentError compute_size_distribution(
            sol, prob, bin_edges, 1800.0; method = :invalid)

        # Test that :kde passes bandwidth_factor through
        t4, c4, m4 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0;
            n_snapshots = 3, method = :kde, bandwidth_factor = 2.0)
        @test size(m4) == (25, 3)
        @test all(m4 .>= 0.0)
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Plotting"])'`

Expected: FAIL — `compute_size_distribution` doesn't accept `method` keyword yet.

- [ ] **Step 3: Rewrite `compute_size_distribution` with method dispatch**

Replace the full contents of `src/diagnostics/distributions.jl` with:

```julia
# src/diagnostics/distributions.jl

"""
    particle_diameters(u, sys, densities) -> Vector{Float64}

Compute sphere-equivalent diameters [m] for all active particles from their masses.

For multi-species particles, the equivalent volume is computed as:
    V_p = Σ μ_k / ρ_k

# Arguments
- `densities::SVector{A, Float64}`: per-species densities [kg/m³]
"""
function particle_diameters(
        u::Vector{Float64}, sys::ParticleSystem, densities::SVector{A, Float64}) where {A}
    A_val = species_val(sys)
    diams = Vector{Float64}(undef, sys.n_active)
    for i in 1:(sys.n_active)
        μ = get_particle(u, i, A_val)
        V_p = 0.0
        for k in 1:A
            V_p += μ[k] / densities[k]
        end
        diams[i] = (6.0 * V_p / π)^(1.0 / 3.0)
    end
    return diams
end

# Backward-compatible single-species shortcut
function particle_diameters(
        u::Vector{Float64}, sys::ParticleSystem, rho::Float64)
    return particle_diameters(u, sys, SVector{1, Float64}(rho))
end

"""
    compute_size_distribution(sol, prob, bin_edges, rho;
                              n_snapshots=30, method=:kde,
                              bandwidth_factor=1.0, n_eval_points=200,
                              smooth_factor=3)
        -> (snapshot_times, bin_centers, dNdlogD_matrix)

Compute size distribution matrix dN/dlogD over time.

# Arguments
- `sol`: SciML solution object
- `prob`: ParticleProblem (JumpProblem)
- `bin_edges`: diameter bin edges [m], strictly increasing
- `rho`: particle density [kg/m³]
- `n_snapshots`: number of time snapshots to evaluate (default 30)
- `method`: `:histogram`, `:histogram_smooth`, or `:kde` (default `:kde`)
- `bandwidth_factor`: KDE bandwidth multiplier (default 1.0, only for `:kde`)
- `n_eval_points`: KDE evaluation grid size (default 200, only for `:kde`)
- `smooth_factor`: histogram oversampling factor (default 3, only for `:histogram_smooth`)

# Returns
- `snapshot_times::Vector{Float64}`: time points
- `bin_centers::Vector{Float64}`: geometric mean of each bin edge pair [m]
- `dNdlogD_matrix::Matrix{Float64}`: matrix of shape (n_bins, n_snapshots)
"""
function compute_size_distribution(
        sol, prob, bin_edges::Vector{Float64}, rho::Float64;
        n_snapshots::Int = 30,
        method::Symbol = :kde,
        bandwidth_factor::Float64 = 1.0,
        n_eval_points::Int = 200,
        smooth_factor::Int = 3)
    if length(bin_edges) < 2
        throw(ArgumentError("bin_edges must have at least 2 elements"))
    end
    for i in 2:length(bin_edges)
        if bin_edges[i] <= bin_edges[i - 1]
            throw(ArgumentError("bin_edges must be strictly increasing"))
        end
    end

    if method ∉ (:histogram, :histogram_smooth, :kde)
        throw(ArgumentError(
            "method must be :histogram, :histogram_smooth, or :kde, got :$method"))
    end

    sys = prob.prob.p
    A = species_val(sys)
    tspan = (sol.t[1], sol.t[end])
    snapshot_times = range(tspan[1], tspan[2]; length = n_snapshots)
    n_bins = length(bin_edges) - 1

    dNdlogD_matrix = zeros(Float64, n_bins, n_snapshots)
    volumes = reconstruct_volumes(sol, prob)
    dlogD = diff(log10.(bin_edges))
    bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])

    for (j, target_t) in enumerate(snapshot_times)
        t_idx = argmin(abs.(sol.t .- target_t))
        u = sol.u[t_idx]
        V_t = volumes[t_idx]

        if sys.n_active == 0
            continue
        end

        diams = particle_diameters(u, sys, rho)

        if method == :histogram
            counts = bin_size_distribution(diams, bin_edges)
            dNdlogD_matrix[:, j] = Float64.(counts) ./ dlogD ./ V_t

        elseif method == :kde
            dNdlogD_matrix[:, j] = kde_log_diameter(
                diams, bin_centers, V_t;
                bandwidth_factor = bandwidth_factor,
                n_eval_points = n_eval_points)

        elseif method == :histogram_smooth
            dNdlogD_matrix[:, j] = smooth_histogram_diameter(
                diams, bin_edges, V_t;
                smooth_factor = smooth_factor)
        end
    end

    return collect(snapshot_times), bin_centers, dNdlogD_matrix
end
```

Key changes from the original:
- Added `method`, `bandwidth_factor`, `n_eval_points`, `smooth_factor` keyword arguments
- Method validation with clear error message
- `:histogram` path preserves existing behavior (via StatsBase-backed `bin_size_distribution`)
- `:kde` and `:histogram_smooth` call the new functions from Tasks 3-4
- Added `sys.n_active == 0` guard before computing diameters

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Plotting"])'`

Expected: PASS — all existing tests + new method dispatch tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics/distributions.jl test/test_plotting.jl
git commit -m "feat: add :kde and :histogram_smooth methods to compute_size_distribution"
```

---

### Task 6: Update `plot_simulation_summary` to Forward Method Params

**Files:**
- Modify: `src/plotting/recipes.jl`
- Modify: `test/test_plotting.jl`

- [ ] **Step 1: Update function signature and call**

In `src/plotting/recipes.jl`, update the `plot_simulation_summary` function to accept and forward the new parameters.

Change the function signature (line 35-38) from:

```julia
function plot_simulation_summary(sol, prob, bin_edges::Vector{Float64}, rho::Float64;
        time_unit::String = "s", diameter_unit::String = "μm",
        n_snapshots::Int = 30, layout::Tuple{Int, Int} = (1, 2),
        size::Tuple{Int, Int} = (1400, 500), kwargs...)
```

to:

```julia
function plot_simulation_summary(sol, prob, bin_edges::Vector{Float64}, rho::Float64;
        time_unit::String = "s", diameter_unit::String = "μm",
        n_snapshots::Int = 30,
        method::Symbol = :kde,
        bandwidth_factor::Float64 = 1.0,
        n_eval_points::Int = 200,
        smooth_factor::Int = 3,
        layout::Tuple{Int, Int} = (1, 2),
        size::Tuple{Int, Int} = (1400, 500), kwargs...)
```

Update the docstring (lines 4-6) to add the new parameters:

```julia
    plot_simulation_summary(sol, prob, bin_edges, rho; time_unit="s", diameter_unit="μm",
                             n_snapshots=30, method=:kde, bandwidth_factor=1.0,
                             n_eval_points=200, smooth_factor=3,
                             layout=(1,2), size=(1400,500), kwargs...)
```

Update the `compute_size_distribution` call (lines 57-59) from:

```julia
    snapshot_times, bin_centers,
    matrix = compute_size_distribution(
        sol, prob, bin_edges, rho; n_snapshots = n_snapshots)
```

to:

```julia
    snapshot_times, bin_centers,
    matrix = compute_size_distribution(
        sol, prob, bin_edges, rho;
        n_snapshots = n_snapshots,
        method = method,
        bandwidth_factor = bandwidth_factor,
        n_eval_points = n_eval_points,
        smooth_factor = smooth_factor)
```

- [ ] **Step 2: Update the existing test to use `:histogram` to avoid breaking on few particles**

In `test/test_plotting.jl`, update the `plot_simulation_summary` test (around line 28-44). The existing test uses `n_sim=50` with `bin_edges = [0.0, 1.0e-6, 2.0e-6]` — only 2 bins. KDE needs more bins to work well, so pass `method=:histogram` for the existing test to preserve its behavior:

Change the call at line 39 from:
```julia
        pl = plot_simulation_summary(sol, prob, bin_edges, 1000.0)
```
to:
```julia
        pl = plot_simulation_summary(sol, prob, bin_edges, 1000.0; method = :histogram)
```

And line 43 from:
```julia
        pl_min = plot_simulation_summary(sol, prob, bin_edges, 1000.0; time_unit = "min")
```
to:
```julia
        pl_min = plot_simulation_summary(sol, prob, bin_edges, 1000.0;
            time_unit = "min", method = :histogram)
```

- [ ] **Step 3: Run all plotting tests**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test(test_args=["Plotting"])'`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/plotting/recipes.jl test/test_plotting.jl
git commit -m "feat: forward method params through plot_simulation_summary"
```

---

### Task 7: Update `save_diagnostics` to Use StatsBase

**Files:**
- Modify: `src/io/diagnostics.jl`

- [ ] **Step 1: Replace `bin_size_distribution` call in `save_diagnostics`**

In `src/io/diagnostics.jl`, the call at line 130:

```julia
                counts = bin_size_distribution(diams, bin_edges)
```

This already works because `bin_size_distribution` is rewritten to use StatsBase internally (Task 2). No code change needed here — the function signature is preserved.

However, verify the test still passes:

- [ ] **Step 2: Run I/O tests**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: all tests pass.

- [ ] **Step 3: Commit** (only if changes were needed)

Skip this step if no code changes were made.

---

### Task 8: Update Module Includes and Exports

**Files:**
- Modify: `src/StochParticles.jl`

By this point the include lines for `kde.jl` and `smooth_histogram.jl` should already be in place from Tasks 3-4. Verify and clean up:

- [ ] **Step 1: Verify the module file has the correct include order**

The `# ---- Diagnostics ----` section in `src/StochParticles.jl` should look like:

```julia
# ---- Diagnostics ----
include("diagnostics/moments.jl")
include("diagnostics/mixing_state.jl")
include("diagnostics/reconstruction.jl")
include("diagnostics/distributions.jl")
include("diagnostics/kde.jl")
include("diagnostics/smooth_histogram.jl")
include("diagnostics/validation.jl")
include("diagnostics/activation.jl")
```

The new includes must come after `distributions.jl` (which defines `particle_diameters`) and before `validation.jl`.

- [ ] **Step 2: Verify exports**

The export line for `compute_size_distribution` already exists at line 41. No new exports needed — `kde_log_diameter` and `smooth_histogram_diameter` are internal functions (accessible via `StochParticles.function_name` for testing but not exported).

- [ ] **Step 3: Run full test suite**

Run: `cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: all tests pass.

- [ ] **Step 4: Commit** (only if changes were needed)

---

### Task 9: Update Examples to Use KDE

**Files:**
- Modify: `examples/aerosol_brownian_coagulation.jl`
- Modify: `examples/cloud_droplet_turbulent_coagulation.jl`
- Modify: `examples/mixing_state_coagulation.jl`

- [ ] **Step 1: Update `aerosol_brownian_coagulation.jl`**

Find the `plot_simulation_summary` call and change it to explicitly use `:kde`:

```julia
pl = plot_simulation_summary(sol, prob, bin_edges, 1800.0;
    time_unit = "min", diameter_unit = "nm",
    method = :kde, bandwidth_factor = 1.0)
```

- [ ] **Step 2: Update `cloud_droplet_turbulent_coagulation.jl`**

Find the `compute_size_distribution` call (if present) and add `method = :kde`. If only `plot_simulation_summary` is used, add the params there:

```julia
pl_summary = plot_simulation_summary(sol, prob, bin_edges, 1000.0;
    time_unit = "s", diameter_unit = "μm",
    method = :kde, bandwidth_factor = 1.0)
```

- [ ] **Step 3: Update `mixing_state_coagulation.jl`**

Same pattern — add `method = :kde` to the relevant distribution/plotting calls.

- [ ] **Step 4: Run each example to verify no errors**

```bash
cd /home/zhangfan/Project/20251205_Julia_PartMC/StochParticles.jl
julia --project=. examples/aerosol_brownian_coagulation.jl
```

Expected: script completes and generates plot without errors.

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "feat: update examples to use KDE for smooth distribution plots"
```

---

## Self-Review Checklist

### 1. Spec Coverage

| Spec Requirement | Task |
|---|---|
| Three methods: `:histogram`, `:histogram_smooth`, `:kde` | Task 5 |
| KDE via KernelDensity.jl on log10(d) | Task 3 |
| Silverman bandwidth with `bandwidth_factor` | Task 3 |
| Smooth histogram with oversample + spline | Task 4 |
| `bin_size_distribution` replaced by StatsBase | Task 2 |
| `compute_size_distribution` gains `method` param | Task 5 |
| `plot_simulation_summary` forwards params | Task 6 |
| Default method `:kde` | Task 5 |
| New deps in Project.toml | Task 1 |
| Tests for all methods | Tasks 3, 4, 5 |
| Examples updated | Task 9 |

### 2. Placeholder Scan

No TBD, TODO, or "implement later" patterns found. Every step contains complete code.

### 3. Type Consistency

- `kde_log_diameter(diams::Vector{Float64}, bin_centers::Vector{Float64}, V_t::Float64)` — matches call in Task 5
- `smooth_histogram_diameter(diams::Vector{Float64}, bin_edges::Vector{Float64}, V_t::Float64)` — matches call in Task 5
- `compute_size_distribution` returns `(Vector{Float64}, Vector{Float64}, Matrix{Float64})` — unchanged from original
- `plot_simulation_summary` forwards `method::Symbol`, `bandwidth_factor::Float64`, `n_eval_points::Int`, `smooth_factor::Int` — matches `compute_size_distribution` signature
