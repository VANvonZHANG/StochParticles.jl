# Smooth Distribution Plotting Design

**Date:** 2026-06-06
**Status:** Approved
**Scope:** Upgrade size distribution (dN/dlogD) computation to produce smooth curves

## Problem

The current `compute_size_distribution` pipeline uses raw histogram binning with no smoothing, producing jagged step-function curves. Root causes:

1. Simple O(n×m) linear-scan binning in `bin_size_distribution`
2. No KDE, spline, or any smoothing applied to binned data
3. Most examples use only 20-30 bins across 3-4 orders of magnitude
4. No interpolation between bin centers

## Design Decisions

### Three Methods

The `compute_size_distribution` function gains a `method` parameter with three options:

| Method | `method` value | Algorithm | Dependencies |
|---|---|---|---|
| Raw histogram (existing) | `:histogram` | Current binning, unchanged | None |
| Smoothed histogram | `:histogram_smooth` | Oversampled bins + B-spline | `Interpolations.jl`, `StatsBase.jl` |
| Kernel density estimation | `:kde` | Gaussian KDE on log10(d) | `KernelDensity.jl`, `Interpolations.jl` |

Default method changes from `:histogram` to `:kde`.

### Equal Particle Weights

CNMC particles are treated as equal-weight. Each particle represents N_total / n_particles real particles. No weighted KDE needed.

### Third-Party Libraries Replace Hand-Written Code

- `bin_size_distribution` (hand-written O(n×m) loop) is replaced by `StatsBase.fit(Histogram, ...)`, which is already a transitive dependency of `KernelDensity.jl`.
- `src/utils/binning.jl` is deleted; all binning uses StatsBase.

## Unified Interface

```julia
function compute_size_distribution(sol, sys, bin_edges;
    n_snapshots::Int=50,
    method::Symbol=:kde,
    bandwidth_factor::Float64=1.0,
    n_eval_points::Int=200,
    smooth_factor::Int=3
)
```

### Parameters

- `method`: `:histogram` | `:histogram_smooth` | `:kde` (default: `:kde`)
- `bandwidth_factor`: KDE bandwidth multiplier. `> 1` = smoother, `< 1` = more sensitive (default: `1.0`). Applies only to `:kde` method.
- `n_eval_points`: Number of evaluation points for KDE fine grid (default: `200`)
- `smooth_factor`: Oversampling multiplier for `:histogram_smooth`. Original bins × smooth_factor = fine bins (default: `3`)

## Method: `:kde` Algorithm

```julia
function kde_log_diameter(diameters, bin_centers, V_t;
    bandwidth_factor=1.0, n_eval_points=200)

    x = log10.(diameters)
    n_particles = length(x)
    number_conc = n_particles / V_t  # [particles / m³]

    # 1. Silverman bandwidth
    sigma = std(x)
    iqr_val = quantile(x, 0.75) - quantile(x, 0.25)
    h = 0.9 * min(sigma, iqr_val / 1.34) * n_particles^(-0.2) * bandwidth_factor

    # 2. KDE on fine grid
    eval_points = range(minimum(x), maximum(x); length=n_eval_points)
    kde_result = KernelDensity.kde(x, eval_points; bandwidth=h)

    # 3. Convert to dN/dlogD: density (probability) × number concentration
    density = kde_result.density  # integrates to 1 over log10(d)
    dNdlogD_fine = density .* number_conc  # [particles / m³ / unit(log10(d))]

    # 4. Interpolate to bin_centers
    itp = Interpolations.linear_interpolation(eval_points, dNdlogD_fine)
    bin_log = log10.(bin_centers)
    dNdlogD = itp.(bin_log)

    return max.(dNdlogD, 0.0)
end
```

Key points:
- KDE operates in log10(d) space because diameters span orders of magnitude
- Silverman's rule provides automatic bandwidth selection
- `bandwidth_factor` lets users tune smoothness without modifying internals
- Final interpolation to `bin_centers` maintains compatibility with heatmap plotting

## Method: `:histogram_smooth` Algorithm

```julia
function smooth_histogram_diameter(diameters, bin_edges, V_t;
    smooth_factor=3)

    x = log10.(diameters)
    log_edges = log10.(bin_edges)

    # 1. Oversampled bins
    n_fine = (length(bin_edges) - 1) * smooth_factor
    fine_edges = range(log_edges[1], log_edges[end]; length=n_fine+1)
    fine_centers = 0.5 .* (fine_edges[1:end-1] .+ fine_edges[2:end])

    # 2. StatsBase binning → dN/dlogD (same normalization as existing :histogram)
    h = fit(Histogram, x, fine_edges)
    dlogD_fine = diff(fine_edges)
    dNdlogD_fine = h.weights ./ dlogD_fine ./ V_t  # counts / (log width × volume)

    # 3. B-spline smoothing
    itp = Interpolations.cubic_spline_interpolation(fine_centers, dNdlogD_fine)

    # 4. Interpolate back to original bin_centers
    orig_centers = 0.5 .* (log_edges[1:end-1] .+ log_edges[2:end])
    dNdlogD = itp.(orig_centers)

    return max.(dNdlogD, 0.0)
end
```

Key points:
- `smooth_factor=3` means 25 original bins → 75 fine bins → spline → 25 smooth values
- Oversampling reduces per-bin particle count fluctuation
- Cubic spline eliminates staircase artifacts
- Suitable for users who prefer histogram-based approach but want smoother output

## Method: `:histogram` (Unchanged)

Existing behavior preserved for backward compatibility. Uses `StatsBase.fit(Histogram, ...)` instead of hand-written loop, but output is identical.

## File Changes

| File | Change |
|---|---|
| `src/diagnostics/distributions.jl` | Add `method` parameter with dispatch to three code paths |
| `src/diagnostics/kde.jl` (new) | `kde_log_diameter()` wrapping KernelDensity.jl |
| `src/diagnostics/smooth_histogram.jl` (new) | `smooth_histogram_diameter()` with oversample + spline |
| `src/utils/binning.jl` | **Delete** — replaced by `StatsBase.fit(Histogram, ...)` |
| `Project.toml` | Add `KernelDensity`, `Interpolations` dependencies |
| `src/StochParticles.jl` | Update `include` and `export` statements |
| `test/test_plotting.jl` | Add tests for `:kde` and `:histogram_smooth` methods |

## Dependencies

```toml
[deps]
KernelDensity = "5ab10a40-..."    # KDE computation
Interpolations = "a2c5..."        # Linear + cubic spline interpolation
# StatsBase — transitive via KernelDensity, no explicit addition needed
```

## Testing Strategy

1. **Unit tests** for `kde_log_diameter` and `smooth_histogram_diameter` with known distributions
2. **Integration test**: run `compute_size_distribution` with all three methods on same simulation, verify:
   - `:kde` and `:histogram_smooth` produce smoother curves than `:histogram`
   - Total number concentration is conserved across methods (integral check)
3. **Edge cases**: few particles (< 10), single-species vs multi-species, very narrow distributions
4. **Visual regression**: example scripts produce plots without error

## Out of Scope

- dV/dlogD (volume distribution) — not requested
- Weighted KDE — equal weights sufficient for CNMC
- Plotting backend changes — stays with Plots.jl + GR
- New plot types — only smoothing algorithm changes
