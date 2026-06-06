# Diagnostics

## Concentration & Moments

```@docs
number_concentration
mass_concentration
species_mass_concentration
species_fractions
```

## Volume & Time Series

```@docs
reconstruct_volumes
extract_concentrations
check_mass_conservation
```

## Particle Diameters

```@docs
particle_diameters
```

## Size Distribution (dN/dlogD)

### High-Level Interface

```@docs
compute_size_distribution
```

### Distribution Methods

Three methods are available for computing dN/dlogD, selected via the `method` keyword:

| Method | Description | When to use |
|--------|-------------|-------------|
| `:kde` (default) | Gaussian kernel density estimation on log₁₀(diameter) | General purpose; produces smooth curves even with sparse data |
| `:histogram_smooth` | Oversampled histogram with linear interpolation | When you want bin-aligned output without KDE's extrapolation |
| `:histogram` | Raw histogram binning | Baseline comparison; no smoothing |

#### KDE Method

```@docs
kde_log_diameter
```

**Key parameters:**
- `bandwidth_factor`: multiplier for the Silverman bandwidth (default `1.0`). Smaller values produce narrower peaks; larger values oversmooth.
- `n_eval_points`: number of points on the KDE evaluation grid (default `200`).

The bandwidth is computed using Silverman's rule: `h = 0.9 × min(σ, IQR/1.34) × n⁻¹ᐟ⁵ × bandwidth_factor`.
If all particles have nearly identical diameters (spread < 10⁻⁶), a fallback bandwidth of `0.1 × bandwidth_factor` is used.

#### Smooth Histogram Method

```@docs
smooth_histogram_diameter
```

**Key parameter:**
- `smooth_factor`: number of sub-bins per original bin (default `3`). Higher values produce smoother output.

This method subdivides each bin into `smooth_factor` fine bins, histograms onto the fine grid, then averages back to the original bins. It avoids the cubic-spline oscillation that can occur with sparse, spiky data.

#### Raw Histogram Method

```@docs
bin_size_distribution
```

## Mixing State

```@docs
mixing_state_index
particle_mixing_entropy
shannon_entropy
```

## Cloud Activation

```@docs
activation_fraction
cloud_droplet_concentration
```
