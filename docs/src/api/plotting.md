# Plotting

StochParticles.jl provides built-in plotting functions built on [Plots.jl](https://docs.juliaplots.org/) for visualizing simulation results.

## Low-Level Functions

```@docs
plot_concentration_evolution
plot_size_distribution_heatmap
plot_kernel_contributions
```

## High-Level Summary

```@docs
plot_simulation_summary
```

### Method Selection

`plot_simulation_summary` forwards distribution-method parameters to [`compute_size_distribution`](@ref):

| Parameter | Default | Applies to | Description |
|-----------|---------|------------|-------------|
| `method` | `:kde` | all | `:histogram`, `:kde`, or `:histogram_smooth` |
| `bandwidth_factor` | `1.0` | `:kde` | Silverman bandwidth multiplier |
| `n_eval_points` | `200` | `:kde` | KDE evaluation grid size |
| `smooth_factor` | `3` | `:histogram_smooth` | histogram oversampling factor |

### Example

```julia
using StochParticles, Plots

# ... run simulation to get sol, prob ...

# Default: KDE-smoothed heatmap
pl = plot_simulation_summary(sol, prob, bin_edges, 1800.0;
                              time_unit="min", diameter_unit="μm")

# Raw histogram instead
pl = plot_simulation_summary(sol, prob, bin_edges, 1800.0;
                              method=:histogram)

# Adjust KDE bandwidth for sparser data
pl = plot_simulation_summary(sol, prob, bin_edges, 1800.0;
                              method=:kde, bandwidth_factor=0.5)
```

For the current end-to-end plotting workflow, run `julia --project=examples examples/simulate_single_component_coagulation.jl` followed by `python examples/plot_single_component_coagulation.py`.
