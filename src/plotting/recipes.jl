# src/plotting/recipes.jl

"""
    plot_simulation_summary(sol, prob, bin_edges, rho; time_unit="s", diameter_unit="μm",
                             n_snapshots=30, method=:kde, bandwidth_factor=1.0,
                             n_eval_points=200, smooth_factor=3,
                             layout=(1,2), size=(1400,500), kwargs...)

Generate a combined summary plot for a particle simulation.

Left panel: concentration evolution (number + normalized mass)
Right panel: size distribution heatmap over time

# Arguments
- `sol`: SciML solution object
- `prob`: ParticleProblem (JumpProblem)
- `bin_edges`: diameter bin edges [m], strictly increasing
- `rho`: particle density [kg/m³]
- `time_unit`: unit for time axis (default "s"), e.g. "min" for aerosol plots
- `diameter_unit`: unit for diameter axis (default "μm"), e.g. "nm" for aerosol plots
- `n_snapshots`: number of time snapshots for size distribution (default 30)
- `method`: distribution method, `:histogram`, `:kde`, or `:histogram_smooth` (default `:kde`)
- `bandwidth_factor`: bandwidth multiplier for `:kde` method (default 1.0)
- `n_eval_points`: number of evaluation points for `:kde` method (default 200)
- `smooth_factor`: oversampling factor for `:histogram_smooth` method (default 3)
- `layout`: plot layout tuple (default (1,2))
- `size`: figure size in pixels (default (1400,500))
- `kwargs`: additional kwargs passed to the combined plot

# Returns
- `Plots.Plot`: the combined summary plot

# Example
```julia
bin_edges = 10.0 .^ range(-9, -5; length=26)  # 1 nm to 1 μm
pl = plot_simulation_summary(sol, prob, bin_edges, 1800.0;
                              time_unit="min", diameter_unit="μm")
savefig(pl, "simulation_summary.png")
```
"""
function plot_simulation_summary(sol, prob, bin_edges::Vector{Float64}, rho::Float64;
        time_unit::String = "s", diameter_unit::String = "μm",
        n_snapshots::Int = 30,
        method::Symbol = :kde,
        bandwidth_factor::Float64 = 1.0,
        n_eval_points::Int = 200,
        smooth_factor::Int = 3,
        layout::Tuple{Int, Int} = (1, 2),
        size::Tuple{Int, Int} = (1400, 500), kwargs...)
    # Extract concentrations
    t, N_conc, M_conc = extract_concentrations(sol, prob)

    # Convert time to display unit
    t_display = if time_unit == "min"
        t ./ 60.0
    elseif time_unit == "hr"
        t ./ 3600.0
    elseif time_unit == "ms"
        t .* 1000.0
    else
        t
    end

    # Left panel: concentration evolution
    pl1 = plot_concentration_evolution(t_display, N_conc, M_conc; time_unit = time_unit)

    # Right panel: size distribution heatmap
    snapshot_times, bin_centers,
    matrix = compute_size_distribution(
        sol, prob, bin_edges, rho;
        n_snapshots = n_snapshots,
        method = method,
        bandwidth_factor = bandwidth_factor,
        n_eval_points = n_eval_points,
        smooth_factor = smooth_factor)

    # Convert snapshot times to display unit
    snapshot_times_display = if time_unit == "min"
        snapshot_times ./ 60.0
    elseif time_unit == "hr"
        snapshot_times ./ 3600.0
    elseif time_unit == "ms"
        snapshot_times .* 1000.0
    else
        snapshot_times
    end

    pl2 = plot_size_distribution_heatmap(
        snapshot_times_display, bin_centers, matrix;
        time_unit = time_unit, diameter_unit = diameter_unit)

    # Combine
    return plot(pl1, pl2; layout = layout, size = size, kwargs...)
end
