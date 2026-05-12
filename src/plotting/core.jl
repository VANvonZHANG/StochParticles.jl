# src/plotting/core.jl
using Plots

"""
    plot_concentration_evolution(t, N_conc, M_conc; time_unit="s", kwargs...)

Plot number concentration and normalized mass concentration vs time.

# Arguments
- `t`: time points
- `N_conc`: number concentration [m⁻³]
- `M_conc`: mass concentration [kg/m³]
- `time_unit`: unit for time axis label (default "s"), e.g. "min"
- `kwargs`: passed to `plot`

# Returns
- `Plots.Plot`: the concentration evolution plot
"""
function plot_concentration_evolution(
        t::Vector{Float64}, N_conc::Vector{Float64}, M_conc::Vector{Float64};
        time_unit::String = "s", kwargs...)
    # Normalize M_conc to same scale as N_conc for visualization
    M_norm = M_conc ./ maximum(M_conc) .* maximum(N_conc)

    pl = plot(t, N_conc;
        xlabel = "Time [$time_unit]",
        ylabel = "Number concentration [m⁻³]",
        label = "N(t)",
        linewidth = 2,
        legend = :topright,
        kwargs...
    )
    plot!(pl, t, M_norm;
        label = "M(t) (normalized)",
        linewidth = 2,
        linestyle = :dash
    )
    return pl
end

"""
    plot_size_distribution_heatmap(snapshot_times, bin_centers, dNdlogD_matrix;
                                    time_unit="s", diameter_unit="μm", kwargs...)

Plot size distribution evolution as a heatmap.

# Arguments
- `snapshot_times`: time points for snapshots
- `bin_centers`: bin center diameters [m]
- `dNdlogD_matrix`: matrix of shape (n_bins, n_snapshots)
- `time_unit`: unit for time axis (default "s")
- `diameter_unit`: unit for diameter axis (default "μm")
- `kwargs`: passed to `heatmap`

# Returns
- `Plots.Plot`: the heatmap plot
"""
function plot_size_distribution_heatmap(
        snapshot_times::Vector{Float64}, bin_centers::Vector{Float64},
        dNdlogD_matrix::Matrix{Float64};
        time_unit::String = "s", diameter_unit::String = "μm",
        kwargs...)
    # Convert diameter to display unit
    if diameter_unit == "μm"
        diam_display = bin_centers .* 1.0e6
    elseif diameter_unit == "nm"
        diam_display = bin_centers .* 1.0e9
    elseif diameter_unit == "m"
        diam_display = bin_centers
    else
        error("Unsupported diameter_unit: $diameter_unit")
    end

    pl = heatmap(snapshot_times, diam_display, dNdlogD_matrix;
        xlabel = "Time [$time_unit]",
        ylabel = "Diameter [$diameter_unit]",
        title = "Size Distribution Evolution",
        color = :viridis,
        yscale = :log10,
        colorbar_title = "dN/dlogD [m⁻³]",
        kwargs...
    )
    return pl
end

"""
    plot_kernel_contributions(labels, values; kwargs...)

Bar chart comparing coagulation kernel contributions.

# Arguments
- `labels`: vector of label strings
- `values`: vector of kernel values [m³/s]
- `kwargs`: passed to `bar`

# Returns
- `Plots.Plot`: the bar chart
"""
function plot_kernel_contributions(labels::Vector{String}, values::Vector{Float64}; kwargs...)
    pl = bar(labels, values;
        ylabel = "Coagulation rate [m³/s]",
        title = "Kernel Contributions",
        legend = false,
        yscale = :log10,
        kwargs...
    )
    return pl
end
