using StochParticles
using OrdinaryDiffEq
using Random
using StaticArrays
using Statistics

const MIXED_PDMP_BASENAME = "coupled_condensation_composite_coagulation_pdmp"
const DIFFEQCALLBACKS_PKGID = Base.PkgId(
    Base.UUID("459566f4-90b8-5000-8ac3-15dfb0a30def"), "DiffEqCallbacks")
const PLOTS_PKGID = Base.PkgId(
    Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots")

Base.@kwdef struct MixedPDMPConfig
    n_sim::Int = 240
    volume::Float64 = 2.4e-9
    tspan::Tuple{Float64, Float64} = (0.0, 120.0)
    saveat::Float64 = 4.0
    seed::Int = 20251205
    densities::SVector{2, Float64} = SVector(1770.0, 1000.0)
    hygroscopicity::SVector{2, Float64} = SVector(0.61, 0.0)
    h2o_idx::Int = 2
    accumulation_fraction::Float64 = 0.70
    accumulation_dg::Float64 = 120.0e-9
    accumulation_sigma::Float64 = 1.45
    giant_dg::Float64 = 1.0e-6
    giant_sigma::Float64 = 1.35
    T::Float64 = 288.15
    p::Float64 = 80_000.0
    supersaturation::Float64 = 0.003
    rho_f::Float64 = 1.06
    mu_f::Float64 = 1.75e-5
    nu::Float64 = 1.65e-5
    rho_p::Float64 = 1000.0
    g::Float64 = 9.81
    epsilon::Float64 = 0.01
    r_lambda::Float64 = 50.0
    activation_radius::Float64 = 1.0e-6
    csv_path::String = joinpath(@__DIR__, "$MIXED_PDMP_BASENAME.csv")
    figure_path::String = joinpath(@__DIR__, "$MIXED_PDMP_BASENAME.png")
end

struct KernelParts{B, G, T, C}
    brownian::B
    gravitational::G
    turbulent::T
    total::C
end

function make_thermo(cfg::MixedPDMPConfig)
    return ThermodynamicsParams(
        cfg.hygroscopicity,
        0.072,
        1000.0,
        18.015e-3,
        2.5e6,
        461.5,
        2.5e-5,
        2.4e-2
    )
end

function vapor_pressure(cfg::MixedPDMPConfig)
    saturation_vapor_pressure(cfg.T) * (1.0 + cfg.supersaturation)
end

gas_state_function(cfg::MixedPDMPConfig) = t -> SVector(cfg.T, vapor_pressure(cfg))

function make_cloud_params(cfg::MixedPDMPConfig)
    return AtmosphericParameters(
        cfg.T,
        cfg.p;
        rho_f = cfg.rho_f,
        mu_f = cfg.mu_f,
        nu = cfg.nu,
        rho_p = cfg.rho_p,
        g = cfg.g
    )
end

function build_initial_particles(cfg::MixedPDMPConfig)
    Random.seed!(cfg.seed)
    n_accum = clamp(round(Int, cfg.accumulation_fraction * cfg.n_sim), 0, cfg.n_sim)
    n_giant = cfg.n_sim - n_accum
    dry_fraction = SVector(1.0, 0.0)

    accum = lognormal_masses(
        n_accum, cfg.accumulation_dg, cfg.accumulation_sigma, cfg.densities;
        fractions = dry_fraction)
    giant = lognormal_masses(
        n_giant, cfg.giant_dg, cfg.giant_sigma, cfg.densities;
        fractions = dry_fraction)

    return SVector{2, Float64}.(vcat(accum, giant))
end

function build_kernel_parts(cfg::MixedPDMPConfig, params)
    brownian = BrownianKernel(params.T, params.p, cfg.densities)
    gravitational = GravitationalKernel(
        params.mu_f, params.rho_f, params.rho_p, params.g, cfg.densities)
    turbulent = AyalaTurbulentKernel(
        cfg.epsilon, cfg.r_lambda, params.nu, params.rho_f, params.rho_p,
        params.g, cfg.densities)
    total = CompositeKernel(brownian, CompositeKernel(gravitational, turbulent))
    return KernelParts{typeof(brownian), typeof(gravitational),
        typeof(turbulent), typeof(total)}(brownian, gravitational, turbulent, total)
end

function representative_pair_indices(diameters)
    n = length(diameters)
    n == 0 && return (0, 0)
    n == 1 && return (1, 1)

    order = sortperm(diameters)
    lo = clamp(ceil(Int, 0.25 * n), 1, n)
    hi = clamp(ceil(Int, 0.75 * n), 1, n)
    if lo == hi
        if hi < n
            hi += 1
        else
            lo -= 1
        end
    end
    return (order[lo], order[hi])
end

function _species_mass(u, sys, species_idx::Int)
    total = 0.0
    for i in 1:(sys.n_active)
        total += get_particle(u, i, Val(2))[species_idx]
    end
    return total
end

function _dry_diameters(u, sys, cfg::MixedPDMPConfig)
    diameters = Vector{Float64}(undef, sys.n_active)
    for i in 1:(sys.n_active)
        μ = get_particle(u, i, Val(2))
        dry_volume = μ[1] / cfg.densities[1]
        diameters[i] = (6.0 * dry_volume / π)^(1.0 / 3.0)
    end
    return diameters
end

function kernel_fraction_triplet(u, sys, _cfg::MixedPDMPConfig, kernel_parts)
    sys.n_active == 0 && return (0.0, 0.0, 0.0)
    sys.n_active == 1 && return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    brownian_sum = 0.0
    gravitational_sum = 0.0
    turbulent_sum = 0.0
    for i in 1:(sys.n_active - 1)
        μ_i = get_particle(u, i, Val(2))
        for j in (i + 1):(sys.n_active)
            μ_j = get_particle(u, j, Val(2))
            brownian_sum += max(kernel_parts.brownian(μ_i, μ_j), 0.0)
            gravitational_sum += max(kernel_parts.gravitational(μ_i, μ_j), 0.0)
            turbulent_sum += max(kernel_parts.turbulent(μ_i, μ_j), 0.0)
        end
    end

    total = brownian_sum + gravitational_sum + turbulent_sum

    if !(isfinite(total)) || total <= 0.0
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    end
    return (brownian_sum / total, gravitational_sum / total, turbulent_sum / total)
end

function evaluate_diagnostics(
        u, sys, cfg::MixedPDMPConfig, kernel_parts, _thermo, t)
    dry_diameters = _dry_diameters(u, sys, cfg)
    wet_diameters = particle_diameters(u, sys, cfg.densities)
    dry_mass = _species_mass(u, sys, 1)
    water_mass = _species_mass(u, sys, cfg.h2o_idx)
    brownian_frac, gravitational_frac, turbulent_frac = kernel_fraction_triplet(u, sys, cfg, kernel_parts)

    activation_frac = activation_fraction(
        u, sys, Val(2);
        mode = :radius_threshold,
        threshold = cfg.activation_radius,
        densities = cfg.densities
    )
    droplet_conc = cloud_droplet_concentration(
        u, sys, Val(2);
        mode = :radius_threshold,
        threshold = cfg.activation_radius,
        densities = cfg.densities
    )

    return (
        time = Float64(t),
        volume = sys.volume,
        active_particles = sys.n_active,
        number_concentration = number_concentration(sys),
        dry_mass_concentration = dry_mass / sys.volume,
        water_mass_concentration = water_mass / sys.volume,
        total_mass_concentration = total_mass(u, Val(2), sys.n_active) / sys.volume,
        mean_dry_diameter = isempty(dry_diameters) ? 0.0 : mean(dry_diameters),
        mean_wet_diameter = isempty(wet_diameters) ? 0.0 : mean(wet_diameters),
        median_wet_diameter = isempty(wet_diameters) ? 0.0 : median(wet_diameters),
        p90_wet_diameter = isempty(wet_diameters) ? 0.0 : quantile(wet_diameters, 0.90),
        activation_fraction = activation_frac,
        cloud_droplet_concentration = droplet_conc,
        brownian_fraction = brownian_frac,
        gravitational_fraction = gravitational_frac,
        turbulent_fraction = turbulent_frac
    )
end

function _load_diffeqcallbacks()
    try
        return Base.require(DIFFEQCALLBACKS_PKGID)
    catch err
        throw(ErrorException(
            "DiffEqCallbacks is required to run the mixed PDMP example. " *
            "Run with `julia --project=StochParticles.jl/examples " *
            "StochParticles.jl/examples/$MIXED_PDMP_BASENAME.jl`. " *
            "Original error: $err"))
    end
end

function build_mixed_pdmp_problem(cfg::MixedPDMPConfig)
    callbacks = _load_diffeqcallbacks()
    particles = build_initial_particles(cfg)
    thermo = make_thermo(cfg)
    gas_state = gas_state_function(cfg)
    params = make_cloud_params(cfg)
    kernel_parts = build_kernel_parts(cfg, params)

    condensation = H2OCondensationProcess(
        thermo, cfg.densities; h2o_idx = cfg.h2o_idx, w = 0.0)
    coagulation = CoagulationProcess(kernel_parts.total, GlobalMajorant())
    prob = ParticleProblem(
        particles, cfg.volume, gas_state, (condensation, coagulation);
        tspan = cfg.tspan, n_sim = cfg.n_sim)

    saved = callbacks.SavedValues(Float64, Any)
    save_func = (u, t, integrator) -> evaluate_diagnostics(
        u, integrator.p, cfg, kernel_parts, thermo, t)
    saving = callbacks.SavingCallback(
        save_func, saved; saveat = cfg.saveat, save_start = true, save_end = true)

    return (
        prob = prob,
        saved = saved,
        callback = saving,
        particles = particles,
        thermo = thermo,
        kernel_parts = kernel_parts,
        params = params
    )
end

rows_from_saved(saved) = collect(saved.saveval)

function write_diagnostics_csv(path, rows)
    isempty(rows) && return path
    header = collect(keys(rows[1]))
    open(path, "w") do io
        println(io, join(string.(header), ","))
        for row in rows
            println(io, join((string(getproperty(row, key)) for key in header), ","))
        end
    end
    return path
end

function spectrum_from_solution(sol, rows, cfg::MixedPDMPConfig; n_bins::Int = 56)
    isempty(rows) && return (
        times = Float64[],
        bin_edges = Float64[],
        bin_centers = Float64[],
        dNdlogD = zeros(Float64, 0, 0)
    )

    diameters_by_time = Vector{Vector{Float64}}(undef, length(rows))
    for (j, row) in enumerate(rows)
        idx = argmin(abs.(sol.t .- row.time))
        sys = ParticleSystem(Val(2), cfg.n_sim, row.volume, gas_state_function(cfg))
        sys.n_active = row.active_particles
        diameters_by_time[j] = particle_diameters(sol.u[idx], sys, cfg.densities)
    end

    all_diameters = reduce(vcat, diameters_by_time; init = Float64[])
    if isempty(all_diameters)
        return (
            times = [row.time for row in rows],
            bin_edges = Float64[],
            bin_centers = Float64[],
            dNdlogD = zeros(Float64, 0, length(rows))
        )
    end

    d_min = max(minimum(all_diameters) / 1.2, 1.0e-9)
    d_max = max(maximum(all_diameters) * 1.2, d_min * 10.0)
    bin_edges = collect(10.0 .^ range(log10(d_min), log10(d_max); length = n_bins + 1))
    bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])
    dlogD = diff(log10.(bin_edges))
    dNdlogD = zeros(Float64, n_bins, length(rows))

    for (j, row) in enumerate(rows)
        counts = bin_size_distribution(diameters_by_time[j], bin_edges)
        dNdlogD[:, j] = Float64.(counts) ./ dlogD ./ row.volume
    end

    return (
        times = [row.time for row in rows],
        bin_edges = bin_edges,
        bin_centers = bin_centers,
        dNdlogD = dNdlogD
    )
end

function _load_plots()
    try
        return Base.require(PLOTS_PKGID)
    catch err
        throw(ErrorException(
            "Plots is required to write the mixed PDMP summary figure. " *
            "Original error: $err"))
    end
end

function make_summary_figure(rows, spectrum, path)
    plots = _load_plots()
    isempty(rows) && return path

    times = [row.time for row in rows]
    number_conc = [row.number_concentration for row in rows]
    wet_mean = [row.mean_wet_diameter for row in rows]
    normalized_n = number_conc ./ number_conc[1]
    normalized_d = wet_mean ./ wet_mean[1]
    activation_pct = [100.0 * row.activation_fraction for row in rows]
    brown = [row.brownian_fraction for row in rows]
    grav = [row.gravitational_fraction for row in rows]
    turb = [row.turbulent_fraction for row in rows]

    p1n = plots.plot(
        times, normalized_n;
        seriestype = :steppost,
        ylabel = "N/N0",
        label = "N/N0 (step)",
        linewidth = 2,
        color = :black,
        title = "A. Number and wet size"
    )
    p1d = plots.plot(
        times, normalized_d;
        xlabel = "Time [s]",
        ylabel = "Dwet/Dwet0",
        label = "mean Dwet / Dwet0",
        linewidth = 2,
        color = :darkorange
    )
    p1 = plots.plot(p1n, p1d; layout = (2, 1))

    p2 = plots.plot(
        times, brown;
        xlabel = "Time [s]",
        ylabel = "Kernel fraction",
        label = "Brownian",
        linewidth = 2,
        color = :steelblue,
        ylim = (0, 1),
        title = "B. Pair-summed kernel fractions"
    )
    plots.plot!(
        p2, times, grav; label = "gravitational", linewidth = 2, color = :darkorange)
    plots.plot!(p2, times, turb; label = "turbulent", linewidth = 2, color = :seagreen)

    p3 = plots.heatmap(
        spectrum.times,
        spectrum.bin_centers .* 1.0e9,
        spectrum.dNdlogD;
        xlabel = "Time [s]",
        ylabel = "Wet diameter [nm]",
        colorbar = false,
        title = "C. Wet-diameter spectrum (dN/dlogD)"
    )
    plots.plot!(p3; yscale = :log10)

    p4 = plots.plot(
        times, activation_pct;
        xlabel = "Time [s]",
        ylabel = "Rwet >= 1 um [%]",
        label = "activation",
        linewidth = 2,
        color = :purple,
        ylim = (0, 100),
        title = "D. Threshold activation"
    )

    fig = plots.plot(
        p1, p2, p3, p4;
        layout = (2, 2),
        size = (1700, 1000),
        dpi = 150,
        margin = 6 * plots.mm
    )
    plots.savefig(fig, path)
    return path
end

function run_mixed_pdmp_demo(cfg::MixedPDMPConfig = MixedPDMPConfig())
    built = build_mixed_pdmp_problem(cfg)
    sol = solve(
        built.prob,
        Tsit5();
        callback = built.callback,
        saveat = cfg.saveat
    )
    rows = rows_from_saved(built.saved)
    spectrum = spectrum_from_solution(sol, rows, cfg)
    write_diagnostics_csv(cfg.csv_path, rows)
    make_summary_figure(rows, spectrum, cfg.figure_path)
    return (
        solution = sol,
        rows = rows,
        spectrum = spectrum,
        csv_path = cfg.csv_path,
        figure_path = cfg.figure_path,
        kernel_parts = built.kernel_parts,
        thermo = built.thermo
    )
end

function main()
    result = run_mixed_pdmp_demo()
    println("Mixed PDMP demonstration complete")
    println("Diagnostics: $(result.csv_path)")
    println("Figure: $(result.figure_path)")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
