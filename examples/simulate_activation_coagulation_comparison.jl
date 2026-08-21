using HDF5
using OrdinaryDiffEq
using Random
using StaticArrays
using StochParticles

include("simulation_io.jl")

const ACTIVATION_BASENAME = "activation_coagulation_comparison"
const A = 2
const EQUAL_KERNEL_FRACTIONS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
const TIME_MAJOR_COMPAT_DATASETS = Set([
    "activation_flag_samples",
    "diameter_samples",
    "dry_diameter_samples",
    "size_distribution_raw",
    "species_mass_concentration"
])

function write_datasets_compat_group!(rep_group)
    datasets_group = create_group(rep_group, "datasets")
    for name in collect(keys(rep_group))
        name == "datasets" && continue
        obj = rep_group[name]
        if obj isa HDF5.Dataset
            data = read(obj)
            datasets_group[name] = name in TIME_MAJOR_COMPAT_DATASETS && ndims(data) == 2 ?
                                   permutedims(data) : data
        end
    end
    return datasets_group
end

function overwrite_initial_diameter_datasets!(
        rep_group, diameter_initial, bin_edges, volume; dry_diameter_initial = nothing)
    diameters = Float64.(collect(diameter_initial))
    summary = diameter_summary(diameters)
    rep_group["mean_diameter"][1] = summary.mean
    rep_group["median_diameter"][1] = summary.median
    rep_group["p90_diameter"][1] = summary.p90
    rep_group["diameter_samples"][1, :] = diameters
    rep_group["size_distribution_raw"][1, :] = dNdlogD_from_diameters(diameters, bin_edges, volume)
    if dry_diameter_initial !== nothing && haskey(rep_group, "dry_diameter_samples")
        rep_group["dry_diameter_samples"][1, :] = Float64.(collect(dry_diameter_initial))
    end
    return nothing
end

Base.@kwdef struct ActivationComparisonConfig
    species_names::Vector{String} = ["SO4", "H2O"]
    n_sim::Int = 1000
    aitken_dg::Float64 = 2.0e-8
    aitken_sigma_g::Float64 = 1.25
    aitken_concentration::Float64 = 8.4e11
    accumulation_dg::Float64 = 2.0e-7
    accumulation_sigma_g::Float64 = 1.4
    accumulation_concentration::Float64 = 2.1e11
    volume::Float64 = 1000.0 / (8.4e11 + 2.1e11)
    tspan::Tuple{Float64, Float64} = (0.0, 600.0)
    saveat::Float64 = 10.0
    densities::SVector{2, Float64} = SVector(1770.0, 1000.0)
    h2o_idx::Int = 2
    T::Float64 = 293.15
    p::Float64 = 1.01325e5
    supersaturation::Float64 = 0.005
    activation_radius::Float64 = 1.0e-6
    mode_dry_diameter_threshold::Float64 = 6.0e-8
    rho_f::Float64 = 1.06
    mu_f::Float64 = 1.75e-5
    nu::Float64 = 1.65e-5
    rho_p::Float64 = 1000.0
    g::Float64 = 9.81
    epsilon::Float64 = 0.01
    r_lambda::Float64 = 50.0
    bin_edges::Vector{Float64} = collect(10.0 .^ range(-8.3, -4.5; length = 96))
    initial_seed::Int = 2026072200
    seed_base::Int = 2026072200
end

function mode_particle_counts(cfg::ActivationComparisonConfig)
    total = cfg.aitken_concentration + cfg.accumulation_concentration
    n_aitken = round(Int, cfg.n_sim * cfg.aitken_concentration / total)
    return n_aitken, cfg.n_sim - n_aitken
end

function average_thermo(_cfg::ActivationComparisonConfig)
    return ThermodynamicsParams(
        SVector(0.455, 0.0),
        0.072,
        1000.0,
        18.015e-3,
        2.5e6,
        461.5,
        2.5e-5,
        2.4e-2
    )
end

function vapor_pressure(cfg::ActivationComparisonConfig)
    saturation_vapor_pressure(cfg.T) * (1.0 + cfg.supersaturation)
end

gas_fn(cfg::ActivationComparisonConfig) = _t -> SVector(cfg.T, vapor_pressure(cfg))

function dry_diameter_from_so4_mass(mass::Real, density::Real)
    return (6.0 * max(Float64(mass), 0.0) / (pi * Float64(density)))^(1.0 / 3.0)
end

function initial_activation_particles(cfg::ActivationComparisonConfig, seed)
    Random.seed!(seed)
    n_aitken, n_accum = mode_particle_counts(cfg)
    particles_aitken = lognormal_masses(
        n_aitken, cfg.aitken_dg, cfg.aitken_sigma_g, cfg.densities)
    particles_accum = lognormal_masses(
        n_accum, cfg.accumulation_dg, cfg.accumulation_sigma_g, cfg.densities)
    dry_particles = [SVector{2, Float64}(m[1], 0.0)
                     for m in vcat(particles_aitken, particles_accum)]
    dry_diams = [dry_diameter_from_so4_mass(p[1], cfg.densities[1])
                 for p in dry_particles]
    thermo_labels = fill(0.455, cfg.n_sim)
    particles = deepcopy(dry_particles)
    pre_equilibrate!(
        particles,
        average_thermo(cfg),
        cfg.densities,
        cfg.T,
        vapor_pressure(cfg);
        h2o_idx = cfg.h2o_idx
    )
    return (particles, dry_diams, thermo_labels)
end

function cloud_params(cfg::ActivationComparisonConfig)
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

function activation_kernel_parts(cfg::ActivationComparisonConfig)
    params = cloud_params(cfg)
    brownian = BrownianKernel(params.T, params.p, cfg.densities)
    gravitational = GravitationalKernel(
        params.mu_f, params.rho_f, params.rho_p, params.g, cfg.densities)
    turbulent = AyalaTurbulentKernel(
        cfg.epsilon,
        cfg.r_lambda,
        params.nu,
        params.rho_f,
        params.rho_p,
        params.g,
        cfg.densities
    )
    total = make_kernel(params, cfg.epsilon, cfg.r_lambda, cfg.densities)
    return (
        brownian = brownian,
        gravitational = gravitational,
        turbulent = turbulent,
        total = total
    )
end

function dry_diameters_from_state(u, sys, cfg::ActivationComparisonConfig)
    diameters = Vector{Float64}(undef, sys.n_active)
    for i in 1:sys.n_active
        particle = get_particle(u, i, Val(A))
        diameters[i] = dry_diameter_from_so4_mass(particle[1], cfg.densities[1])
    end
    return diameters
end

function activation_flags_from_state(u, sys, cfg::ActivationComparisonConfig)
    flags = Vector{Float64}(undef, sys.n_active)
    for i in 1:sys.n_active
        particle = get_particle(u, i, Val(A))
        wet_volume = 0.0
        for k in 1:A
            wet_volume += max(particle[k], 0.0) / cfg.densities[k]
        end
        wet_radius = cbrt(3.0 * wet_volume / (4.0 * pi))
        flags[i] = wet_radius >= cfg.activation_radius ? 1.0 : 0.0
    end
    return flags
end

function kernel_fraction_triplet(u, sys, parts)
    sys.n_active < 2 && return EQUAL_KERNEL_FRACTIONS

    brownian_sum = 0.0
    gravitational_sum = 0.0
    turbulent_sum = 0.0
    for i in 1:(sys.n_active - 1)
        particle_i = get_particle(u, i, Val(A))
        for j in (i + 1):sys.n_active
            particle_j = get_particle(u, j, Val(A))
            brownian_sum += max(parts.brownian(particle_i, particle_j), 0.0)
            gravitational_sum += max(parts.gravitational(particle_i, particle_j), 0.0)
            turbulent_sum += max(parts.turbulent(particle_i, particle_j), 0.0)
        end
    end

    total = brownian_sum + gravitational_sum + turbulent_sum
    if !isfinite(total) || total <= 0.0
        return EQUAL_KERNEL_FRACTIONS
    end

    return (brownian_sum / total, gravitational_sum / total, turbulent_sum / total)
end

function record_extras(u, sys, cfg::ActivationComparisonConfig)
    return (
        activation_fraction = activation_fraction(
            u,
            sys,
            Val(A);
            mode = :radius_threshold,
            threshold = cfg.activation_radius,
            densities = cfg.densities
        ),
        cloud_droplet_concentration = cloud_droplet_concentration(
            u,
            sys,
            Val(A);
            mode = :radius_threshold,
            threshold = cfg.activation_radius,
            densities = cfg.densities
        ),
        dry_diameter_samples = dry_diameters_from_state(u, sys, cfg),
        activation_flag_samples = activation_flags_from_state(u, sys, cfg)
    )
end

function solve_activation_scenario(cfg::ActivationComparisonConfig, particles, scenario)
    thermo = average_thermo(cfg)
    condensation = H2OCondensationProcess(
        thermo, cfg.densities; h2o_idx = cfg.h2o_idx, w = 0.0)

    processes = if scenario == :activation_only
        (condensation,)
    elseif scenario == :activation_with_coagulation
        parts = activation_kernel_parts(cfg)
        coagulation = CoagulationProcess(parts.total, GlobalMajorant())
        (condensation, coagulation)
    else
        throw(ArgumentError("unknown activation scenario '$scenario'"))
    end

    prob = ParticleProblem(
        deepcopy(particles),
        cfg.volume,
        gas_fn(cfg),
        processes;
        tspan = cfg.tspan,
        n_sim = cfg.n_sim
    )

    parts = scenario == :activation_with_coagulation ? activation_kernel_parts(cfg) :
            nothing
    record_func = (t, u, sys) -> begin
        base = base_diagnostic_record(t, u, sys, Val(A), cfg.densities, cfg.bin_edges)
        extras = record_extras(u, sys, cfg)
        if parts === nothing
            return merge_record(base, extras)
        end

        brownian_frac, gravitational_frac, turbulent_frac = kernel_fraction_triplet(u, sys, parts)
        return merge_record(
            base,
            merge(
                extras,
                (
                    kernel_fraction_brownian = brownian_frac,
                    kernel_fraction_gravitational = gravitational_frac,
                    kernel_fraction_turbulent = turbulent_frac
                )
            )
        )
    end

    return solve_with_records(prob, Tsit5(); saveat = cfg.saveat, record_func = record_func)
end

function case_attributes(cfg::ActivationComparisonConfig, scenario)
    notes = scenario == :activation_only ?
            "Fixed-supersaturation activation with H2O condensation only." :
            "Fixed-supersaturation activation with H2O condensation and composite coagulation."
    return Dict{String, Any}(
        "species_names" => cfg.species_names,
        "n_sim" => cfg.n_sim,
        "volume" => cfg.volume,
        "t_start" => Float64(cfg.tspan[1]),
        "t_end" => Float64(cfg.tspan[2]),
        "saveat" => cfg.saveat,
        "supersaturation" => cfg.supersaturation,
        "activation_radius" => cfg.activation_radius,
        "notes" => notes
    )
end

function write_scenario_replicate!(
        case_group, cfg::ActivationComparisonConfig, replicate_idx, seed,
        initial_seed, particles, dry_diameter_initial, thermo_labels, scenario)
    scenario_particles = deepcopy(particles)
    diameter_initial = diameters_from_masses(scenario_particles, cfg.densities)
    Random.seed!(seed)
    sol, records = solve_activation_scenario(cfg, scenario_particles, scenario)
    @assert sol.retcode == ReturnCode.Success

    rep_group = create_replicate_group(case_group, replicate_idx)
    seed_attrs = Dict{String, Any}(
        "initial_seed" => initial_seed,
        "process_seed" => seed,
        "seed" => seed
    )
    _write_attrs!(rep_group, seed_attrs)
    write_records_common!(
        rep_group,
        records,
        cfg.n_sim,
        cfg.bin_edges;
        dry_diameter_initial = dry_diameter_initial,
        extra_attrs = seed_attrs
    )
    write_vector(rep_group, "thermo_kappa_initial", thermo_labels)
    overwrite_initial_diameter_datasets!(
        rep_group,
        diameter_initial,
        cfg.bin_edges,
        cfg.volume;
        dry_diameter_initial = dry_diameter_initial
    )
    write_datasets_compat_group!(rep_group)
    return nothing
end

function run_replicate!(case_groups, cfg::ActivationComparisonConfig, replicate_idx)
    process_seed = cfg.seed_base + replicate_idx
    particles, dry_diameter_initial, thermo_labels = initial_activation_particles(cfg, cfg.initial_seed)

    write_scenario_replicate!(
        case_groups.activation_only,
        cfg,
        replicate_idx,
        process_seed,
        cfg.initial_seed,
        particles,
        dry_diameter_initial,
        thermo_labels,
        :activation_only
    )
    write_scenario_replicate!(
        case_groups.activation_with_coagulation,
        cfg,
        replicate_idx,
        process_seed,
        cfg.initial_seed,
        particles,
        dry_diameter_initial,
        thermo_labels,
        :activation_with_coagulation
    )
    return nothing
end

function main()
    cfg = ActivationComparisonConfig()
    n_replicates = example_replicates()
    h5_path = joinpath(example_data_dir(), ACTIVATION_BASENAME * ".h5")
    recreate_h5(
        h5_path;
        scene_name = ACTIVATION_BASENAME,
        n_replicates = n_replicates,
        notes = "Paired activation-only and activation-with-coagulation simulations."
    )

    h5open(h5_path, "r+") do file
        case_groups = (
            activation_only = ensure_case_group(
                file,
                "activation_only";
                attrs_dict = case_attributes(cfg, :activation_only)
            ),
            activation_with_coagulation = ensure_case_group(
                file,
                "activation_with_coagulation";
                attrs_dict = case_attributes(cfg, :activation_with_coagulation)
            )
        )
        for replicate_idx in 1:n_replicates
            run_replicate!(case_groups, cfg, replicate_idx)
        end
    end

    println("Wrote activation coagulation comparison simulations to $h5_path")
    return h5_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
