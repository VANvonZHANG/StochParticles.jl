using HDF5
using Random
using StaticArrays
using StochParticles

include("simulation_io.jl")

const SINGLE_COMPONENT_BASENAME = "single_component_coagulation"
const A = 1
const EQUAL_KERNEL_FRACTIONS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
const SINGLE_COMPONENT_DEFAULT_N_SIM = 1000
const TIME_MAJOR_COMPAT_DATASETS = Set([
    "diameter_samples",
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

function overwrite_initial_diameter_datasets!(rep_group, diameter_initial, bin_edges, volume)
    diameters = Float64.(collect(diameter_initial))
    summary = diameter_summary(diameters)
    rep_group["mean_diameter"][1] = summary.mean
    rep_group["median_diameter"][1] = summary.median
    rep_group["p90_diameter"][1] = summary.p90
    rep_group["diameter_samples"][1, :] = diameters
    rep_group["size_distribution_raw"][1, :] = dNdlogD_from_diameters(diameters, bin_edges, volume)
    return nothing
end

function kernel_fraction_triplet(u, sys, parts)
    sys.n_active < 2 && return EQUAL_KERNEL_FRACTIONS

    brownian_sum = 0.0
    gravitational_sum = 0.0
    turbulent_sum = 0.0
    for i in 1:(sys.n_active - 1)
        μ_i = get_particle(u, i, Val(A))
        for j in (i + 1):(sys.n_active)
            μ_j = get_particle(u, j, Val(A))
            brownian_sum += max(parts.brownian(μ_i, μ_j), 0.0)
            gravitational_sum += max(parts.gravitational(μ_i, μ_j), 0.0)
            turbulent_sum += max(parts.turbulent(μ_i, μ_j), 0.0)
        end
    end

    total = brownian_sum + gravitational_sum + turbulent_sum
    if !isfinite(total) || total <= 0.0
        return EQUAL_KERNEL_FRACTIONS
    end

    return (brownian_sum / total, gravitational_sum / total, turbulent_sum / total)
end

function kernel_fraction_triplet(component_sums::NTuple{3, Float64})
    total = component_sums[1] + component_sums[2] + component_sums[3]
    if !isfinite(total) || total <= 0.0
        return EQUAL_KERNEL_FRACTIONS
    end
    return (component_sums[1] / total, component_sums[2] / total, component_sums[3] / total)
end

function initial_particles(n_sim, mode_small, sigma_small, mode_large, sigma_large, density)
    n_half = div(n_sim, 2)
    particles_small = lognormal_masses(n_half, mode_small, sigma_small, density)
    particles_large = lognormal_masses(n_sim - n_half, mode_large, sigma_large, density)
    return vcat(particles_small, particles_large)
end

function single_component_n_sim(default::Int = SINGLE_COMPONENT_DEFAULT_N_SIM)
    default >= 1 || throw(ArgumentError("default n_sim must be at least 1"))
    value = get(ENV, "STOCHPARTICLES_SINGLE_COMPONENT_N_SIM", string(default))
    n_sim = tryparse(Int, value)
    n_sim === nothing &&
        throw(ArgumentError("STOCHPARTICLES_SINGLE_COMPONENT_N_SIM must be an integer, got '$value'"))
    n_sim >= 2 ||
        throw(ArgumentError("STOCHPARTICLES_SINGLE_COMPONENT_N_SIM must be at least 2, got $n_sim"))
    return n_sim
end

function early_dense_then_regular_saveat(
        t_start, dense_end, t_end, dense_step, regular_step)
    dense_times = collect(t_start:dense_step:dense_end)
    regular_times = collect((dense_end + regular_step):regular_step:t_end)
    return vcat(dense_times, regular_times)
end

function case_attributes(cfg)
    return Dict{String, Any}(
        "species_names" => cfg.species_names,
        "density" => Float64(cfg.densities[1]),
        "n_sim" => cfg.n_sim,
        "volume" => cfg.volume,
        "coagulation_method" => "direct_ssa_non_cnmc",
        "t_start" => Float64(cfg.tspan[1]),
        "t_end" => Float64(cfg.tspan[2]),
        "saveat" => cfg.saveat,
        "notes" => cfg.notes
    )
end

function build_aerosol_case()
    params = standard_aerosol_atmosphere()
    densities = SVector(1800.0)
    n_sim = single_component_n_sim()
    return (
        name = "aerosol_brownian",
        species_names = ["SO4"],
        densities = densities,
        n_sim = n_sim,
        volume = 2.5e-12 * n_sim,
        tspan = (0.0, 3600.0),
        saveat = early_dense_then_regular_saveat(0.0, 720.0, 3600.0, 10.0, 60.0),
        initial_seed = 2026072100,
        seed_base = 2026072100,
        bin_edges = collect(10.0 .^ range(-9, -5; length = 81)),
        notes = "Single-component sulfate aerosol Brownian coagulation.",
        build_particles = seed -> begin
            Random.seed!(seed)
            initial_particles(n_sim, 5.0e-9, 1.5, 1.0e-7, 1.5, 1800.0)
        end,
        build_kernel = () -> BrownianKernel(params.T, params.p, densities),
        kernel_parts = nothing
    )
end

function build_cloud_case()
    params = standard_cloud_atmosphere()
    densities = SVector(1000.0)
    n_sim = single_component_n_sim()
    brownian = BrownianKernel(params.T, params.p, densities)
    gravitational = GravitationalKernel(
        params.mu_f, params.rho_f, params.rho_p, params.g, densities)
    turbulent = AyalaTurbulentKernel(
        0.01, 50.0, params.nu, params.rho_f, params.rho_p, params.g, densities)
    parts = (brownian = brownian, gravitational = gravitational, turbulent = turbulent)

    return (
        name = "cloud_composite",
        species_names = ["H2O"],
        densities = densities,
        n_sim = n_sim,
        volume = 5.0e-9 * n_sim,
        tspan = (0.0, 200.0),
        saveat = 4.0,
        initial_seed = 2026073100,
        seed_base = 2026073100,
        bin_edges = collect(10.0 .^ range(-6, -3; length = 81)),
        notes = "Single-component water cloud droplet composite coagulation.",
        build_particles = seed -> begin
            Random.seed!(seed)
            initial_particles(n_sim, 5.0e-6, 1.3, 2.5e-5, 1.3, 1000.0)
        end,
        build_kernel = () -> make_kernel(params, 0.01, 50.0, densities),
        kernel_parts = () -> parts
    )
end

function save_times_for_case(cfg)
    t_start, t_end = cfg.tspan
    if cfg.saveat isa Number
        times = collect(t_start:Float64(cfg.saveat):t_end)
        if isempty(times) || !isapprox(times[end], t_end; rtol = 0.0, atol = eps(Float64))
            push!(times, t_end)
        end
        return times
    end

    times = sort!(unique!(Float64.(collect(cfg.saveat))))
    if isempty(times) || !isapprox(times[1], t_start; rtol = 0.0, atol = eps(Float64))
        pushfirst!(times, t_start)
    end
    if !isapprox(times[end], t_end; rtol = 0.0, atol = eps(Float64))
        push!(times, t_end)
    end
    return times
end

function single_species_diameters(masses, active_particles, density)
    diameters = Vector{Float64}(undef, active_particles)
    for idx in 1:active_particles
        diameters[idx] = cbrt(6.0 * masses[idx] / (π * density))
    end
    return diameters
end

function coagulation_record(t, masses, active_particles, cfg, component_sums)
    diameters = single_species_diameters(masses, active_particles, cfg.densities[1])
    summary = diameter_summary(diameters)
    total_mass = sum(@view masses[1:active_particles]) / cfg.volume
    base = (
        time = Float64(t),
        volume = Float64(cfg.volume),
        active_particles = active_particles,
        number_concentration = active_particles / cfg.volume,
        total_mass_concentration = total_mass,
        species_mass_concentration = [total_mass],
        mean_diameter = summary.mean,
        median_diameter = summary.median,
        p90_diameter = summary.p90,
        diameter_samples = diameters,
        size_distribution_raw = dNdlogD_from_diameters(diameters, cfg.bin_edges, cfg.volume)
    )

    component_sums === nothing && return base
    brownian_frac, gravitational_frac,
    turbulent_frac = kernel_fraction_triplet(component_sums)
    return merge_record(base,
        (
            kernel_fraction_brownian = brownian_frac,
            kernel_fraction_gravitational = gravitational_frac,
            kernel_fraction_turbulent = turbulent_frac
        ))
end

function pair_kernel_value(kernel, mass_i, mass_j)
    return max(kernel(SVector{1, Float64}(mass_i), SVector{1, Float64}(mass_j)), 0.0)
end

function initialize_kernel_cache!(kernel_cache, masses, active_particles, kernel)
    fill!(kernel_cache, 0.0)
    total = 0.0
    for i in 1:(active_particles - 1)
        mass_i = masses[i]
        for j in (i + 1):active_particles
            value = pair_kernel_value(kernel, mass_i, masses[j])
            kernel_cache[i, j] = value
            kernel_cache[j, i] = value
            total += value
        end
    end
    return total
end

function initialize_component_caches!(component_caches, masses, active_particles, parts)
    component_sums = zeros(Float64, length(component_caches))
    kernels = (parts.brownian, parts.gravitational, parts.turbulent)
    for cache in component_caches
        fill!(cache, 0.0)
    end

    for i in 1:(active_particles - 1)
        mass_i = masses[i]
        for j in (i + 1):active_particles
            mass_j = masses[j]
            for component_idx in eachindex(component_caches)
                value = pair_kernel_value(kernels[component_idx], mass_i, mass_j)
                component_caches[component_idx][i, j] = value
                component_caches[component_idx][j, i] = value
                component_sums[component_idx] += value
            end
        end
    end
    return (component_sums[1], component_sums[2], component_sums[3])
end

function sum_upper(cache, active_particles)
    total = 0.0
    for i in 1:(active_particles - 1)
        for j in (i + 1):active_particles
            total += cache[i, j]
        end
    end
    return total
end

function component_sums_from_caches(component_caches, active_particles)
    return (
        sum_upper(component_caches[1], active_particles),
        sum_upper(component_caches[2], active_particles),
        sum_upper(component_caches[3], active_particles)
    )
end

function refresh_cache_index!(cache, masses, active_particles, kernel, idx)
    if idx < 1 || idx > active_particles
        return nothing
    end
    for k in 1:active_particles
        if k == idx
            cache[idx, idx] = 0.0
        else
            value = pair_kernel_value(kernel, masses[idx], masses[k])
            cache[idx, k] = value
            cache[k, idx] = value
        end
    end
    return nothing
end

function refresh_component_cache_index!(
        component_caches, masses, active_particles, parts, idx)
    kernels = (parts.brownian, parts.gravitational, parts.turbulent)
    for component_idx in eachindex(component_caches)
        refresh_cache_index!(
            component_caches[component_idx], masses, active_particles, kernels[component_idx], idx)
    end
    return nothing
end

function zero_cache_index!(cache, idx)
    cache[idx, :] .= 0.0
    cache[:, idx] .= 0.0
    return nothing
end

function sample_pair_from_cache(cache, active_particles, total_kernel_sum)
    target = rand() * total_kernel_sum
    cumulative = 0.0
    for i in 1:(active_particles - 1)
        for j in (i + 1):active_particles
            cumulative += cache[i, j]
            if cumulative >= target
                return i, j
            end
        end
    end
    return active_particles - 1, active_particles
end

function update_after_coagulation!(
        kernel_cache, component_caches, masses, active_particles, kernel, parts, i, j)
    masses[i] += masses[j]
    if j < active_particles
        masses[j] = masses[active_particles]
    end
    masses[active_particles] = 0.0

    new_active_particles = active_particles - 1
    zero_cache_index!(kernel_cache, active_particles)
    if component_caches !== nothing
        for cache in component_caches
            zero_cache_index!(cache, active_particles)
        end
    end

    refresh_cache_index!(kernel_cache, masses, new_active_particles, kernel, i)
    if j <= new_active_particles
        refresh_cache_index!(kernel_cache, masses, new_active_particles, kernel, j)
    end

    if component_caches !== nothing
        refresh_component_cache_index!(
            component_caches, masses, new_active_particles, parts, i)
        if j <= new_active_particles
            refresh_component_cache_index!(
                component_caches, masses, new_active_particles, parts, j)
        end
    end

    return new_active_particles
end

function direct_coagulation_records(cfg, initial_masses)
    masses = [Float64(μ[1]) for μ in initial_masses]
    active_particles = length(masses)
    active_particles == cfg.n_sim ||
        throw(DimensionMismatch("initial particles length $(length(masses)) must equal n_sim $(cfg.n_sim)"))

    kernel = cfg.build_kernel()
    parts = cfg.kernel_parts === nothing ? nothing : cfg.kernel_parts()
    kernel_cache = zeros(Float64, cfg.n_sim, cfg.n_sim)
    total_kernel_sum = initialize_kernel_cache!(kernel_cache, masses, active_particles, kernel)

    component_caches = nothing
    component_sums = nothing
    if parts !== nothing
        component_caches = [zeros(Float64, cfg.n_sim, cfg.n_sim) for _ in 1:3]
        component_sums = initialize_component_caches!(
            component_caches, masses, active_particles, parts)
    end

    save_times = save_times_for_case(cfg)
    records = Vector{Any}(undef, length(save_times))
    current_time = Float64(cfg.tspan[1])

    for save_idx in eachindex(save_times)
        save_time = save_times[save_idx]
        while active_particles >= 2 && total_kernel_sum > 0.0
            total_rate = total_kernel_sum / cfg.volume
            next_time = current_time - log(rand()) / total_rate
            next_time > save_time && break

            current_time = next_time
            i, j = sample_pair_from_cache(kernel_cache, active_particles, total_kernel_sum)
            active_particles = update_after_coagulation!(
                kernel_cache,
                component_caches,
                masses,
                active_particles,
                kernel,
                parts,
                i,
                j
            )
            total_kernel_sum = sum_upper(kernel_cache, active_particles)
            if component_caches !== nothing
                component_sums = component_sums_from_caches(component_caches, active_particles)
            end
        end
        current_time = save_time
        records[save_idx] = coagulation_record(
            save_time, masses, active_particles, cfg, component_sums)
    end

    return records
end

function run_replicate!(case_group, cfg, replicate_idx)
    process_seed = cfg.seed_base + replicate_idx
    particles = cfg.build_particles(cfg.initial_seed)
    Random.seed!(process_seed)
    dry_diameter_initial = diameters_from_masses(particles, cfg.densities)
    records = direct_coagulation_records(cfg, particles)

    rep_group = create_replicate_group(case_group, replicate_idx)
    seed_attrs = Dict{String, Any}(
        "initial_seed" => cfg.initial_seed,
        "process_seed" => process_seed,
        "seed" => process_seed
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
    overwrite_initial_diameter_datasets!(
        rep_group, dry_diameter_initial, cfg.bin_edges, cfg.volume)
    write_datasets_compat_group!(rep_group)
    return nothing
end

function run_case!(file, cfg, n_replicates)
    case_group = ensure_case_group(file, cfg.name; attrs_dict = case_attributes(cfg))
    for replicate_idx in 1:n_replicates
        run_replicate!(case_group, cfg, replicate_idx)
    end
    return nothing
end

function main()
    n_replicates = example_replicates()
    h5_path = joinpath(example_data_dir(), SINGLE_COMPONENT_BASENAME * ".h5")
    recreate_h5(
        h5_path;
        scene_name = SINGLE_COMPONENT_BASENAME,
        n_replicates = n_replicates,
        notes = "Single-component aerosol Brownian and cloud composite coagulation simulations."
    )

    cases = (build_aerosol_case(), build_cloud_case())
    h5open(h5_path, "r+") do file
        for cfg in cases
            run_case!(file, cfg, n_replicates)
        end
    end

    println("Wrote single-component coagulation simulations to $h5_path")
    return h5_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
