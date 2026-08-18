using HDF5
using Random
using StaticArrays
using StochParticles

include("simulation_io.jl")

const MIXING_BASENAME = "mixing_state_coagulation"
const A = 2
const TIME_MAJOR_COMPAT_DATASETS = Set([
    "bc_mass_fraction_samples",
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

Base.@kwdef struct MixingStateCoagulationConfig
    species_names::Vector{String} = ["SO4", "BC"]
    n_sim::Int = 4000
    volume::Float64 = 1.0e-7
    tspan::Tuple{Float64, Float64} = (0.0, 24.0 * 3600.0)
    saveat::Vector{Float64} = early_dense_then_regular_saveat(
        0.0, 2.0 * 3600.0, 24.0 * 3600.0, 10.0 * 60.0, 60.0 * 60.0)
    T::Float64 = 298.0
    p::Float64 = 1.01325e5
    densities::SVector{2, Float64} = SVector(1770.0, 1800.0)
    bin_edges::Vector{Float64} = collect(10.0 .^ range(-9, -6; length = 81))
    initial_seed::Int = 2026092100
    seed_base::Int = 2026092100
    notes::String = "SO4/BC external-to-internal mixing by Brownian coagulation."
end

function early_dense_then_regular_saveat(
        t_start, dense_end, t_end, dense_step, regular_step)
    dense_times = collect(t_start:dense_step:dense_end)
    regular_times = collect((dense_end + regular_step):regular_step:t_end)
    return vcat(dense_times, regular_times)
end

function save_times_for_case(cfg::MixingStateCoagulationConfig)
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

function initial_mixing_particles(cfg::MixingStateCoagulationConfig, seed)
    Random.seed!(seed)

    n_so4 = div(cfg.n_sim, 2)
    n_bc = cfg.n_sim - n_so4

    so4_particles = lognormal_masses(
        n_so4,
        1.0e-7,
        1.5,
        cfg.densities;
        fractions = SVector(1.0, 0.0)
    )
    bc_particles = lognormal_masses(
        n_bc,
        5.0e-8,
        1.3,
        cfg.densities;
        fractions = SVector(0.0, 1.0)
    )
    return vcat(so4_particles, bc_particles)
end

function bc_mass_fraction_samples(particles, active_particles)
    fractions = Vector{Float64}(undef, active_particles)
    for particle_idx in 1:active_particles
        μ = particles[particle_idx]
        total = sum(μ)
        fractions[particle_idx] = total > 0.0 ? μ[2] / total : 0.0
    end
    return fractions
end

function active_particle_diameters(particles, active_particles, densities)
    diameters = Vector{Float64}(undef, active_particles)
    for particle_idx in 1:active_particles
        μ = particles[particle_idx]
        volume = 0.0
        for species_idx in 1:A
            volume += μ[species_idx] / densities[species_idx]
        end
        diameters[particle_idx] = cbrt(6.0 * volume / π)
    end
    return diameters
end

function direct_mixing_state_index(particles, active_particles)
    active_particles > 0 || return NaN

    m_total = 0.0
    f_bar = zero(SVector{A, Float64})
    entropy_sum = 0.0
    for particle_idx in 1:active_particles
        μ = particles[particle_idx]
        fractions = species_fractions(μ)
        entropy_sum += shannon_entropy(fractions)
        m_i = sum(μ)
        m_total += m_i
        f_bar += fractions * m_i
    end
    m_total > 0.0 || return NaN
    f_bar /= m_total

    D_eps = entropy_sum / active_particles
    D_gamma = shannon_entropy(f_bar)
    chi = D_gamma > 0.0 ? D_eps / D_gamma : 0.0
    return chi == 0.0 ? 0.0 : chi
end

function mixing_state_record(t, particles, active_particles, cfg::MixingStateCoagulationConfig)
    diameters = active_particle_diameters(particles, active_particles, cfg.densities)
    summary = diameter_summary(diameters)

    species_mass = zeros(Float64, A)
    for particle_idx in 1:active_particles
        μ = particles[particle_idx]
        for species_idx in 1:A
            species_mass[species_idx] += μ[species_idx]
        end
    end
    species_mass ./= cfg.volume

    return (
        time = Float64(t),
        volume = Float64(cfg.volume),
        active_particles = active_particles,
        number_concentration = active_particles / cfg.volume,
        total_mass_concentration = sum(species_mass),
        species_mass_concentration = species_mass,
        mean_diameter = summary.mean,
        median_diameter = summary.median,
        p90_diameter = summary.p90,
        diameter_samples = diameters,
        size_distribution_raw = dNdlogD_from_diameters(diameters, cfg.bin_edges, cfg.volume),
        mixing_state_index = direct_mixing_state_index(particles, active_particles),
        bc_mass_fraction_samples = bc_mass_fraction_samples(particles, active_particles)
    )
end

function pair_kernel_value(kernel, μ_i, μ_j)
    return max(kernel(μ_i, μ_j), 0.0)
end

function initialize_kernel_cache!(
        kernel_cache, row_sums, particles, active_particles, kernel)
    fill!(kernel_cache, 0.0)
    fill!(row_sums, 0.0)
    total = 0.0
    for i in 1:(active_particles - 1)
        μ_i = particles[i]
        for j in (i + 1):active_particles
            value = pair_kernel_value(kernel, μ_i, particles[j])
            kernel_cache[i, j] = value
            kernel_cache[j, i] = value
            row_sums[i] += value
            row_sums[j] += value
            total += value
        end
    end
    return total
end

function refresh_cache_index!(
        kernel_cache, row_sums, particles, active_particles, kernel, idx)
    if idx < 1 || idx > active_particles
        return nothing
    end
    for k in 1:active_particles
        if k == idx
            kernel_cache[idx, idx] = 0.0
        else
            old_value = kernel_cache[idx, k]
            new_value = pair_kernel_value(kernel, particles[idx], particles[k])
            delta = new_value - old_value
            kernel_cache[idx, k] = new_value
            kernel_cache[k, idx] = new_value
            row_sums[idx] += delta
            row_sums[k] += delta
        end
    end
    row_sums[idx] = max(row_sums[idx], 0.0)
    return nothing
end

function zero_cache_index!(kernel_cache, row_sums, active_particles, idx)
    for k in 1:active_particles
        k == idx && continue
        old_value = kernel_cache[idx, k]
        kernel_cache[idx, k] = 0.0
        kernel_cache[k, idx] = 0.0
        row_sums[k] = max(row_sums[k] - old_value, 0.0)
    end
    row_sums[idx] = 0.0
    return nothing
end

function sample_pair_from_cache(kernel_cache, row_sums, active_particles, total_kernel_sum)
    row_target = rand() * 2.0 * total_kernel_sum
    row_cumulative = 0.0
    i = active_particles
    for row_idx in 1:active_particles
        row_cumulative += row_sums[row_idx]
        if row_cumulative >= row_target
            i = row_idx
            break
        end
    end

    partner_target = rand() * row_sums[i]
    partner_cumulative = 0.0
    for j in 1:active_particles
        j == i && continue
        partner_cumulative += kernel_cache[i, j]
        if partner_cumulative >= partner_target
            return i < j ? (i, j) : (j, i)
        end
    end

    fallback_j = i == active_particles ? active_particles - 1 : active_particles
    return i < fallback_j ? (i, fallback_j) : (fallback_j, i)
end

function update_after_coagulation!(
        kernel_cache, row_sums, particles, active_particles, kernel, i, j)
    particles[i] += particles[j]
    if j < active_particles
        particles[j] = particles[active_particles]
    end
    particles[active_particles] = zero(particles[i])

    zero_cache_index!(kernel_cache, row_sums, active_particles, active_particles)
    new_active_particles = active_particles - 1

    refresh_cache_index!(kernel_cache, row_sums, particles, new_active_particles, kernel, i)
    if j <= new_active_particles
        refresh_cache_index!(
            kernel_cache, row_sums, particles, new_active_particles, kernel, j)
    end

    return new_active_particles
end

function direct_mixing_state_coagulation_records(cfg, initial_particles)
    particles = copy(initial_particles)
    active_particles = length(particles)
    active_particles == cfg.n_sim ||
        throw(DimensionMismatch(
            "initial particles length $(length(particles)) must equal n_sim $(cfg.n_sim)"))

    kernel = BrownianKernel(cfg.T, cfg.p, cfg.densities)
    kernel_cache = zeros(Float64, cfg.n_sim, cfg.n_sim)
    row_sums = zeros(Float64, cfg.n_sim)
    total_kernel_sum = initialize_kernel_cache!(
        kernel_cache, row_sums, particles, active_particles, kernel)

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
            i, j = sample_pair_from_cache(
                kernel_cache, row_sums, active_particles, total_kernel_sum)
            active_particles = update_after_coagulation!(
                kernel_cache, row_sums, particles, active_particles, kernel, i, j)
            total_kernel_sum = 0.5 * sum(@view row_sums[1:active_particles])
        end
        current_time = save_time
        records[save_idx] = mixing_state_record(save_time, particles, active_particles, cfg)
    end

    return records
end

function case_attributes(cfg::MixingStateCoagulationConfig)
    return Dict{String, Any}(
        "species_names" => cfg.species_names,
        "n_sim" => cfg.n_sim,
        "volume" => cfg.volume,
        "coagulation_method" => "direct_ssa_kernel_cache_non_cnmc",
        "t_start" => Float64(cfg.tspan[1]),
        "t_end" => Float64(cfg.tspan[2]),
        "saveat" => cfg.saveat,
        "notes" => cfg.notes
    )
end

function run_replicate!(case_group, cfg::MixingStateCoagulationConfig, replicate_idx)
    process_seed = cfg.seed_base + replicate_idx
    particles = initial_mixing_particles(cfg, cfg.initial_seed)
    Random.seed!(process_seed)
    dry_diameter_initial = diameters_from_masses(particles, cfg.densities)
    records = direct_mixing_state_coagulation_records(cfg, particles)

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

function main()
    cfg = MixingStateCoagulationConfig()
    n_replicates = example_replicates()
    h5_path = joinpath(example_data_dir(), MIXING_BASENAME * ".h5")
    recreate_h5(
        h5_path;
        scene_name = MIXING_BASENAME,
        n_replicates = n_replicates,
        notes = cfg.notes
    )

    h5open(h5_path, "r+") do file
        case_group = ensure_case_group(
            file,
            MIXING_BASENAME;
            attrs_dict = case_attributes(cfg)
        )
        for replicate_idx in 1:n_replicates
            run_replicate!(case_group, cfg, replicate_idx)
        end
    end

    println("Wrote mixing state coagulation simulation to $h5_path")
    return h5_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
