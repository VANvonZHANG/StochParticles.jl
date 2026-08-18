using HDF5
using OrdinaryDiffEq
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
    n_sim::Int = 2000
    volume::Float64 = 1.0e-6
    tspan::Tuple{Float64, Float64} = (0.0, 3600.0)
    saveat::Float64 = 60.0
    T::Float64 = 298.0
    p::Float64 = 1.01325e5
    densities::SVector{2, Float64} = SVector(1770.0, 1800.0)
    bin_edges::Vector{Float64} = collect(10.0 .^ range(-9, -6; length = 81))
    initial_seed::Int = 2026092100
    seed_base::Int = 2026092100
    notes::String = "SO4/BC external-to-internal mixing by Brownian coagulation."
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

function bc_mass_fraction_samples(u, sys)
    fractions = Vector{Float64}(undef, sys.n_active)
    for particle_idx in 1:sys.n_active
        μ = get_particle(u, particle_idx, Val(A))
        total = sum(μ)
        fractions[particle_idx] = total > 0.0 ? μ[2] / total : 0.0
    end
    return fractions
end

function record_extras(u, sys)
    return (
        mixing_state_index = mixing_state_index(u, sys),
        bc_mass_fraction_samples = bc_mass_fraction_samples(u, sys)
    )
end

function case_attributes(cfg::MixingStateCoagulationConfig)
    return Dict{String, Any}(
        "species_names" => cfg.species_names,
        "n_sim" => cfg.n_sim,
        "volume" => cfg.volume,
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

    coagulation = CoagulationProcess(
        BrownianKernel(cfg.T, cfg.p, cfg.densities),
        GlobalMajorant()
    )
    gas_fn = _t -> SVector(0.0, 0.0)
    prob = ParticleProblem(
        particles,
        cfg.volume,
        gas_fn,
        (coagulation,);
        tspan = cfg.tspan,
        n_sim = cfg.n_sim
    )

    record_func = (t, u, sys) -> begin
        base = base_diagnostic_record(t, u, sys, Val(A), cfg.densities, cfg.bin_edges)
        return merge_record(base, record_extras(u, sys))
    end

    sol, records = solve_with_records(prob, Tsit5(); saveat = cfg.saveat, record_func = record_func)
    @assert sol.retcode == ReturnCode.Success

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
