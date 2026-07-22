using HDF5
using OrdinaryDiffEq
using Random
using StaticArrays
using StochParticles

include("simulation_io.jl")

const SINGLE_COMPONENT_BASENAME = "single_component_coagulation"
const A = 1
const EQUAL_KERNEL_FRACTIONS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

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

function initial_particles(n_sim, mode_small, sigma_small, mode_large, sigma_large, density)
    n_half = div(n_sim, 2)
    particles_small = lognormal_masses(n_half, mode_small, sigma_small, density)
    particles_large = lognormal_masses(n_sim - n_half, mode_large, sigma_large, density)
    return vcat(particles_small, particles_large)
end

function case_attributes(cfg)
    return Dict{String, Any}(
        "species_names" => cfg.species_names,
        "density" => Float64(cfg.densities[1]),
        "n_sim" => cfg.n_sim,
        "volume" => cfg.volume,
        "t_start" => Float64(cfg.tspan[1]),
        "t_end" => Float64(cfg.tspan[2]),
        "saveat" => cfg.saveat,
        "notes" => cfg.notes,
    )
end

function build_aerosol_case()
    params = standard_aerosol_atmosphere()
    densities = SVector(1800.0)
    return (
        name = "aerosol_brownian",
        species_names = ["SO4"],
        densities = densities,
        n_sim = 200,
        volume = 5.0e-10,
        tspan = (0.0, 3600.0),
        saveat = 60.0,
        seed_base = 2026072100,
        bin_edges = collect(10.0 .^ range(-9, -5; length = 81)),
        notes = "Single-component sulfate aerosol Brownian coagulation.",
        build_particles = seed -> begin
            Random.seed!(seed)
            initial_particles(200, 5.0e-9, 1.5, 1.0e-7, 1.5, 1800.0)
        end,
        build_kernel = () -> BrownianKernel(params.T, params.p, densities),
        record_extras = nothing,
    )
end

function build_cloud_case()
    params = standard_cloud_atmosphere()
    densities = SVector(1000.0)
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
        n_sim = 200,
        volume = 1.0e-6,
        tspan = (0.0, 200.0),
        saveat = 4.0,
        seed_base = 2026073100,
        bin_edges = collect(10.0 .^ range(-6, -3; length = 81)),
        notes = "Single-component water cloud droplet composite coagulation.",
        build_particles = seed -> begin
            Random.seed!(seed)
            initial_particles(200, 5.0e-6, 1.3, 2.5e-5, 1.3, 1000.0)
        end,
        build_kernel = () -> make_kernel(params, 0.01, 50.0, densities),
        record_extras = (u, sys) -> begin
            brownian_frac, gravitational_frac, turbulent_frac =
                kernel_fraction_triplet(u, sys, parts)
            return (
                kernel_fraction_brownian = brownian_frac,
                kernel_fraction_gravitational = gravitational_frac,
                kernel_fraction_turbulent = turbulent_frac,
            )
        end,
    )
end

function run_replicate!(case_group, cfg, replicate_idx)
    seed = cfg.seed_base + replicate_idx
    particles = cfg.build_particles(seed)
    dry_diameter_initial = diameters_from_masses(particles, cfg.densities)

    kernel = cfg.build_kernel()
    coagulation = CoagulationProcess(kernel, GlobalMajorant())
    gas_fn = _t -> SVector(0.0)
    prob = ParticleProblem(
        particles, cfg.volume, gas_fn, (coagulation,); tspan = cfg.tspan, n_sim = cfg.n_sim)

    record_func = (t, u, sys) -> begin
        base = base_diagnostic_record(t, u, sys, Val(A), cfg.densities, cfg.bin_edges)
        if cfg.record_extras === nothing
            return base
        end
        return merge_record(base, cfg.record_extras(u, sys))
    end

    sol, records = solve_with_records(prob, Tsit5(); saveat = cfg.saveat, record_func = record_func)
    @assert sol.retcode == ReturnCode.Success

    rep_group = create_replicate_group(case_group, replicate_idx)
    attrs(rep_group)["seed"] = seed
    write_records_common!(
        rep_group,
        records,
        cfg.n_sim,
        cfg.bin_edges;
        dry_diameter_initial = dry_diameter_initial,
        extra_attrs = Dict{String, Any}("seed" => seed),
    )
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
        notes = "Single-component aerosol Brownian and cloud composite coagulation simulations.",
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
