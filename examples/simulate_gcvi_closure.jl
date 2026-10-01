using HDF5
using OrdinaryDiffEq
using Random
using StaticArrays
using Statistics
using StochParticles

include("simulation_io.jl")

const GCVI_BASENAME = "gcvi_closure"
const A = 5                      # [AS, AN, OA, BC, H2O]
const DRY_SPECIES = [1, 2, 3]    # closure chi mask (AS, AN, OA); BC inert

Base.@kwdef struct GcviClosureConfig
    n_sim::Int = 1000
    # synthetic "SMPS" spectrum: bimodal lognormal baked onto the table
    aitken_dg::Float64 = 6.0e-8
    aitken_sigma_g::Float64 = 1.45
    aitken_concentration::Float64 = 8.0e11      # [m^-3]
    accumulation_dg::Float64 = 1.6e-7
    accumulation_sigma_g::Float64 = 1.55
    accumulation_concentration::Float64 = 3.2e11
    bin_edges::Vector{Float64} = collect(10.0 .^ range(-8.3, -4.5; length = 96))
    # size-resolved fbar(D) anchors (AS, AN, OA, BC): small particles
    # OA-rich, large particles AS-rich; trend strength sized so that
    # reachable_chi_max stays above 0.85 (spec §7 tuning gate)
    fbar_anchors::Vector{Tuple{Float64,
        SVector{4, Float64}}} = [
        (3.0e-8, SVector(0.18, 0.15, 0.57, 0.10)),
        (3.0e-7, SVector(0.35, 0.20, 0.30, 0.15))
    ]
    chi_grid::Vector{Float64} = [0.10, 0.25, 0.40, 0.60, 0.75]
    chi_true::Float64 = 0.50                     # off-grid twin truth
    # open-loop environment: linear S ramp to S_peak, then hold
    T0::Float64 = 288.15
    S_peak::Float64 = 0.004
    ramp_time::Float64 = 120.0
    tspan::Tuple{Float64, Float64} = (0.0, 600.0)
    saveat::Float64 = 60.0
    dt_split::Float64 = 10.0
    # physics
    densities::SVector{5, Float64} = SVector(1770.0, 1720.0, 1400.0, 1800.0, 1000.0)
    kappas::SVector{5, Float64} = SVector(0.61, 0.67, 0.10, 0.0, 0.0)
    h2o_idx::Int = 5
    activation_radius::Float64 = 1.0e-6
    # virtual GCVI
    gcvi::GCVIResponse = GCVIResponse()
    initial_seed_base::Int = 2026093000
    seed_base::Int = 2026093000
end

function thermo(cfg::GcviClosureConfig)
    ThermodynamicsParams(
        cfg.kappas, 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)
end

function env_profile(cfg::GcviClosureConfig)
    PrescribedProfile(
        [0.0, cfg.ramp_time, cfg.tspan[2]],
        [cfg.T0, cfg.T0, cfg.T0],
        [0.0, cfg.S_peak, cfg.S_peak])
end

function spectrum(cfg::GcviClosureConfig)
    return lognormal_table(cfg.aitken_dg, cfg.aitken_sigma_g,
        cfg.aitken_concentration, cfg.bin_edges) +
           lognormal_table(cfg.accumulation_dg, cfg.accumulation_sigma_g,
        cfg.accumulation_concentration, cfg.bin_edges)
end

function fbar(cfg::GcviClosureConfig)
    SizeResolvedComposition(spectrum(cfg); anchors = cfg.fbar_anchors)
end

volume(cfg::GcviClosureConfig) = cfg.n_sim / number_concentration(spectrum(cfg))

function population_spec(cfg::GcviClosureConfig, chi_target::Float64)
    return SyntheticPopulationSpec(
        n_sim = cfg.n_sim, spectrum = spectrum(cfg), fbar = fbar(cfg),
        chi_target = chi_target, densities = cfg.densities,
        h2o_idx = cfg.h2o_idx, chi_species = DRY_SPECIES,
        T0 = cfg.T0, S0 = 0.0)
end

function dry_diameters_from_state(u, sys, cfg::GcviClosureConfig)
    out = Vector{Float64}(undef, sys.n_active)
    for i in 1:(sys.n_active)
        μ = get_particle(u, i, Val(A))
        V_dry = sum(μ[k] / cfg.densities[k] for k in 1:4)
        out[i] = cbrt(6.0 * V_dry / pi)
    end
    return out
end

function record_extras(t, u, sys, cfg::GcviClosureConfig)
    dry_diams = dry_diameters_from_state(u, sys, cfg)
    cr_flags = classify_cr_ci(u, sys, Val(A), cfg.gcvi, cfg.densities)
    smps = virtual_smps(cr_flags, dry_diams, cfg.bin_edges, sys.volume)
    acsm = virtual_acsm(u, sys, Val(A), cr_flags; species = DRY_SPECIES)
    return (
        activation_fraction = activation_fraction(
            u, sys, Val(A); mode = :radius_threshold,
            threshold = cfg.activation_radius, densities = cfg.densities),
        chi_dry = mixing_state_index(u, sys; species = DRY_SPECIES),
        dry_diameter_samples = dry_diams,
        gcvi_cr_flags = Float64.(cr_flags),
        cr_spectrum = smps.cr,
        ci_spectrum = smps.ci,
        cr_chemistry = collect(acsm.cr),
        ci_chemistry = collect(acsm.ci)
    )
end

function solve_case(cfg::GcviClosureConfig, particles)
    condensation = H2OCondensationProcess(
        thermo(cfg), cfg.densities; h2o_idx = cfg.h2o_idx, w = 0.0)
    record_func = (t,
        u,
        sys) -> merge_record(
        base_diagnostic_record(t, u, sys, Val(A), cfg.densities, cfg.bin_edges),
        record_extras(t, u, sys, cfg))
    # kg-scale states need explicit tolerances: solver defaults (abstol = 1e-6)
    # exceed particle masses (~1e-16 kg) by ~10 orders, letting water mass go
    # negative within accepted steps
    return solve_split(particles, volume(cfg), env_profile(cfg),
        (condensation,), Tsit5();
        tspan = cfg.tspan, n_sim = cfg.n_sim, dt_split = cfg.dt_split,
        saveat = cfg.saveat, record_func = record_func,
        abstol = 1.0e-24, reltol = 1.0e-6)
end

function write_twin_obs(path, cfg::GcviClosureConfig, truth_records, chis_true)
    # pooled observation (schema v2): real SMPS/ACSM instruments average over
    # the sampling window, so the twin obs is the truth-case replicate mean,
    # not a single realization (a single-replicate obs makes the closure cost
    # measure replicate identity rather than chi)
    n = length(truth_records)
    cr_spec = mean([collect(r.cr_spectrum) for r in truth_records])
    ci_spec = mean([collect(r.ci_spectrum) for r in truth_records])
    cr_chem = mean([collect(r.cr_chemistry) for r in truth_records])
    ci_chem = mean([collect(r.ci_chemistry) for r in truth_records])
    chi_true = mean(chis_true)
    h5open(path, "w") do file
        g = create_group(file, "obs")
        attrs(g)["schema_version"] = "synthetic-obs-v2"
        attrs(g)["chi_true"] = chi_true
        attrs(g)["n_replicates_pooled"] = n
        attrs(g)["acsm_species"] = "AS,AN,OA"
        g["bin_edges"] = collect(cfg.bin_edges)
        g["cr_dNdlogD"] = cr_spec
        g["ci_dNdlogD"] = ci_spec
        g["cr_chemistry"] = cr_chem
        g["ci_chemistry"] = ci_chem
    end
    return path
end

function main()
    cfg = GcviClosureConfig()
    chi_inf = reachable_chi_max(spectrum(cfg), fbar(cfg);
        densities = cfg.densities, chi_species = DRY_SPECIES)
    @assert chi_inf > 0.85 "chi_inf = $chi_inf <= 0.85: weaken fbar anchors (spec §7)"
    @assert !(cfg.chi_true in cfg.chi_grid) "chi_true must be off-grid"
    chi_cases = vcat(cfg.chi_grid, cfg.chi_true)
    n_replicates = example_replicates()
    h5_path = joinpath(example_data_dir(), GCVI_BASENAME * ".h5")
    recreate_h5(h5_path; scene_name = GCVI_BASENAME, n_replicates = n_replicates,
        notes = "GCVI closure M2: calibrated chi grid (truth off-grid at 0.5), " *
                "tabulated spectrum, size-resolved fbar(D), open-loop S(t) ramp.")

    truth_records = []
    truth_chis = Float64[]
    chis_realized = Dict{Int, Vector{Float64}}()
    h5open(h5_path, "r+") do file
        for (case_idx, chi) in enumerate(chi_cases)
            truth = chi == cfg.chi_true
            case_group = ensure_case_group(file, "chi_$(chi)";
                attrs_dict = Dict{String, Any}(
                    "chi_target" => chi, "truth" => truth))
            case_nu = nothing
            for replicate_idx in 1:n_replicates
                # per-replicate population resampling: replicate spread IS the
                # identifiability noise (chi fluctuation + J noise floor)
                initial_seed = cfg.initial_seed_base + 100 * case_idx + replicate_idx
                particles, dry0,
                meta = synthesize_population(
                    population_spec(cfg, chi);
                    seed = initial_seed, thermo = thermo(cfg))
                case_nu === nothing && (case_nu = meta.nu)
                push!(get!(chis_realized, case_idx, Float64[]), meta.chi_realized)
                attrs_dict = Dict{String, Any}(
                    "chi_target" => chi, "chi_realized" => meta.chi_realized,
                    "nu" => meta.nu, "seed" =>
                        cfg.seed_base + 1000 * case_idx + replicate_idx,
                    "initial_seed" => initial_seed, "truth" => truth)
                Random.seed!(cfg.seed_base + 1000 * case_idx + replicate_idx)
                sol, records = solve_case(cfg, particles)
                @assert sol.retcode == ReturnCode.Success
                rep_group = create_replicate_group(case_group, replicate_idx)
                _write_attrs!(rep_group, attrs_dict)
                write_records_common!(rep_group, records, cfg.n_sim, cfg.bin_edges;
                    dry_diameter_initial = dry0, extra_attrs = attrs_dict)
                if truth
                    push!(truth_records, records[end])
                    push!(truth_chis, meta.chi_realized)
                end
                println("case chi=$chi rep=$replicate_idx: " *
                        "chi_realized=$(round(meta.chi_realized, digits = 4)) " *
                        "nu=$(round(meta.nu, digits = 2))")
            end
            attrs(case_group)["nu"] = case_nu
            # case-level chi gate (user-adjudicated 2026-09-30): the per-replicate
            # hard assert was infeasible — realized-chi has sd ~0.011 at n_sim = 1000
            # in the sparse-mixing regime (measured 5/64 seeds beyond 0.02), and
            # single-replicate excursions ARE the resampling noise this experiment
            # must quantify (sigma_J). Gate the calibration accuracy at case level.
            mean_dev = abs(mean(chis_realized[case_idx]) - chi)
            @assert mean_dev < 0.02 "case-level chi deviation $mean_dev >= 0.02 for chi=$chi"
            sigma_chi = std(chis_realized[case_idx])
            @assert sigma_chi <= 0.02 "sigma_chi = $sigma_chi exceeds 0.02 for chi=$chi"
            println("case chi=$chi DONE: nu=$(round(case_nu, digits = 2)), " *
                    "chi_realized=$(round(mean(chis_realized[case_idx]), digits = 4)) " *
                    "± $(round(sigma_chi, digits = 4))")
        end
    end

    obs_path = joinpath(example_data_dir(), "synthetic", "twin_obs_v0.h5")
    mkpath(dirname(obs_path))
    write_twin_obs(obs_path, cfg, truth_records, truth_chis)
    println("Wrote GCVI closure M2 run to $h5_path")
    println("Wrote twin observations to $obs_path (pooled over $(length(truth_chis)) " *
            "truth replicates, chi_true = $(round(mean(truth_chis), digits = 4)))")
    println("chi_inf = $(round(chi_inf, digits = 4))")
    return h5_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
