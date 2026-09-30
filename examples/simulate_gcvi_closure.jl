using Distributions
using HDF5
using OrdinaryDiffEq
using Random
using StaticArrays
using StochParticles

include("simulation_io.jl")

const GCVI_BASENAME = "gcvi_closure"
const A = 5                      # [AS, AN, OA, BC, H2O]
const DRY_SPECIES = [1, 2, 3]    # closure chi mask (AS, AN, OA); BC inert

Base.@kwdef struct GcviClosureConfig
    n_sim::Int = 1000
    # synthetic "SMPS" spectrum: bimodal lognormal
    aitken_dg::Float64 = 6.0e-8
    aitken_sigma_g::Float64 = 1.45
    aitken_concentration::Float64 = 8.0e11      # [m^-3]
    accumulation_dg::Float64 = 1.6e-7
    accumulation_sigma_g::Float64 = 1.55
    accumulation_concentration::Float64 = 3.2e11
    # constant dry mean composition fbar (AS, AN, OA, BC) — M0 simplification,
    # replaced by size-resolved fbar(D) in M2
    fbar::SVector{4, Float64} = SVector(0.25, 0.15, 0.50, 0.10)
    # hand-picked Dirichlet concentrations: external / intermediate / internal
    nu_values::Vector{Float64} = [1.2, 12.0, 120.0]
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
    bin_edges::Vector{Float64} = collect(10.0 .^ range(-8.3, -4.5; length = 96))
    initial_seed_base::Int = 2026092500
    seed_base::Int = 2026092500
    truth_nu_index::Int = 2                       # middle nu is the twin truth
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

function volume(cfg::GcviClosureConfig)
    cfg.n_sim / (cfg.aitken_concentration + cfg.accumulation_concentration)
end

function mode_counts(cfg::GcviClosureConfig)
    total = cfg.aitken_concentration + cfg.accumulation_concentration
    n_aitken = round(Int, cfg.n_sim * cfg.aitken_concentration / total)
    return n_aitken, cfg.n_sim - n_aitken
end

"""Sample initial particles: SMPS-like bimodal dry diameters + Dirichlet(nu*fbar)
composition. M0 simplification: constant fbar, hand-picked nu (no calibration)."""
function initial_particles(cfg::GcviClosureConfig, nu::Float64)
    n_aitken, n_accum = mode_counts(cfg)
    diameters = vcat(
        cfg.aitken_dg .* exp.(log(cfg.aitken_sigma_g) .* randn(n_aitken)),
        cfg.accumulation_dg .* exp.(log(cfg.accumulation_sigma_g) .* randn(n_accum)))
    particles = Vector{SVector{5, Float64}}(undef, cfg.n_sim)
    for i in 1:(cfg.n_sim)
        f = rand(Dirichlet(nu .* cfg.fbar))
        rho_eff = 1.0 / sum(f[k] / cfg.densities[k] for k in 1:4)
        m_total = (pi / 6.0) * diameters[i]^3 * rho_eff
        particles[i] = SVector{5, Float64}(m_total .* f..., 0.0)
    end
    # equilibrate haze to Kohler equilibrium at the (subsaturated) ramp start
    pre_equilibrate!(particles, thermo(cfg), cfg.densities, cfg.T0,
        saturation_vapor_pressure(cfg.T0); h2o_idx = cfg.h2o_idx)
    return particles
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

function measured_chi(cfg::GcviClosureConfig, particles)
    sys = ParticleSystem(Val(A), cfg.n_sim, volume(cfg), env_profile(cfg))
    return mixing_state_index(make_u0(particles), sys; species = DRY_SPECIES)
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

function write_twin_obs(path, cfg::GcviClosureConfig, final_record, chi_true)
    h5open(path, "w") do file
        g = create_group(file, "obs")
        attrs(g)["schema_version"] = "synthetic-obs-v1"
        attrs(g)["chi_true"] = chi_true
        attrs(g)["acsm_species"] = "AS,AN,OA"
        g["bin_edges"] = collect(cfg.bin_edges)
        g["cr_dNdlogD"] = collect(final_record.cr_spectrum)
        g["ci_dNdlogD"] = collect(final_record.ci_spectrum)
        g["cr_chemistry"] = collect(final_record.cr_chemistry)
        g["ci_chemistry"] = collect(final_record.ci_chemistry)
    end
    return path
end

function main()
    cfg = GcviClosureConfig()
    n_replicates = example_replicates()
    h5_path = joinpath(example_data_dir(), GCVI_BASENAME * ".h5")
    recreate_h5(h5_path; scene_name = GCVI_BASENAME, n_replicates = n_replicates,
        notes = "GCVI closure M0 walking skeleton: nu scan, open-loop S(t) ramp, virtual GCVI.")

    truth_record_final = nothing
    truth_chi = NaN
    h5open(h5_path, "r+") do file
        for (nu_idx, nu) in enumerate(cfg.nu_values)
            case_group = ensure_case_group(file, "nu_$(nu)";
                attrs_dict = Dict{String, Any}(
                    "nu" => nu, "truth" => nu_idx == cfg.truth_nu_index))
            for replicate_idx in 1:n_replicates
                # fixed initial population per nu; process stochasticity per replicate
                Random.seed!(cfg.initial_seed_base + nu_idx)
                particles = initial_particles(cfg, nu)
                chi_realized = measured_chi(cfg, particles)
                # M0 note: processes = condensation only (no jumps) -> solve_split consumes no RNG; replicates within a case are expected bitwise identical; the per-replicate seed matters once jump processes are enabled.
                Random.seed!(cfg.seed_base + 1000 * nu_idx + replicate_idx)

                sol, records = solve_case(cfg, deepcopy(particles))
                @assert sol.retcode == ReturnCode.Success

                attrs_dict = Dict{String, Any}(
                    "nu" => nu, "chi_realized" => chi_realized,
                    "seed" => cfg.seed_base + 1000 * nu_idx + replicate_idx,
                    "initial_seed" => cfg.initial_seed_base + nu_idx)
                rep_group = create_replicate_group(case_group, replicate_idx)
                _write_attrs!(rep_group, attrs_dict)
                sys0 = ParticleSystem(Val(A), cfg.n_sim, volume(cfg), env_profile(cfg))
                dry0 = dry_diameters_from_state(make_u0(particles), sys0, cfg)
                write_records_common!(rep_group, records, cfg.n_sim, cfg.bin_edges;
                    dry_diameter_initial = dry0, extra_attrs = attrs_dict)

                if nu_idx == cfg.truth_nu_index && replicate_idx == 1
                    truth_record_final = records[end]
                    truth_chi = chi_realized
                end
            end
        end
    end

    obs_path = joinpath(example_data_dir(), "synthetic", "twin_obs_v0.h5")
    mkpath(dirname(obs_path))
    write_twin_obs(obs_path, cfg, truth_record_final, truth_chi)
    println("Wrote GCVI closure skeleton to $h5_path")
    println("Wrote twin observations to $obs_path (chi_true = $truth_chi)")
    return h5_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
