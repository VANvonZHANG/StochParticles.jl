using Test

include("../examples/simulate_activation_coagulation_comparison.jl")

@testset "Activation comparison example configuration" begin
    cfg = ActivationComparisonConfig()

    n_aitken, n_accum = mode_particle_counts(cfg)
    @test n_aitken == 800
    @test n_accum == 200
    @test n_aitken + n_accum == cfg.n_sim
    @test cfg.n_sim == 1000
    @test cfg.volume * (cfg.aitken_concentration + cfg.accumulation_concentration) ≈ cfg.n_sim
    @test cfg.supersaturation == 0.005
    @test cfg.mode_dry_diameter_threshold == 6.0e-8

    particles, dry_diams, thermo_labels = initial_activation_particles(cfg, cfg.initial_seed)
    @test length(particles) == cfg.n_sim
    @test length(dry_diams) == cfg.n_sim
    @test length(thermo_labels) == cfg.n_sim
    @test all(thermo_labels .== 0.455)

    aitken_diams = dry_diams[1:n_aitken]
    accum_diams = dry_diams[(n_aitken + 1):end]
    @test maximum(aitken_diams) < cfg.mode_dry_diameter_threshold
    @test minimum(accum_diams) > cfg.mode_dry_diameter_threshold

    thermo = average_thermo(cfg)
    sc_aitken = minimum(critical_supersaturation(
        SVector{2, Float64}(m[1], 0.0), thermo, cfg.densities, cfg.T)
        for m in particles[1:n_aitken])
    sc_accum = maximum(critical_supersaturation(
        SVector{2, Float64}(m[1], 0.0), thermo, cfg.densities, cfg.T)
        for m in particles[(n_aitken + 1):end])
    @test sc_accum < cfg.supersaturation
    @test cfg.supersaturation < sc_aitken
end

@testset "Mode classification and per-mode counts" begin
    cfg = ActivationComparisonConfig()
    @test mode_id_from_dry_diameter(2.0e-8, cfg) == 1
    @test mode_id_from_dry_diameter(5.9e-8, cfg) == 1
    @test mode_id_from_dry_diameter(6.0e-8, cfg) == 2
    @test mode_id_from_dry_diameter(2.0e-7, cfg) == 2

    ids = [1, 1, 2, 2, 2]
    flags = [1.0, 0.0, 1.0, 1.0, 0.0]
    n_aitken, n_droplet, act_aitken, act_droplet = per_mode_counts(ids, flags)
    @test (n_aitken, n_droplet) == (2, 3)
    @test (act_aitken, act_droplet) == (1, 2)
    @test per_mode_counts(Int[], Float64[]) == (0, 0, 0, 0)
end
