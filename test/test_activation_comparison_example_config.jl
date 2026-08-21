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
