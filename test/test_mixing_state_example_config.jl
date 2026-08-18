using Test

include("../examples/simulate_mixing_state_coagulation.jl")

@testset "Mixing-state example configuration" begin
    cfg = MixingStateCoagulationConfig()

    @test cfg.n_sim == 4000
    @test cfg.volume == 1.0e-7
    @test cfg.tspan == (0.0, 24.0 * 3600.0)

    save_times = save_times_for_case(cfg)
    @test first(save_times) == cfg.tspan[1]
    @test last(save_times) == cfg.tspan[2]
    @test length(save_times) <= 40
    @test maximum(diff(save_times)) <= 3600.0

    particles = initial_mixing_particles(cfg, cfg.initial_seed)
    @test length(particles) == 4000
    @test count(μ -> μ[1] > 0.0 && μ[2] == 0.0, particles) == 2000
    @test count(μ -> μ[1] == 0.0 && μ[2] > 0.0, particles) == 2000
    @test all(μ -> (μ[1] > 0.0) != (μ[2] > 0.0), particles)
end
