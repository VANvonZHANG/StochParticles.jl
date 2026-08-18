using Test
using StaticArrays
using StochParticles

include(joinpath(@__DIR__, "..", "examples", "coupled_condensation_composite_coagulation_pdmp.jl"))

@testset "Mixed PDMP demonstration helpers" begin
    cfg = MixedPDMPConfig(
        n_sim = 24,
        volume = 2.4e-6,
        tspan = (0.0, 2.0),
        saveat = 0.5,
        seed = 42
    )

    particles = build_initial_particles(cfg)
    @test length(particles) == cfg.n_sim
    @test all(p -> p[1] > 0.0, particles)
    @test all(p -> p[2] == 0.0, particles)

    thermo = make_thermo(cfg)
    params = make_cloud_params(cfg)
    kernel_parts = build_kernel_parts(cfg, params)

    @test kernel_parts.brownian isa BrownianKernel
    @test kernel_parts.gravitational isa GravitationalKernel
    @test kernel_parts.turbulent isa AyalaTurbulentKernel
    @test kernel_parts.total isa CompositeKernel

    μ_i, μ_j = particles[1], particles[end]
    @test kernel_parts.total(μ_i, μ_j) ≈
          kernel_parts.brownian(μ_i, μ_j) +
          kernel_parts.gravitational(μ_i, μ_j) +
          kernel_parts.turbulent(μ_i, μ_j)

    u0 = make_u0(particles)
    sys = ParticleSystem(Val(2), cfg.n_sim, cfg.volume, gas_state_function(cfg))
    row = evaluate_diagnostics(u0, sys, cfg, kernel_parts, thermo, 0.0)

    @test row.time == 0.0
    @test row.number_concentration > 0.0
    @test row.total_mass_concentration > 0.0
    @test row.total_mass_concentration ≈ total_mass(u0, Val(2), sys.n_active) / sys.volume
    @test row.mean_dry_diameter > 0.0
    @test row.mean_wet_diameter > 0.0
    @test row.mean_wet_diameter >= row.mean_dry_diameter
    @test 0.0 <= row.activation_fraction <= 1.0
    @test row.cloud_droplet_concentration >= 0.0
    @test row.cloud_droplet_concentration <= row.number_concentration
    @test row.brownian_fraction >= 0.0
    @test row.gravitational_fraction >= 0.0
    @test row.turbulent_fraction >= 0.0
    @test isapprox(
        row.brownian_fraction + row.gravitational_fraction + row.turbulent_fraction,
        1.0;
        atol = 1.0e-10
    )
end
