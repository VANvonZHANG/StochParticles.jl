using StochParticles
using StaticArrays
using Test

@testset "H2OCondensationFlux basic" begin
    thermo = ThermodynamicsParams(
        SVector(0.61, 0.0),
        0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2,
    )
    densities = SVector(1770.0, 1000.0)  # SO4, H2O
    h2o_idx = 2

    flux = H2OCondensationFlux(thermo, h2o_idx, densities, 1.0)

    # Particle: 100 nm dry SO4 with some water
    # m_dry_SO4 = π/6 · (100e-9)³ · 1770 ≈ 9.27e-19 kg
    m_dry = SVector(9.27e-19, 1.0e-18)  # SO4, H2O

    # Environment: T=293K, p_v slightly below saturation
    T = 293.15
    p_v = saturation_vapor_pressure(T) * 0.99  # 99% RH
    env = SVector(T, p_v)

    # Dummy sys and t (not used in basic flux)
    sys = nothing
    t = 0.0

    dμ = flux(m_dry, env, sys, t)

    # At 99% RH, small particle should be near equilibrium or slightly evaporating
    # The flux should be small (near zero)
    @test abs(dμ[2]) < 5e-15  # very small mass change rate
end

@testset "H2OCondensationFlux condensation" begin
    thermo = ThermodynamicsParams(
        SVector(0.61, 0.0),
        0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2,
    )
    densities = SVector(1770.0, 1000.0)
    h2o_idx = 2
    flux = H2OCondensationFlux(thermo, h2o_idx, densities, 1.0)

    # Same particle
    m_dry = SVector(9.27e-19, 1.0e-18)

    # Environment: supersaturated (105% RH)
    T = 293.15
    p_v = saturation_vapor_pressure(T) * 1.05
    env = SVector(T, p_v)

    dμ = flux(m_dry, env, nothing, 0.0)

    # Should condense (positive water mass change)
    @test dμ[2] > 0.0
    # Dry mass should not change
    @test dμ[1] == 0.0
end

@testset "H2OCondensationProcess convenience constructor" begin
    thermo = ThermodynamicsParams(
        SVector(0.61, 0.0),
        0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2,
    )
    densities = SVector(1770.0, 1000.0)

    proc = H2OCondensationProcess(thermo, densities; h2o_idx=2, w=1.0)

    @test proc isa CondensationProcess
    @test provides_drift(proc) == true

    # Test that it produces flux
    sys = ParticleSystem(Val(2), 10, 1.0e-6, t -> SVector(293.15, 1000.0))
    μ = SVector(1e-18, 1e-18)
    dμ = apply_drift(proc, μ, sys, 0.0)
    @test dμ isa SVector{2, Float64}
end
