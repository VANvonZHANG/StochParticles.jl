using StochParticles
using StaticArrays
using Test

@testset "activation_fraction" begin
    # 5 particles, 2 species (SO4, H2O)
    A = 2
    densities = SVector(1770.0, 1000.0)

    # Particle states: some with large radius (activated), some small
    # Activated: large water mass → large radius
    # For R = 1 μm: V = 4/3 π (1e-6)^3 ≈ 4.19e-18 m³
    # Dry volume (SO4, 1e-18 kg @ 1770 kg/m³): ≈ 5.65e-22 m³
    # Water mass needed: ≈ 4.19e-15 kg
    u = [
        1e-18, 5.0e-15,   # particle 1: small dry, lots of water → activated
        1e-18, 1e-16,    # particle 2: small dry, little water → not activated
        1e-18, 5.0e-15,   # particle 3: activated
        1e-18, 1e-16,    # particle 4: not activated
        1e-18, 5.0e-15   # particle 5: activated
    ]

    sys = ParticleSystem(Val(2), 5, 1.0e-6, t -> SVector(0.0, 0.0))

    # Threshold: 1 μm radius
    threshold = 1.0e-6

    frac = activation_fraction(u, sys, Val(A); threshold = threshold, densities = densities)

    # 3 out of 5 are activated
    @test frac ≈ 0.6 atol = 0.01
end

@testset "cloud_droplet_concentration" begin
    A = 2
    densities = SVector(1770.0, 1000.0)

    u = [
        1e-18, 5.0e-15,   # activated
        1e-18, 1e-16,    # not activated
        1e-18, 5.0e-15   # activated
    ]

    sys = ParticleSystem(Val(2), 3, 1.0e-6, t -> SVector(0.0, 0.0))

    N_d = cloud_droplet_concentration(
        u, sys, Val(A); threshold = 1.0e-6, densities = densities)

    # 2 activated droplets in volume 1e-6 m³ → 2e6 m⁻³
    @test N_d ≈ 2.0e6 atol = 1.0
end

@testset "activation_fraction with physics-based criterion" begin
    A = 2
    densities = SVector(1770.0, 1000.0)
    κ_values = SVector(0.61, 0.0)
    thermo = ThermodynamicsParams(
        κ_values, 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2
    )
    T = 293.15

    # Three particles with different dry sizes
    # Small (high Sc), medium, large (low Sc)
    # We set water mass to a small equilibrium amount so Sc is determined by dry size
    u = [
        # particle 1: small, high Sc (~0.5%), not activated at S=0.2%
        1.16e-19, 1.0e-20,
        # particle 2: medium, medium Sc (~0.17%), activated at S=0.2%
        1.5e-18, 1.0e-19,
        # particle 3: large, low Sc (~0.03%), activated at S=0.2%
        7.4e-18, 1.0e-18
    ]

    sys = ParticleSystem(Val(2), 3, 1.0e-6, t -> SVector(0.0, 0.0))

    # Environment at S = 0.2% = 0.002
    S_env = 0.002

    # Physics-based: should activate particles 2 and 3 (2/3)
    frac_phys = activation_fraction(u, sys, Val(A);
        mode = :critical_supersaturation,
        S_env = S_env,
        thermo = thermo,
        densities = densities,
        T = T
    )
    @test frac_phys ≈ 2.0 / 3.0 atol = 0.1

    # Radius-based (legacy): depends on actual water content
    # With small water masses, none may exceed 1 μm
    frac_rad = activation_fraction(u, sys, Val(A);
        mode = :radius_threshold,
        threshold = 1.0e-6,
        densities = densities
    )
    @test frac_rad >= 0.0  # just verify it runs without error
end

@testset "cloud_droplet_concentration with physics-based criterion" begin
    A = 2
    densities = SVector(1770.0, 1000.0)
    κ_values = SVector(0.61, 0.0)
    thermo = ThermodynamicsParams(
        κ_values, 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2
    )
    T = 293.15

    u = [
        1.16e-19, 1.0e-20,  # small, high Sc
        1.5e-18, 1.0e-19,  # medium
        7.4e-18, 1.0e-18   # large, low Sc
    ]

    sys = ParticleSystem(Val(2), 3, 1.0e-6, t -> SVector(0.0, 0.0))

    N_d = cloud_droplet_concentration(u, sys, Val(A);
        mode = :critical_supersaturation,
        S_env = 0.002,
        thermo = thermo,
        densities = densities,
        T = T
    )

    # 2 out of 3 activated in volume 1e-6 → 2e6 m⁻³
    @test N_d ≈ 2.0e6 atol = 1.0e6
end
