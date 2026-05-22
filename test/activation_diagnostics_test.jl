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
        1e-18, 5.0e-15,   # particle 5: activated
    ]

    sys = ParticleSystem(Val(2), 5, 1.0e-6, t -> SVector(0.0, 0.0))

    # Threshold: 1 μm radius
    threshold = 1.0e-6

    frac = activation_fraction(u, sys, Val(A); threshold=threshold, densities=densities)

    # 3 out of 5 are activated
    @test frac ≈ 0.6 atol = 0.01
end

@testset "cloud_droplet_concentration" begin
    A = 2
    densities = SVector(1770.0, 1000.0)

    u = [
        1e-18, 5.0e-15,   # activated
        1e-18, 1e-16,    # not activated
        1e-18, 5.0e-15,   # activated
    ]

    sys = ParticleSystem(Val(2), 3, 1.0e-6, t -> SVector(0.0, 0.0))

    N_d = cloud_droplet_concentration(u, sys, Val(A); threshold=1.0e-6, densities=densities)

    # 2 activated droplets in volume 1e-6 m³ → 2e6 m⁻³
    @test N_d ≈ 2.0e6 atol = 1.0
end
