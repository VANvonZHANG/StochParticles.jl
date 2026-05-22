using StochParticles
using StaticArrays
using Test

@testset "κ-Köhler water activity" begin
    # Pure water: κ=0, a_w should be 1.0
    κ = SVector(0.0, 0.0, 0.0)
    densities = SVector(1770.0, 1800.0, 1000.0)  # SO4, BC, H2O
    m_dry = SVector(1e-18, 0.0, 0.0)  # 1 fg SO4, no water
    m_w = 1e-18  # 1 fg water
    a_w = water_activity(m_dry, m_w, κ, densities)
    @test a_w ≈ 1.0 atol = 1e-10

    # Pure ammonium sulfate (κ≈0.61) with known water content
    κ_so4 = 0.61
    κ = SVector(κ_so4, 0.0, 0.0)
    m_dry = SVector(1e-18, 0.0, 0.0)
    m_w = 1e-17  # 10 fg water
    a_w = water_activity(m_dry, m_w, κ, densities)
    # V_w = 10e-18/1000 = 1e-20
    # V_dry = 1e-18/1770 ≈ 5.65e-22
    # a_w = 1e-20 / (1e-20 + 5.65e-22 * 0.61) ≈ 0.967
    @test a_w ≈ 0.967 atol = 0.01
end
