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

@testset "Saturation vapor pressure" begin
    # At 273.15 K, p_sat ≈ 611 Pa
    p_sat_0 = saturation_vapor_pressure(273.15)
    @test p_sat_0 ≈ 611.0 atol = 10.0

    # At 293.15 K, p_sat ≈ 2340 Pa
    p_sat_20 = saturation_vapor_pressure(293.15)
    @test p_sat_20 ≈ 2340.0 atol = 50.0
end

@testset "Modified diffusion coefficient" begin
    thermo = ThermodynamicsParams(
        SVector(0.61, 0.0, 0.0),  # κ values
        0.072,                     # σ [N/m]
        1000.0,                    # ρ_w [kg/m³]
        18.015e-3,                 # M_w [kg/mol]
        2.5e6,                     # L_v [J/kg]
        461.5,                     # R_v [J/kg/K]
        2.5e-5,                    # D_v [m²/s]
        2.4e-2,                    # k_a [W/m/K]
    )

    T = 293.15
    p_sat = saturation_vapor_pressure(T)
    D_v_prime = modified_diffusion_coefficient(thermo, T, p_sat)

    # D_v' should be slightly less than D_v due to latent heat resistance
    @test D_v_prime < thermo.D_v
    @test D_v_prime > 0.5 * thermo.D_v  # not too small
end
