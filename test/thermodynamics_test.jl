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
        2.4e-2                    # k_a [W/m/K]
    )

    # Anchor 1 — first principles (same derivation written out independently;
    # guards against transcription errors in the coefficient assembly):
    #   1/D′ = 1/D_v + (L_v/k_a)·(ρ_sat/T)·(L_v·M_w/(R_u·T) − 1),  ρ_sat = p_sat·M_w/(R_u·T)
    # (exact Clausius-Clapeyron differential; the −1 matches Ghan 2011 A1 /
    # Pruppacher & Klett / pyrcel G_b.)
    for T in (273.15, 285.0, 293.15, 303.15)
        p_sat = saturation_vapor_pressure(T)
        R_u = thermo.R_v * thermo.M_w
        ρ_sat = p_sat * thermo.M_w / (R_u * T)
        expected = 1.0 / (1.0 / thermo.D_v +
                    thermo.L_v / thermo.k_a * ρ_sat / T *
                    (thermo.L_v * thermo.M_w / (R_u * T) - 1.0))
        @test modified_diffusion_coefficient(thermo, T, p_sat) ≈ expected rtol = 1e-12
    end

    # Anchor 2 — physical magnitude at 293.15 K: latent-heat conduction must
    # dominate the total resistance (D′ ≈ 0.27·D_v). The pre-fix assembly
    # (spurious p_sat/p_atm factor, thermal term ~40× too weak) returned
    # ≈ 0.94·D_v — regression guard against reintroduction.
    T = 293.15
    D_v_prime = modified_diffusion_coefficient(thermo, T, saturation_vapor_pressure(T))
    @test 0.15 * thermo.D_v < D_v_prime < 0.5 * thermo.D_v

    # Anchor 3 — pyrcel cross-check at the M4 campaign point (285 K).
    # Corrected continuum D′ with library constants ≈ 9.13e-6 m²/s. pyrcel's
    # own G-form (L_v = 2.25e6 J/kg, Magnus e_s, ka = 1e-3·(4.39+0.071·T))
    # gives ≈ 1.06e-5 — a ~16% constant-level gap (their L_v is 9% low),
    # documented here rather than absorbed into a tolerance.
    D_285 = modified_diffusion_coefficient(thermo, 285.0, saturation_vapor_pressure(285.0))
    @test D_285 ≈ 9.13e-6 rtol = 0.05
end

@testset "Equilibrium vapor pressure" begin
    thermo = ThermodynamicsParams(
        SVector(0.61, 0.0, 0.0),
        0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2
    )
    densities = SVector(1770.0, 1800.0, 1000.0)
    T = 293.15

    # Small dry particle (~10 nm diameter equivalent)
    # m_dry = pi/6 * d^3 * rho = pi/6 * (10e-9)^3 * 1770 ≈ 9.27e-22 kg
    m_dry = SVector(9.27e-22, 0.0, 0.0)
    m_w = 5e-21  # ~5x dry mass in water

    p_eq = equilibrium_vapor_pressure(m_dry, m_w, thermo, densities, T)
    p_sat = saturation_vapor_pressure(T)

    # For this small particle, Kelvin effect dominates over water activity reduction
    # resulting in supersaturated equilibrium (p_eq > p_sat)
    @test p_eq > p_sat
end

@testset "Particle wet radius" begin
    densities = SVector(1770.0, 1800.0, 1000.0)
    m_dry = SVector(1e-18, 0.0, 0.0)
    m_w = 1e-18
    R = particle_wet_radius(m_dry, m_w, densities)
    # V_total = V_dry + V_w = 1e-18/1770 + 1e-18/1000 ≈ 1.565e-21
    # R = (3V/4π)^(1/3) ≈ 7.20e-8
    @test R ≈ 7.20e-8 atol = 1e-9
end

@testset "Critical supersaturation" begin
    thermo = ThermodynamicsParams(
        SVector(0.61, 0.0, 0.0),
        0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2
    )
    densities = SVector(1770.0, 1800.0, 1000.0)
    T = 293.15

    # Small particle (50 nm dry diameter)
    # Higher Sc (harder to activate)
    m_dry_small = SVector(1.16e-19, 0.0, 0.0)  # ~50 nm SO4
    Sc_small = critical_supersaturation(m_dry_small, thermo, densities, T)
    @test Sc_small > 0.0
    @test Sc_small > 0.001  # small particles have high Sc

    # Large particle (200 nm dry diameter)
    # Lower Sc (easier to activate)
    m_dry_large = SVector(7.4e-18, 0.0, 0.0)  # ~200 nm SO4
    Sc_large = critical_supersaturation(m_dry_large, thermo, densities, T)
    @test Sc_large > 0.0
    @test Sc_large < Sc_small  # larger particles have lower Sc

    # 100 nm ammonium sulfate at 293K
    # Exact Köhler maximum: Sc ≈ 0.15% (literature: 0.1%–0.2%)
    m_dry = SVector(9.27e-19, 0.0, 0.0)  # ~100 nm SO4
    Sc = critical_supersaturation(m_dry, thermo, densities, T)
    @test 0.001 < Sc < 0.003  # ~0.15%
end

@testset "water_activity pure water (V_dry=0)" begin
    κ = SVector(0.0, 0.0)
    densities = SVector(1000.0, 1000.0)
    m_dry = SVector(0.0, 0.0)
    m_w = 1e-18
    a_w = water_activity(m_dry, m_w, κ, densities)
    @test a_w == 1.0
end

@testset "critical_supersaturation zero dry mass" begin
    thermo = ThermodynamicsParams(
        SVector(0.61, 0.0),
        0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2
    )
    densities = SVector(1770.0, 1000.0)
    m_dry = SVector(0.0, 0.0)
    Sc = critical_supersaturation(m_dry, thermo, densities, 293.15)
    @test Sc == 0.0
end

@testset "particle_wet_radius zero water" begin
    densities = SVector(1770.0, 1000.0)
    m_dry = SVector(1e-18, 0.0)
    R = particle_wet_radius(m_dry, 0.0, densities)
    # Only dry volume contributes
    V_dry = 1e-18 / 1770.0
    @test R ≈ cbrt(3.0 * V_dry / (4.0 * π)) atol = 1e-12
end

@testset "equilibrium_water_mass" begin
    T = 293.15
    p_sat = saturation_vapor_pressure(T)
    S = 0.003
    p_v = p_sat * (1.0 + S)
    densities = SVector(1770.0, 1000.0)
    thermo = ThermodynamicsParams(
        SVector(0.455, 0.0), 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)

    # Small particle (non-activated, Sc > S)
    d_dry = 20e-9
    m_so4 = (π / 6.0) * d_dry^3 * densities[1]
    m_dry = SVector{2, Float64}(m_so4, 0.0)
    m_eq = equilibrium_water_mass(m_dry, thermo, densities, T, p_v)
    @test m_eq > 0.0
    # At equilibrium, p_eq should equal p_v
    p_eq_at_eq = equilibrium_vapor_pressure(m_dry, m_eq, thermo, densities, T)
    @test isapprox(p_eq_at_eq, p_v; rtol = 1e-10)

    # Larger particle (non-activated)
    d_dry = 50e-9
    m_so4 = (π / 6.0) * d_dry^3 * densities[1]
    m_dry = SVector{2, Float64}(m_so4, 0.0)
    m_eq = equilibrium_water_mass(m_dry, thermo, densities, T, p_v)
    @test m_eq > 0.0
    p_eq_at_eq = equilibrium_vapor_pressure(m_dry, m_eq, thermo, densities, T)
    @test isapprox(p_eq_at_eq, p_v; rtol = 1e-10)

    # Larger particle should have larger equilibrium water mass
    m_eq_small = equilibrium_water_mass(
        SVector{2, Float64}((π/6.0)*(20e-9)^3*densities[1], 0.0), thermo, densities, T, p_v)
    m_eq_large = equilibrium_water_mass(
        SVector{2, Float64}((π/6.0)*(50e-9)^3*densities[1], 0.0), thermo, densities, T, p_v)
    @test m_eq_large > m_eq_small
end
