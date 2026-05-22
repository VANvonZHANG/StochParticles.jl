using StochParticles
using StaticArrays
using Test

@testset "ParcelState struct" begin
    parcel = ParcelState(293.15, 1.01325e5, 0.01, 0.0)
    @test parcel.T == 293.15
    @test parcel.p == 1.01325e5
    @test parcel.qv == 0.01
    @test parcel.S == 0.0
end

@testset "Parcel drift with no condensation" begin
    parcel = ParcelState(293.15, 1.01325e5, 0.01, 0.0)
    w = 1.0  # m/s updraft
    cp = 1005.0
    g = 9.81
    rho_a = 1.225

    # Empty condensation rates
    dm_w = Float64[]
    volume = 1.0e-6

    dparcel = parcel_drift(parcel, dm_w, w, cp, g, rho_a, volume)

    # Adiabatic cooling only
    @test dparcel.T ≈ -g / cp * w atol = 1e-6
    @test dparcel.p ≈ -rho_a * g * w atol = 1e-3
    @test dparcel.qv == 0.0
end

@testset "Parcel drift with condensation" begin
    parcel = ParcelState(293.15, 1.01325e5, 0.01, 0.0)
    w = 1.0
    cp = 1005.0
    g = 9.81
    rho_a = 1.225
    L_v = 2.5e6

    # Two particles condensing 1e-15 kg/s each
    dm_w = [1.0e-15, 1.0e-15]
    volume = 1.0e-6

    dparcel = parcel_drift(parcel, dm_w, w, cp, g, rho_a, volume)

    # Latent heating should oppose adiabatic cooling
    total_condensation = sum(dm_w)
    latent_heat = L_v * total_condensation / (rho_a * cp * volume)
    expected_dT = -g / cp * w + latent_heat
    @test dparcel.T ≈ expected_dT atol = 1e-10

    # Moisture depletion
    expected_dqv = -total_condensation / (rho_a * volume)
    @test dparcel.qv ≈ expected_dqv atol = 1e-10
end
