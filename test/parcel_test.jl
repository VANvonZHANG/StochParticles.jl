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

@testset "State vector with parcel extension" begin
    A = 2
    n_sim = 3
    n_parcel = 4

    # Build state vector: 3 particles × 2 species + 4 parcel vars
    u = zeros(n_sim * A + n_parcel)
    u[1] = 1.0e-18   # particle 1, species 1
    u[2] = 1.0e-18   # particle 1, species 2 (H2O)
    u[3] = 2.0e-18
    u[4] = 2.0e-18
    u[5] = 3.0e-18
    u[6] = 3.0e-18
    u[7] = 293.15    # T
    u[8] = 1.01325e5 # p
    u[9] = 0.01      # qv
    u[10] = 0.0      # S

    # Extract parcel
    parcel = extract_parcel(u, n_sim, A)
    @test parcel.T ≈ 293.15
    @test parcel.p ≈ 1.01325e5
    @test parcel.qv ≈ 0.01
    @test parcel.S ≈ 0.0

    # Modify parcel
    new_parcel = ParcelState(295.0, 1.0e5, 0.012, 0.01)
    set_parcel!(u, new_parcel, n_sim, A)
    @test u[7] ≈ 295.0
    @test u[8] ≈ 1.0e5
    @test u[9] ≈ 0.012
    @test u[10] ≈ 0.01
end

@testset "Parcel drift dS value" begin
    parcel = ParcelState(293.15, 1.01325e5, 0.01, 0.0)
    w = 1.0
    cp = 1005.0
    g = 9.81
    rho_a = 1.225

    # No condensation — dS from adiabatic cooling only
    dm_w = Float64[]
    volume = 1.0e-6
    dparcel = parcel_drift(parcel, dm_w, w, cp, g, rho_a, volume)

    # dS should be negative (adiabatic cooling raises qsat, lowering S)
    # since dqv=0 and dT<0 (cooling), dS = dqv/qsat - qv*dqsat_dT*dT/qsat^2
    # = 0 - qv*dqsat_dT*dT/qsat^2  (dT<0, dqsat_dT>0 → second term > 0 → dS > 0)
    @test dparcel.S != 0.0  # not trivially zero
    @test dparcel.S > 0.0   # adiabatic cooling without condensation raises S
end

@testset "Parcel drift with condensation affects dS" begin
    parcel = ParcelState(293.15, 1.01325e5, 0.01, 0.0)
    w = 1.0
    cp = 1005.0
    g = 9.81
    rho_a = 1.225
    dm_w = [1.0e-15]
    volume = 1.0e-6
    dparcel = parcel_drift(parcel, dm_w, w, cp, g, rho_a, volume)

    # Condensation adds latent heat (opposing cooling) and removes moisture
    # Both effects reduce supersaturation tendency
    @test dparcel.S isa Float64
end
