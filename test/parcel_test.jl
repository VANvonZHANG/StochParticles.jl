using StochParticles
using StaticArrays
using Test

const THERMO2 = ThermodynamicsParams(SVector(0.61, 0.0), 0.072, 1000.0,
    18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)

@testset "ParcelState 3-field" begin
    parcel = ParcelState(293.15, 1.01325e5, 0.01)
    @test parcel.T == 293.15
    @test parcel.p == 1.01325e5
    @test parcel.qv == 0.01
    @test_throws MethodError ParcelState(293.15, 1.01325e5, 0.01, 0.0)
end

@testset "parcel_supersaturation diagnosis" begin
    T = 285.0
    p_sat = saturation_vapor_pressure(T)
    sat = ParcelState(T, 9.0e4, StochParticles.EPSILON_MA * p_sat / 9.0e4)
    @test parcel_supersaturation(sat) ≈ 0.0 atol = 1e-12
    sub = ParcelState(T, 9.0e4, 0.998 * StochParticles.EPSILON_MA * p_sat / 9.0e4)
    @test parcel_supersaturation(sub) ≈ -0.002 rtol = 1e-6
end

@testset "parcel_drift: dry adiabatic + diagnosed rho" begin
    T, p0 = 285.0, 9.0e4
    parcel = ParcelState(T, p0, 0.0096)
    m_air, w = 2.0e-10, 0.5
    d = parcel_drift(parcel, 0.0, w, m_air; thermo = THERMO2)
    rho_a = p0 / (StochParticles.R_DRY_AIR * T)          # 诊断密度，非 1.225
    @test d isa SVector{3, Float64}
    @test d[1] ≈ -9.81 / StochParticles.CP_DRY_AIR * w atol = 1e-12
    @test d[2] ≈ -rho_a * 9.81 * w atol = 1e-9
    @test d[3] == 0.0
end

@testset "parcel_drift: condensation coupling" begin
    parcel = ParcelState(293.15, 1.01325e5, 0.01)
    m_air, w, rate = 1.0e-9, 1.0, 3.0e-15
    d = parcel_drift(parcel, rate, w, m_air; thermo = THERMO2)
    @test d[1] ≈ -9.81 / StochParticles.CP_DRY_AIR * w +
                 2.5e6 * rate / (m_air * StochParticles.CP_DRY_AIR) atol = 1e-14
    @test d[3] ≈ -rate / m_air atol = 1e-20
end

@testset "extract/set 3-slot tail" begin
    n_sim, A = 3, 2
    u = zeros(n_sim * A + 3)
    u[n_sim*A+1] = 285.0; u[n_sim*A+2] = 9.0e4; u[n_sim*A+3] = 0.0096
    pr = extract_parcel(u, n_sim, A)
    @test pr isa ParcelState
    @test pr.T == 285.0 && pr.p == 9.0e4 && pr.qv == 0.0096
    set_parcel_state!(u, ParcelState(280.0, 8.9e4, 0.009), n_sim, A)
    @test u[7] == 280.0 && u[8] == 8.9e4 && u[9] == 0.009
    du = zeros(n_sim * A + 3)
    set_parcel_drift!(du, SVector(1.0, 2.0, 3.0), n_sim, A)
    @test du[7] == 1.0 && du[8] == 2.0 && du[9] == 3.0
    @test du[1:6] == zeros(6)      # 颗粒槽不受污染
end
