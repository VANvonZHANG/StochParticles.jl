using StochParticles
using StaticArrays
using Test
using OrdinaryDiffEq

const THERMO2 = ThermodynamicsParams(SVector(0.61, 0.0), 0.072, 1000.0,
    18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)
const DENS2 = SVector(1770.0, 1000.0)

@testset "M3-1: dry adiabatic parcel vs closed form" begin
    # 无凝结过程（processes = (parcel,)），闭式解：T = T0 − Γ·w·t，
    # p = p0·(T/T0)^{cp/R_d}，qv 恒定。n_sim=1, A=2 → 尾部 = u[3:5]
    T0, p0, w = 285.0, 9.0e4, 0.5
    qv0 = StochParticles.EPSILON_MA * saturation_vapor_pressure(T0) / p0 * 0.998
    V = 1.0e-6
    m_air = p0 / (StochParticles.R_DRY_AIR * T0) * V
    pp = ParcelProcess(w, 2, m_air, THERMO2, T0, p0, qv0)
    dummy = [SVector(1.0e-18, 0.0)]
    sol, records = solve_split(dummy, V, ParcelCoupled(pp.parcel), (pp,), Tsit5();
        tspan = (0.0, 600.0), n_sim = 1, dt_split = 10.0, saveat = 60.0,
        record_func = (t, u, sys) -> (t = t, T = u[3], p = u[4], qv = u[5]),
        abstol = 1.0e-10, reltol = 1.0e-8)
    Γ = 9.81 / StochParticles.CP_DRY_AIR
    for r in records
        T_an = T0 - Γ * w * r.t
        p_an = p0 * (T_an / T0)^(StochParticles.CP_DRY_AIR / StochParticles.R_DRY_AIR)
        @test r.T ≈ T_an rtol = 1e-5
        @test r.p ≈ p_an rtol = 1e-5
        @test r.qv ≈ qv0 atol = 1e-14
    end
end

@testset "parcel tail isolation" begin
    n_sim, A = 3, 2
    u = zeros(n_sim * A + 3)
    u[1:6] .= 1.0e-18
    set_parcel_state!(u, ParcelState(285.0, 9.0e4, 0.01), n_sim, A)
    @test total_mass(u, Val(A), n_sim) ≈ 6.0e-18   # total_mass 不吞尾部
    μ = get_particle(u, 3, Val(A))                 # get/set 不越界
    @test μ == SVector(1.0e-18, 1.0e-18)
end

@testset "ParcelCoupled semantics" begin
    T = 285.0
    p_sat = saturation_vapor_pressure(T)
    pr = ParcelState(T, 9.0e4, 0.998 * StochParticles.EPSILON_MA * p_sat / 9.0e4)
    env = ParcelCoupled(Ref(pr))
    g0 = env(0.0)
    @test g0 == env(123.0)                          # 忽略 t
    @test g0 isa SVector{2, Float64}
    @test g0[1] == T
    @test g0[2] ≈ p_sat * (1.0 + parcel_supersaturation(pr)) rtol = 1e-12
    pr.T = 280.0                                    # 活状态随 Ref 变化
    @test env(0.0)[1] == 280.0
end

@testset "du length and RHS layout with parcel process" begin
    # n_sim=2, A=2 → 颗粒 u[1:4]，尾部 du[5:7] = (dT, dp, dqv)
    T0, p0, w = 285.0, 9.0e4, 0.5
    V = 1.0e-6
    m_air = p0 / (StochParticles.R_DRY_AIR * T0) * V
    pp = ParcelProcess(w, 2, m_air, THERMO2, T0, p0, 0.0096)
    particles = [SVector(1.0e-18, 0.0) for _ in 1:2]
    sys = ParticleSystem(Val(2), 2, V, ParcelCoupled(pp.parcel))
    u0 = make_u0(particles)
    append!(u0, (T0, p0, 0.0096))
    f = make_ode_func((pp,))
    du = zeros(length(u0))
    f(du, u0, sys, 0.0)
    @test length(du) == 7
    @test du[1:4] == zeros(4)                       # 无凝结进程 → 颗粒漂移全零
    @test du[5] ≈ -9.81 / StochParticles.CP_DRY_AIR * w atol = 1e-12      # dT
    @test du[6] ≈ -(p0 / (StochParticles.R_DRY_AIR * T0)) * 9.81 * w atol = 1e-9  # dp
    @test du[7] == 0.0                              # dqv（无凝结）
end
