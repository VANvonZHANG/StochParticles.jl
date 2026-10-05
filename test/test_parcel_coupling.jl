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

# ---- M3-3/M3-4: closed-loop physics integration (spec §4) ----

const DENS5 = SVector(1770.0, 1720.0, 1400.0, 1800.0, 1000.0)
const THERMO5 = ThermodynamicsParams(SVector(0.61, 0.67, 0.10, 0.0, 0.0),
    0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)

_m3_edges() = collect(10.0 .^ range(-8.3, -4.5; length = 96))

function _m3_population(n_sim, seed, N_total)
    tbl = lognormal_table(6.0e-8, 1.45, 8.0e11, _m3_edges()) +
          lognormal_table(1.6e-7, 1.55, 3.2e11, _m3_edges())
    fb = SizeResolvedComposition(tbl; anchors = [
        (3.0e-8, SVector(0.18, 0.15, 0.57, 0.10)),
        (3.0e-7, SVector(0.35, 0.20, 0.30, 0.15))])
    spec = SyntheticPopulationSpec(n_sim = n_sim, spectrum = tbl, fbar = fb,
        chi_target = 0.5, densities = DENS5, h2o_idx = 5, chi_species = [1, 2, 3],
        T0 = 285.0, S0 = -0.002)
    return synthesize_population(spec; seed = seed, thermo = THERMO5, nu = 5.0)
end

function _m3_mono_population(n_sim, seed, dg, sg, N)
    tbl = lognormal_table(dg, sg, N, _m3_edges())
    fb = SizeResolvedComposition(tbl; anchors = [
        (3.0e-8, SVector(0.18, 0.15, 0.57, 0.10)),
        (3.0e-7, SVector(0.35, 0.20, 0.30, 0.15))])
    spec = SyntheticPopulationSpec(n_sim = n_sim, spectrum = tbl, fbar = fb,
        chi_target = 0.5, densities = DENS5, h2o_idx = 5, chi_species = [1, 2, 3],
        T0 = 285.0, S0 = -0.002)
    return synthesize_population(spec; seed = seed, thermo = THERMO5, nu = 5.0)[1]
end

function _closed_loop_run(particles, n_sim, N_total; gate = :sc_threshold)
    V = n_sim / N_total
    T0, p0 = 285.0, 9.0e4
    m_air = p0 / (StochParticles.R_DRY_AIR * T0) * V
    qv0 = 0.998 * StochParticles.EPSILON_MA * saturation_vapor_pressure(T0) / p0
    pp = ParcelProcess(0.5, 5, m_air, THERMO5, T0, p0, qv0)
    cond = H2OCondensationProcess(THERMO5, DENS5; h2o_idx = 5, w = 0.0,
        activation_gate = gate)
    record = (t, u, sys) -> (
        t = t,
        L_water = sum(get_particle(u, i, Val(5))[5] for i in 1:sys.n_active) +
                  m_air * u[sys.n_sim * 5 + 3],
        act_frac = activation_fraction(u, sys, Val(5); mode = :radius_threshold,
            threshold = 1.0e-6, densities = DENS5),
        S = parcel_supersaturation(extract_parcel(u, sys.n_sim, 5)),
    )
    return solve_split(particles, V, ParcelCoupled(pp.parcel), (cond, pp), Tsit5();
        tspan = (0.0, 600.0), n_sim = n_sim, dt_split = 10.0, saveat = 60.0,
        record_func = record, abstol = 1.0e-24, reltol = 1.0e-5)
end

@testset "M3-3: water conservation (linear invariant, RK-exact)" begin
    particles, _, _ = _m3_population(200, 101, 1.12e12)
    sol, records = _closed_loop_run(particles, 200, 1.12e12)
    @test sol.retcode == ReturnCode.Success
    L0 = records[1].L_water
    worst = maximum(abs(r.L_water - L0) / L0 for r in records)
    @test worst < 1.0e-10   # L = Σm_w + m_air·qv 线性不变量；RHS 恒满足 → RK 精确保持
    println("M3-3 worst relative drift = $worst")
end

@testset "M3-4: competition signature" begin
    n_sim = 300
    particles_bi, _, _ = _m3_population(n_sim, 202, 1.12e12)
    _, rec_bi = _closed_loop_run(particles_bi, n_sim, 1.12e12)
    _, rec_a = _closed_loop_run(
        _m3_mono_population(n_sim, 203, 6.0e-8, 1.45, 8.0e11), n_sim, 8.0e11)
    _, rec_c = _closed_loop_run(
        _m3_mono_population(n_sim, 204, 1.6e-7, 1.55, 3.2e11), n_sim, 3.2e11)
    Nd_bi = rec_bi[end].act_frac * 1.12e12
    Nd_a = rec_a[end].act_frac * 8.0e11
    Nd_c = rec_c[end].act_frac * 3.2e11
    println("M3-4 Nd: bi=$Nd_bi a=$Nd_a c=$Nd_c (sum_mono=$(Nd_a + Nd_c))")
    @test Nd_bi < Nd_a + Nd_c                       # 竞争：双模态 < 独立之和
    @test (Nd_a + Nd_c - Nd_bi) / (Nd_a + Nd_c) > 0.01
    S_max_bi = maximum(r.S for r in rec_bi)
    S_max_a = maximum(r.S for r in rec_a)
    S_max_c = maximum(r.S for r in rec_c)
    println("M3-4 S_max: bi=$S_max_bi aitken_only=$S_max_a accum_only=$S_max_c")
    @test S_max_bi < S_max_a   # 往 Aitken 里加大 CCN 积聚模 → 峰前汇增强 → S_max 压制
                              #（bi vs 仅C 添加的是小 CCN：门冻结其 haze 吸湿、活化在 S_max 之后，压不了峰）
end

@testset "reequilibrate_haze! adjusts haze and conserves water" begin
    n_sim, A = 2, 2
    sys = ParticleSystem(Val(A), n_sim, 1.0e-12, t -> SVector(285.0, 1000.0))
    T, p_sat = 285.0, saturation_vapor_pressure(285.0)
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
    m_dry = SVector(m_as, 0.0)                   # 100nm AS → Sc ≈ 0.16%
    m_w_eq = equilibrium_water_mass(m_dry, THERMO2, DENS2, T, p_sat * 0.999)  # haze @ S=-0.1%
    u = zeros(n_sim * A + 3)
    set_particle!(u, 1, Val(A), SVector(m_as, m_w_eq))
    set_particle!(u, 2, Val(A), SVector(m_as, m_w_eq))
    m_air = 1.0e-10
    u[n_sim*A+1] = T; u[n_sim*A+2] = 9.0e4; u[n_sim*A+3] = 0.0096
    L_before = u[2] + u[4] + m_air * u[7]
    # S 升到 +0.1%（仍 < Sc=0.16% → 两颗粒均在霾支）：haze 吸水到新平衡
    delta = reequilibrate_haze!(u, sys, THERMO2, DENS2; h2o_idx = 2,
        T = T, S = 0.001, m_air = m_air)
    @test delta > 0.0
    m_w_new = equilibrium_water_mass(m_dry, THERMO2, DENS2, T, p_sat * 1.001)
    @test u[2] ≈ m_w_new rtol = 1e-3
    L_after = u[2] + u[4] + m_air * u[7]
    @test L_after ≈ L_before rtol = 1e-12         # 守恒强制（qv 回写）
end
