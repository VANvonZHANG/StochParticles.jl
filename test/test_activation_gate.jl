using StochParticles
using StaticArrays
using Test
using OrdinaryDiffEq

const THERMO2 = ThermodynamicsParams(SVector(0.61, 0.0), 0.072, 1000.0,
    18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)
const DENS2 = SVector(1770.0, 1000.0)

@testset "critical_point returns (Sc, D_crit)" begin
    T = 285.0
    r_dry = 50.0e-9
    m_as = 4.0 / 3.0 * π * r_dry^3 * 1770.0
    m_dry = SVector(m_as, 0.0)
    Sc, D_crit = critical_point(m_dry, THERMO2, DENS2, T)
    A_kelvin = 2.0 * 0.072 / (461.5 * T * 1000.0)
    Sc_analytic = sqrt(4.0 * A_kelvin^3 / (27.0 * 0.61 * r_dry^3))
    @test Sc ≈ Sc_analytic rtol = 0.15   # exact Köhler vs κ-approx, 15% 容差
    @test D_crit > 2.0 * r_dry           # 峰在湿径 > 干径处
    # 与 critical_supersaturation 一致（薄包装）
    @test critical_supersaturation(m_dry, THERMO2, DENS2, T) == Sc
    # D_crit 是曲线峰：略小于/大于 D_crit 的湿径其平衡 S 均不超过 Sc
    m_w(R) = 4.0 / 3.0 * π * R^3 * 1000.0 - m_as
    for scale in (0.7, 1.3)
        p_eq = equilibrium_vapor_pressure(m_dry, m_w(0.5 * scale * D_crit),
            THERMO2, DENS2, T)
        @test p_eq / saturation_vapor_pressure(T) - 1.0 <= Sc + 1e-12
    end
end

@testset "activation_gate: 2x3 unit matrix" begin
    T = 285.0
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
    m_dry = SVector(m_as, 0.0)
    Sc, D_crit = critical_point(m_dry, THERMO2, DENS2, T)
    R_wet = 1.5 * D_crit                       # 过峰湿径
    m_w = 4.0 / 3.0 * π * R_wet^3 * 1000.0 - m_as
    μ = SVector(m_as, m_w)
    p_sat = saturation_vapor_pressure(T)

    # 情形1：S_env ∈ (S_eq(D_wet), Sc) —— 过峰滴应继续生长
    S_eq = equilibrium_vapor_pressure(m_dry, m_w, THERMO2, DENS2, T) / p_sat - 1.0
    S_mid = 0.5 * (S_eq + Sc)
    env = SVector(T, p_sat * (1.0 + S_mid))
    f_sc = H2OCondensationFlux(THERMO2, 2, DENS2, 0.0, :sc_threshold)
    f_ba = H2OCondensationFlux(THERMO2, 2, DENS2, 0.0, :branch_aware)
    @test f_sc(μ, env, nothing, 0.0) == zero(μ)              # 门冻结
    @test f_ba(μ, env, nothing, 0.0)[2] > 0.0                # 分支感知：正通量

    # 情形2：S_env < S_eq(D_wet) —— 过峰滴应蒸发（branch 版负通量）
    S_low = S_eq - 0.5 * (S_eq + 1.0e-3)                     # 深落
    env2 = SVector(T, p_sat * (1.0 + S_low))
    @test f_ba(μ, env2, nothing, 0.0)[2] < 0.0
    @test f_sc(μ, env2, nothing, 0.0) == zero(μ)

    # 情形3：未活化（D_wet < D_crit）—— 两模式一致零通量
    m_w_haze = equilibrium_water_mass(m_dry, THERMO2, DENS2, T, p_sat * (1.0 + S_mid))
    μ_haze = SVector(m_as, m_w_haze)
    @test f_ba(μ_haze, env, nothing, 0.0) == zero(μ)
    @test f_sc(μ_haze, env, nothing, 0.0) == zero(μ)
end

@testset "activation_gate: continuity and S_env > Sc identity" begin
    T = 285.0
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
    m_dry = SVector(m_as, 0.0)
    Sc, D_crit = critical_point(m_dry, THERMO2, DENS2, T)
    p_sat = saturation_vapor_pressure(T)
    R_wet = 1.5 * D_crit
    m_w = 4.0 / 3.0 * π * R_wet^3 * 1000.0 - m_as
    μ = SVector(m_as, m_w)
    f_sc = H2OCondensationFlux(THERMO2, 2, DENS2, 0.0, :sc_threshold)
    f_ba = H2OCondensationFlux(THERMO2, 2, DENS2, 0.0, :branch_aware)
    env_above = SVector(T, p_sat * (1.0 + Sc + 1.0e-4))
    @test f_sc(μ, env_above, nothing, 0.0) == f_ba(μ, env_above, nothing, 0.0)
    # 穿峰连续性：D_wet 跨 D_crit（S_env 恰在 Sc 之上），两模式输出连续且无 NaN
    for scale in (0.98, 1.02)
        R = scale * 0.5 * D_crit
        mw = 4.0 / 3.0 * π * R^3 * 1000.0 - m_as
        v = f_ba(SVector(m_as, mw), SVector(T, p_sat * (1.0 + Sc + 1.0e-6)),
            nothing, 0.0)
        @test !isnan(v[2]) && !isinf(v[2])
    end
    @test_throws ArgumentError H2OCondensationFlux(THERMO2, 2, DENS2, 0.0, :bogus)
end

@testset "activation_gate: kappa=0 no-peak curve is safe" begin
    T = 285.0
    thermo_bc = ThermodynamicsParams(SVector(0.0, 0.0), 0.072, 1000.0,
        18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)      # κ=0 干槽（纯 Kelvin，无峰）
    dens_bc = SVector(1800.0, 1000.0)
    m_bc = 4.0 / 3.0 * π * (80.0e-9)^3 * 1800.0
    m_dry = SVector(m_bc, 0.0)
    Sc, D_crit = critical_point(m_dry, thermo_bc, dens_bc, T)
    @test isfinite(Sc) && isfinite(D_crit)
    f_ba = H2OCondensationFlux(thermo_bc, 2, dens_bc, 0.0, :branch_aware)
    p_sat = saturation_vapor_pressure(T)
    μ = SVector(m_bc, 0.0)                           # 完全干 BC（水量恰零）
    @test f_ba(μ, SVector(T, p_sat * 1.0005), nothing, 0.0) == zero(μ)  # 永冻 haze
end

@testset "branch_aware hysteresis: open-loop S rise-fall" begin
    T = 285.0
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0   # 100nm AS → Sc ≈ 0.16%
    particles = [SVector(m_as, 0.0)]
    pre_equilibrate!(particles, THERMO2, DENS2, T,
        saturation_vapor_pressure(T); h2o_idx = 2)
    env = PrescribedProfile([0.0, 100.0, 600.0], [T, T, T],
        [0.0, 0.005, -0.005])                      # 爬到 0.5% 再线性落到 −0.5%
    wet_D(u) = 2.0 * particle_wet_radius(SVector(u[1], 0.0), u[2], DENS2)
    results = Dict{Symbol, Vector{Float64}}()
    for gate in (:sc_threshold, :branch_aware)
        cond = H2OCondensationProcess(THERMO2, DENS2; h2o_idx = 2, w = 0.0,
            activation_gate = gate)
        _, recs = solve_split(deepcopy(particles), 1.0e-15, env, (cond,), Tsit5();
            tspan = (0.0, 600.0), n_sim = 1, dt_split = 5.0, saveat = 30.0,
            record_func = (t, u, sys) -> (t = t, D = wet_D(u)),
            abstol = 1.0e-24, reltol = 1.0e-8)
        results[gate] = [r.D for r in recs]
    end
    sc_path, ba_path = results[:sc_threshold], results[:branch_aware]
    println("hysteresis: sc peak=$(round(maximum(sc_path)*1e9,digits=1))nm end=$(round(sc_path[end]*1e9,digits=1))nm | ba peak=$(round(maximum(ba_path)*1e9,digits=1))nm end=$(round(ba_path[end]*1e9,digits=1))nm")
    @test maximum(ba_path) > 5.0e-7                # 两模式都活化（S_max=0.5% ≫ Sc）
    @test maximum(sc_path) > 5.0e-7
    @test ba_path[end] < maximum(ba_path)          # branch 版深落后蒸发（退回霾支）
    @test sc_path[end] > ba_path[end]              # 冻结版停在高位、branch 版缩回
end
