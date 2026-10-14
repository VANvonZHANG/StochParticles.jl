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

@testset "critical_point: kappa=0 no-peak curve is safe" begin
    T = 285.0
    thermo_bc = ThermodynamicsParams(SVector(0.0, 0.0), 0.072, 1000.0,
        18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)      # κ=0 干槽（纯 Kelvin，无峰）
    dens_bc = SVector(1800.0, 1000.0)
    m_bc = 4.0 / 3.0 * π * (80.0e-9)^3 * 1800.0
    m_dry = SVector(m_bc, 0.0)
    Sc, D_crit = critical_point(m_dry, thermo_bc, dens_bc, T)
    @test isfinite(Sc) && isfinite(D_crit)   # pre_equilibrate!/诊断层依赖此鲁棒性
end
@testset "flux guard: non-physical trial state returns zero" begin
    f = H2OCondensationFlux(THERMO2, 2, DENS2, 0.0)
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
    p_sat = saturation_vapor_pressure(285.0)
    env = SVector(285.0, p_sat * 1.004, 9.0e4)
    @test f(SVector(-1e-18, 1e-19), env, nothing, 0.0) == SVector(0.0, 0.0)
    @test f(SVector(m_as, -1e-18), env, nothing, 0.0) == SVector(0.0, 0.0)
    @test f(SVector(0.0, 0.0), env, nothing, 0.0) == SVector(0.0, 0.0)  # V_dry==0 case
end

@testset "kinetic flux — no gate, haze participates in budget" begin
    T = 285.0
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
    m_dry = SVector(m_as, 0.0)
    p_sat = saturation_vapor_pressure(T)
    f_ki = H2OCondensationFlux(THERMO2, 2, DENS2, 0.0)
    # haze at its S=-0.1% equilibrium, ambient S=+0.05%: kinetic MUST
    # grow (positive flux) — the haze swarm is a real vapor sink
    m_haze = equilibrium_water_mass(m_dry, THERMO2, DENS2, T, p_sat * 0.999)
    μ_haze = SVector(m_as, m_haze)
    env = SVector(T, p_sat * 1.0005, 9.0e4)
    @test f_ki(μ_haze, env, nothing, 0.0)[2] > 0.0
    # same ambient, haze above its equilibrium: evaporates
    env_dry = SVector(T, p_sat * 0.995, 9.0e4)
    @test f_ki(μ_haze, env_dry, nothing, 0.0)[2] < 0.0
end
@testset "accepted-state clamp: fast evaporation at loose tolerance" begin
    # a big wet droplet under deep subsaturation evaporates fast; with loose
    # rtol the integrator can accept a slightly-negative water endpoint —
    # solve_split must clamp it instead of letting diagnostics DomainError
    T = 285.0
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
    m_w0 = 4.0 / 3.0 * π * (2.0e-6)^3 * 1000.0 - m_as
    particles = [SVector(m_as, m_w0)]
    pp = ParcelProcess(0.5, 2, 1.0e-10, THERMO2, T, 9.0e4, 0.0096)
    cond = H2OCondensationProcess(THERMO2, DENS2; h2o_idx = 2, w = 0.0)
    rec = (t,
        u,
        sys) -> (t = t,
        w = get_particle(u, 1, Val(2))[2],
        d = particle_wet_radius(SVector(u[1], 0.0), max(u[2], 0.0), DENS2))
    sol,
    recs = solve_split(particles, 1.0e-12, ParcelCoupled(pp.parcel),
        (cond, pp), Tsit5(); tspan = (0.0, 60.0), n_sim = 1,
        dt_split = 1.0, saveat = 1.0, record_func = rec,
        abstol = 1.0e-24, reltol = 1.0e-4)
    @test all(r.w >= 0.0 for r in recs)      # clamped, never negative
    @test sol.retcode == ReturnCode.Success
end

@testset "fuchs transition factor: limits and pyrcel parity" begin
    T = 285.0
    Mw = 18.015e-3
    Dv = 2.5e-5
    lam = Dv * sqrt(2 * pi * Mw / (8.314 * T))   # primer convention
    # continuum limit: huge droplet -> factor -> 1
    @test fuchs_transition_factor(1.0e-3, T, Dv, Mw) > 0.99
    # transition regime: 100nm radius -> Kn' ~ 1.74, factor ~ 0.365
    f100 = fuchs_transition_factor(1.0e-7, T, Dv, Mw)
    @test 0.30 < f100 < 0.45
    # free-molecular scaling: factor ∝ R for small R
    # free-molecular scaling approached: factor ∝ R at small R (Kn' ~ 17 and
    # ~8.6 here still carry the +1 correction; exact ratio 2(1+Kn)/(1+2Kn))
    @test 1.85 <
          fuchs_transition_factor(2e-8, T, Dv, Mw) /
          fuchs_transition_factor(1e-8, T, Dv, Mw) < 2.0
end

@testset "saturation_vapor_pressure: Buck-1996 vs reference points" begin
    # Buck (1981, rev. 1996) coefficients {611.21, 18.678, 234.5, 257.14};
    # audit 2026-10-08: replaces constant-L_v C-C form (which ran −0.6% at 285 K)
    @test saturation_vapor_pressure(273.15) ≈ 611.2 atol = 0.5
    @test saturation_vapor_pressure(285.0) ≈ 1388.6 rtol = 5e-4
    @test saturation_vapor_pressure(290.0) ≈ 1919.3 rtol = 5e-4
    # monotonic and convex in T
    es = [saturation_vapor_pressure(T) for T in 270.0:5.0:300.0]
    @test all(diff(es) .> 0)
    @test all(diff(log.(es)) .> 0)
end
