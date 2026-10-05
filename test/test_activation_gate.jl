using StochParticles
using StaticArrays
using Test

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
