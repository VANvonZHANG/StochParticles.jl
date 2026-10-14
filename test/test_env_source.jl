using StochParticles
using Test
using StaticArrays
using ForwardDiff

@testset "PrescribedProfile" begin
    src = PrescribedProfile(
        [0.0, 120.0, 600.0], [288.15, 288.15, 288.15], [0.0, 0.004, 0.004])

    # 节点处取精确值
    @test src(0.0)[1] ≈ 288.15
    @test src(0.0)[2] ≈ saturation_vapor_pressure(288.15)
    @test src(120.0)[2] ≈ saturation_vapor_pressure(288.15) * 1.004

    # 节点间线性插值（t=60 → S=0.002）
    @test src(60.0)[2] ≈ saturation_vapor_pressure(288.15) * 1.002

    # 两端平推外插
    @test src(-10.0)[2] ≈ saturation_vapor_pressure(288.15)
    @test src(1.0e4)[2] ≈ saturation_vapor_pressure(288.15) * 1.004

    # p_v single-point-definition invariant + return type (3 slots: T, p_v, p_air)
    T, p_v, p_air = src(60.0)
    @test p_v ≈ saturation_vapor_pressure(T) * (1.0 + 0.002)
    @test p_air == 101325.0
    @test src(60.0) isa SVector{3, Float64}

    # p_air pass-through + constructor validation
    src_p = PrescribedProfile([0.0], [288.15], [0.0], 9.0e4)
    @test src_p(0.0)[3] == 9.0e4
    @test_throws ArgumentError PrescribedProfile([0.0], [288.0], [0.0], -1.0)
    @test_throws ArgumentError PrescribedProfile([0.0, 0.0], [288.0, 288.0], [0.0, 0.0])
    @test_throws ArgumentError PrescribedProfile([0.0], [288.0, 288.0], [0.0])
end

@testset "parcel_env ≡ ParcelCoupled (env direct pass-through)" begin
    # same formulas, bit-exact: the RHS now diagnoses g from the tail via
    # parcel_env instead of reading the Ref box through ParcelCoupled
    for (
        T, p, qv) in ((285.0, 9.0e4, 0.0096), (283.2, 8.5e4, 0.0075),
        (290.0, 1.0e5, 0.012))
        ref = ParcelCoupled(Ref(ParcelState(T, p, qv)))(0.0)
        @test parcel_env(T, p, qv) == ref
    end
    # eltype flows — the property the Float64-fixed box lacked
    d = ForwardDiff.derivative(qv -> parcel_env(285.0, 9.0e4, qv)[2], 0.0096)
    h = 1e-8
    @test d ≈
          (parcel_env(285.0, 9.0e4, 0.0096 + h)[2] -
           parcel_env(285.0, 9.0e4, 0.0096 - h)[2]) / (2h) rtol = 1e-5
end

@testset "apply_drift g-form (single interface form)" begin
    # drift processes take the PRECOMPUTED environment; the legacy
    # (proc, μ, sys, t) form was removed (dispatch ambiguities with
    # per-process g-form methods — see interface.jl)
    T = 285.0
    p_sat = saturation_vapor_pressure(T)
    env = SVector(T, p_sat * 1.0004, 9.0e4)
    thermo = ThermodynamicsParams(SVector(0.61, 0.0), 0.072, 1000.0,
        18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)
    cond = H2OCondensationProcess(thermo, SVector(1770.0, 1000.0);
        h2o_idx = 2, w = 0.0, activation_gate = :kinetic)
    sys = ParticleSystem(Val(2), 1, 1.0, t -> env)
    m_as = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
    m_w = 4.0 / 3.0 * π * (80.0e-9)^3 * 1000.0 - m_as
    μ0 = SVector(m_as, m_w)
    d_dir = @inferred apply_drift(cond, μ0, env, 0.0)
    # callers holding a ParticleSystem evaluate g themselves
    d_via_sys = @inferred apply_drift(cond, μ0, sys.gas_phase(0.0), 0.0)
    @test d_via_sys == d_dir
end
