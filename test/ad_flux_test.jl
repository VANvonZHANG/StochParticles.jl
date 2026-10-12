using StochParticles
using StaticArrays
using ForwardDiff
using Test

# PR1 (M4 ki-enabler) acceptance: the flux layer and thermodynamic helpers
# accept Dual numbers and differentiate EXACTLY (against central finite
# differences). Full-RHS AD transparency — the parcel tail path — arrives
# with the env pass-through and parcel_drift-purity PRs; here we verify the
# layers those PRs build on are already dual-transparent.

const THERMO_AD = ThermodynamicsParams(SVector(0.61, 0.0), 0.072, 1000.0,
    18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)
const DENS_AD = SVector(1770.0, 1000.0)
const M_AS_AD = 4.0 / 3.0 * π * (50.0e-9)^3 * 1770.0
const M_W_AD = 4.0 / 3.0 * π * (80.0e-9)^3 * 1000.0 - M_AS_AD

@testset "thermo helpers differentiate exactly" begin
    h = 1e-4
    @test ForwardDiff.derivative(saturation_vapor_pressure, 285.0) ≈
          (saturation_vapor_pressure(285.0 + h) - saturation_vapor_pressure(285.0 - h)) /
          (2h) rtol = 1e-6
    @test ForwardDiff.derivative(R -> fuchs_transition_factor(R, 285.0, 2.5e-5, 18.015e-3),
        1e-7) ≈
          (fuchs_transition_factor(1e-7 * (1 + 1e-5), 285.0, 2.5e-5, 18.015e-3) -
           fuchs_transition_factor(1e-7 * (1 - 1e-5), 285.0, 2.5e-5, 18.015e-3)) /
          (2e-5 * 1e-7) rtol = 1e-6
end

@testset "H2O flux ∂dm/dt/∂m_w: kinetic (smooth) and active-gate regimes" begin
    T = 285.0
    p_sat = saturation_vapor_pressure(T)
    env = SVector(T, p_sat * 1.0004)
    f_ki = H2OCondensationFlux(THERMO_AD, 2, DENS_AD, 0.0, :kinetic)
    fdk(mw) = f_ki(SVector(M_AS_AD, mw), env, nothing, 0.0)[2]
    h = 1e-6 * M_W_AD
    @test ForwardDiff.derivative(fdk, M_W_AD) ≈
          (fdk(M_W_AD + h) - fdk(M_W_AD - h)) / (2h) rtol = 1e-5

    # gated mode in the ACTIVE region (S_env > Sc) is flux-identical to
    # kinetic — its derivative must match too
    Sc, _ = critical_point(SVector(M_AS_AD, 0.0), THERMO_AD, DENS_AD, T)
    env_act = SVector(T, p_sat * (1 + Sc + 1e-3))
    f_sc = H2OCondensationFlux(THERMO_AD, 2, DENS_AD, 0.0, :sc_threshold)
    fda(mw) = f_sc(SVector(M_AS_AD, mw), env_act, nothing, 0.0)[2]
    @test ForwardDiff.derivative(fda, M_W_AD) ≈
          (fda(M_W_AD + h) - fda(M_W_AD - h)) / (2h) rtol = 1e-5
end

@testset "H2O flux gradient w.r.t. env (T, p_v) — the channel the box truncates" begin
    T = 285.0
    p_sat = saturation_vapor_pressure(T)
    env = SVector(T, p_sat * 1.0004)
    f_ki = H2OCondensationFlux(THERMO_AD, 2, DENS_AD, 0.0, :kinetic)
    ge(e) = f_ki(SVector(M_AS_AD, M_W_AD), e, nothing, 0.0)[2]
    grad = ForwardDiff.gradient(ge, env)
    hT, hP = 1e-4, 1e-2
    @test grad[1] ≈ (ge(SVector(T + hT, env[2])) - ge(SVector(T - hT, env[2]))) / (2hT) rtol = 1e-5
    @test grad[2] ≈
          (ge(SVector(env[1], env[2] + hP)) - ge(SVector(env[1], env[2] - hP))) /
          (2hP) rtol = 1e-5
end

@testset "drift interface passes Duals through unchanged" begin
    T = 285.0
    p_sat = saturation_vapor_pressure(T)
    env = SVector(T, p_sat * 1.0004)
    cond = H2OCondensationProcess(THERMO_AD, DENS_AD; h2o_idx = 2, w = 0.0,
        activation_gate = :kinetic)
    sys = ParticleSystem(Val(2), 1, 1.0, t -> env)
    μ0 = SVector(M_AS_AD, M_W_AD)
    @inferred apply_drift(cond, μ0, sys.gas_phase(0.0), 0.0)  # type-stable
    via_drift = ForwardDiff.derivative(
        mw -> apply_drift(cond, SVector(M_AS_AD, mw), env, 0.0)[2], M_W_AD)
    f_ki = H2OCondensationFlux(THERMO_AD, 2, DENS_AD, 0.0, :kinetic)
    direct = ForwardDiff.derivative(
        mw -> f_ki(SVector(M_AS_AD, mw), env, nothing, 0.0)[2], M_W_AD)
    @test via_drift == direct                           # bit-equal: same code path
end
