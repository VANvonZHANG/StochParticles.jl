# test/test_condensation.jl
using StochParticles
using Test
using StaticArrays
using OrdinaryDiffEq

@testset "CondensationProcess" begin
    @testset "provides drift" begin
        flux = (μ, g, t) -> -0.1 .* μ  # simple exponential decay
        proc = CondensationProcess(flux)
        @test provides_drift(proc) == true
    end

    @testset "ODE integration with condensation" begin
        # 3 particles, 2 species, decay flux
        flux = (μ, g, t) -> -0.1 .* μ
        gas_fn = t -> SVector(0.0, 0.0)
        proc = CondensationProcess(flux)

        sys = ParticleSystem(Val(2), 3, 1.0, gas_fn)
        particles = [SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)]
        u0 = make_u0(particles)

        ode_func! = make_ode_func((proc,))
        prob = ODEProblem(ode_func!, u0, (0.0, 1.0), sys)
        sol = solve(prob, Tsit5())

        # After 1s with decay rate 0.1: μ(t) = μ(0) * exp(-0.1)
        expected = exp(-0.1)
        @test sol.u[end][1] ≈ 1.0 * expected atol=1e-6
        @test sol.u[end][2] ≈ 2.0 * expected atol=1e-6
        @test sol.u[end][5] ≈ 5.0 * expected atol=1e-6
    end
end
