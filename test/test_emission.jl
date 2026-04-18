# test/test_emission.jl
using StochParticles
using Test
using StaticArrays

@testset "EmissionProcess" begin
    @testset "emission jump adds a particle" begin
        rate_density = (μ, t) -> 1.0
        sampler = (t) -> SVector(1.0e-18)
        proc = EmissionProcess(1.0, sampler)

        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(1), 10, 1.0, gas_fn)  # n_sim=10, 10 slots
        u0 = fill(0.0, 10)
        for i in 1:5
            set_particle!(u0, i, Val(1), SVector(2.0))
        end
        sys.n_active = 5  # only 5 active particles initially

        mock = MockIntegrator(u0, sys, 0.0)
        make_emission_jump(proc).affect!(mock)

        @test sys.n_active == 6
        @test get_particle(mock.u, 6, Val(1)) == SVector(1.0e-18)
    end

    @testset "emission respects n_sim limit" begin
        rate_density = (μ, t) -> 1.0
        sampler = (t) -> SVector(1.0e-18)
        proc = EmissionProcess(1.0, sampler)

        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(1), 5, 1.0, gas_fn)
        u0 = fill(0.0, 5)  # already at n_sim
        sys.n_active = 5

        mock = MockIntegrator(u0, sys, 0.0)
        make_emission_jump(proc).affect!(mock)
        @test sys.n_active == 5
    end
end
