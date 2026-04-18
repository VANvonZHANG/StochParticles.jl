# test/test_dilution.jl
using StochParticles
using Test
using StaticArrays

struct MockIntegrator
    u::Vector{Float64}
    p::ParticleSystem
    t::Float64
end

@testset "DilutionProcess" begin
    @testset "dilution death jump removes a particle" begin
        dilution_rate = (t) -> 0.5
        background_sampler = (t) -> SVector(0.5)
        proc = DilutionProcess(dilution_rate, background_sampler)

        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(1), 10, 1.0, gas_fn)
        u0 = fill(1.0, 20)
        sys.n_active = 10
        mass_before = total_mass(u0, Val(1), 10)

        mock = MockIntegrator(u0, sys, 0.0)
        dilution_death_affect!(mock, proc)

        @test sys.n_active == 9
        mass_after = total_mass(mock.u, Val(1), 9)
        @test mass_after < mass_before  # one particle removed
    end

    @testset "dilution birth jump adds background particle" begin
        dilution_rate = (t) -> 0.5
        background_sampler = (t) -> SVector(0.5)
        proc = DilutionProcess(dilution_rate, background_sampler)

        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(1), 10, 1.0, gas_fn)
        u0 = fill(0.0, 20)
        sys.n_active = 5

        mock = MockIntegrator(u0, sys, 0.0)
        dilution_birth_affect!(mock, proc)

        @test sys.n_active == 6
        @test get_particle(mock.u, 6, Val(1)) == SVector(0.5)
    end
end
