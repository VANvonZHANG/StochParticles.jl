# test/test_particle_system.jl
using StochParticles
using Test
using StaticArrays

@testset "ParticleSystem" begin
    @testset "construction" begin
        gas_fn = t -> SVector(0.0, 0.0)
        sys = ParticleSystem(Val(2), 100, 1.0, gas_fn)
        @test sys.n_active == 100
        @test sys.volume == 1.0
        @test sys.n_sim == 100
        @test species_val(sys) === Val(2)
    end

    @testset "get_particle / set_particle!" begin
        A = 3
        u = zeros(Float64, 10 * A)  # 10 particles, 3 species
        set_particle!(u, 3, Val(A), SVector(1.0, 2.0, 3.0))
        μ = get_particle(u, 3, Val(A))
        @test μ == SVector(1.0, 2.0, 3.0)
        # Verify neighbors untouched
        @test get_particle(u, 2, Val(A)) == SVector(0.0, 0.0, 0.0)
        @test get_particle(u, 4, Val(A)) == SVector(0.0, 0.0, 0.0)
    end

    @testset "make_u0" begin
        particles = [SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)]
        u0 = make_u0(particles)
        @test length(u0) == 6
        @test get_particle(u0, 1, Val(2)) == SVector(1.0, 2.0)
        @test get_particle(u0, 3, Val(2)) == SVector(5.0, 6.0)
    end

    @testset "total_mass" begin
        particles = [SVector(1.0, 2.0), SVector(3.0, 4.0)]
        u0 = make_u0(particles)
        @test total_mass(u0, Val(2), 2) ≈ 10.0
    end
end
