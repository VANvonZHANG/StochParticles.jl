using StochParticles
using Test
using StaticArrays

@testset "Mixing State" begin
    @testset "species_fractions" begin
        # Normal case
        μ = SVector(3.0, 7.0)
        f = species_fractions(μ)
        @test f ≈ SVector(0.3, 0.7)
        @test sum(f) ≈ 1.0

        # Uniform fallback for zero mass
        μ_zero = SVector(0.0, 0.0)
        f_zero = species_fractions(μ_zero)
        @test f_zero ≈ SVector(0.5, 0.5)

        # Single species
        μ1 = SVector(5.0)
        f1 = species_fractions(μ1)
        @test f1 ≈ SVector(1.0)
    end
end
