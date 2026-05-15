using StochParticles
using Test
using StaticArrays

@testset "Mixing State" begin
    @testset "species_fractions" begin
        # Normal case
        μ = SVector(3.0, 7.0)
        f = @inferred(species_fractions(μ))
        @test f ≈ SVector(0.3, 0.7)
        @test sum(f) ≈ 1.0

        # Uniform fallback for zero mass
        μ_zero = SVector(0.0, 0.0)
        f_zero = @inferred(species_fractions(μ_zero))
        @test f_zero ≈ SVector(0.5, 0.5)

        # Single species
        μ1 = SVector(5.0)
        f1 = @inferred(species_fractions(μ1))
        @test f1 ≈ SVector(1.0)

        # Very small total mass
        μ_tiny = SVector(1e-300, 2e-300)
        @test @inferred(species_fractions(μ_tiny)) ≈ SVector(1.0 / 3.0, 2.0 / 3.0)

        # Three species
        μ3 = SVector(1.0, 2.0, 3.0)
        @test @inferred(species_fractions(μ3)) ≈ SVector(1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0)
    end
end
