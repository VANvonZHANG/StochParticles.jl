using StochParticles
using Test

@testset "Utils" begin
    @testset "bin_size_distribution" begin
        diams = [0.5, 1.5, 2.5, 3.5, 4.5]
        edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        counts = bin_size_distribution(diams, edges)
        @test counts == [1, 1, 1, 1, 1]

        @test bin_size_distribution(Float64[], edges) == zeros(Int, 5)

        diams2 = [1.1, 1.2, 1.9]
        counts2 = bin_size_distribution(diams2, edges)
        @test counts2 == [0, 3, 0, 0, 0]

        @test_throws ArgumentError bin_size_distribution(diams, [1.0, 1.0])
        @test_throws ArgumentError bin_size_distribution(diams, [2.0, 1.0])
        @test_throws ArgumentError bin_size_distribution(diams, [1.0])
    end
end
