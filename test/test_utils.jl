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

    @testset "standard atmospheres" begin
        aerosol = standard_aerosol_atmosphere()
        @test aerosol.T == 293.15
        @test aerosol.p == 101325.0
        @test aerosol.rho_p == 1800.0

        cloud = standard_cloud_atmosphere()
        @test cloud.T == 288.15
        @test cloud.p == 80000.0
        @test cloud.rho_p == 1000.0
    end

    @testset "lognormal_masses" begin
        masses = lognormal_masses(10000, 1.0e-7, 1.5, 1000.0)
        @test length(masses) == 10000
        @test all(m -> m[1] > 0, masses)

        diams = diameters_from_masses(masses, 1000.0)
        @test length(diams) == 10000
        @test all(d -> d > 0, diams)

        for (m, d) in zip(masses, diams)
            @test m[1] ≈ (π / 6.0) * d^3 * 1000.0 rtol = 1e-10
        end
    end
end
