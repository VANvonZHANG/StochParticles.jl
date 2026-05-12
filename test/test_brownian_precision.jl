# test/test_brownian_precision.jl
using StochParticles
using Test
using StaticArrays

"""
    continuum_brownian_kernel(T, mu_f, d_i, d_j)

Simplified continuum-regime formula for comparison.
K_cont = (2 * kb * T) / (3 * mu_f) * (d_i + d_j)^2 / (d_i * d_j)
"""
function continuum_brownian_kernel(T, mu_f, d_i, d_j)
    kb = 1.380649e-23
    return (2.0 * kb * T) / (3.0 * mu_f) * (d_i + d_j)^2 / (d_i * d_j)
end

@testset "BrownianKernel Transition-Regime Precision" begin
    T = 293.15
    p = 101325.0
    densities = SVector(1000.0)
    kernel = BrownianKernel(T, p, densities)

    # Pre-compute mu_f via Sutherland formula for the continuum comparison
    mu_f = 1.8325e-5 * (416.16 / (T + 120.0)) * (T / 296.16)^1.5

    @testset "Small particles: enhanced by slip correction" begin
        # ~20 nm particles — slip correction significantly enhances coagulation
        mu_small = SVector(4.0e-21)
        K_full = kernel(mu_small, mu_small)
        d_small = 20.0e-9
        K_cont = continuum_brownian_kernel(T, mu_f, d_small, d_small)

        @test K_full > K_cont * 1.2  # slip correction should enhance by >20%
        @test K_full > 0
    end

    @testset "Medium particles: moderate enhancement" begin
        # ~60 nm particles — moderate slip correction
        mu_medium = SVector(4.0e-18)
        K_full = kernel(mu_medium, mu_medium)
        d_medium = 60.0e-9
        K_cont = continuum_brownian_kernel(T, mu_f, d_medium, d_medium)

        @test K_full > K_cont * 1.05   # still enhanced, but less
        @test K_full < K_cont * 1.5    # but not drastically
    end

    @testset "Large particles: continuum limit" begin
        # 10 um particles — continuum formula should be very close
        mu_large = SVector(5.235987755982988e-13)  # 10 um water droplet
        K_full = kernel(mu_large, mu_large)
        d_large = 10.0e-6
        K_cont = continuum_brownian_kernel(T, mu_f, d_large, d_large)

        @test K_full ≈ K_cont rtol=0.05  # within 5% of continuum limit
    end

    @testset "Cross-size: small + large" begin
        # ~20 nm colliding with 10 um
        mu_small = SVector(4.0e-21)
        mu_large = SVector(5.235987755982988e-13)
        K_full = kernel(mu_small, mu_large)

        d_small = 20.0e-9
        d_large = 10.0e-6
        K_cont = continuum_brownian_kernel(T, mu_f, d_small, d_large)

        @test K_full > K_cont          # enhanced by slip correction on small particle
        @test K_full > 0
    end

    @testset "Symmetry" begin
        mu_i = SVector(1.0e-18)
        mu_j = SVector(2.0e-18)
        @test kernel(mu_i, mu_j) ≈ kernel(mu_j, mu_i)
    end
end
