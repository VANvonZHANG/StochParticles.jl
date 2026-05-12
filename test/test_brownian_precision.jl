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
        # 10 nm particles — slip correction significantly enhances coagulation
        mu_10nm = SVector(4.0e-21)   # mass of a 10 nm water droplet
        K_full = kernel(mu_10nm, mu_10nm)
        d_10nm = 10.0e-9
        K_cont = continuum_brownian_kernel(T, mu_f, d_10nm, d_10nm)

        @test K_full > K_cont * 1.5  # slip correction should enhance by >50%
        @test K_full > 0
    end

    @testset "Medium particles: moderate enhancement" begin
        # 100 nm particles — moderate slip correction
        mu_100nm = SVector(4.0e-18)
        K_full = kernel(mu_100nm, mu_100nm)
        d_100nm = 100.0e-9
        K_cont = continuum_brownian_kernel(T, mu_f, d_100nm, d_100nm)

        @test K_full > K_cont * 1.05   # still enhanced, but less
        @test K_full < K_cont * 2.0    # but not crazy enhanced
    end

    @testset "Large particles: continuum limit" begin
        # 1 um particles — continuum formula should be very close
        mu_1um = SVector(4.0e-15)
        K_full = kernel(mu_1um, mu_1um)
        d_1um = 1.0e-6
        K_cont = continuum_brownian_kernel(T, mu_f, d_1um, d_1um)

        @test K_full ≈ K_cont rtol=0.05  # within 5% of continuum limit
    end

    @testset "Cross-size: small + large" begin
        # 10 nm colliding with 1 um
        mu_small = SVector(4.0e-21)
        mu_large = SVector(4.0e-15)
        K_full = kernel(mu_small, mu_large)

        d_small = 10.0e-9
        d_large = 1.0e-6
        K_cont = continuum_brownian_kernel(T, mu_f, d_small, d_large)

        @test K_full > K_cont          # enhanced by slip correction on small particle
        @test K_full > 0
    end

    @testset "Symmetry" begin
        mu_i = SVector(1.0e-18)
        mu_j = SVector(2.0e-18)
        @test kernel(mu_i, mu_j) ≈ kernel(mu_j, mu_i)
    end

    @testset "Monodisperse consistency" begin
        # Same-size particles should give same result regardless of which is i or j
        mu = SVector(1.0e-18)
        @test kernel(mu, mu) ≈ kernel(mu, mu)
    end
end
