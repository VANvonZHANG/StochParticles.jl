# test/test_coagulation.jl
using StochParticles
using Test
using StaticArrays

@testset "Coagulation Kernels" begin
    @testset "BrownianKernel" begin
        kernel = BrownianKernel(293.15, 1.81e-5, SVector(1000.0))  # single species, ρ=1000
        μ_i = SVector(1.0e-18)   # 1 attogram
        μ_j = SVector(1.0e-18)
        K = kernel(μ_i, μ_j)
        @test K > 0  # kernel must be positive
        # Symmetry
        @test kernel(μ_i, μ_j) ≈ kernel(μ_j, μ_i)
    end

    @testset "CompositeKernel" begin
        k1 = BrownianKernel(293.15, 1.81e-5, SVector(1000.0))
        k2 = BrownianKernel(293.15, 1.81e-5, SVector(1000.0))
        comp = CompositeKernel(k1, k2)
        μ = SVector(1.0e-18)
        @test comp(μ, μ) ≈ 2 * k1(μ, μ)  # sum of both kernels
    end

    @testset "compute_majorant" begin
        kernel = BrownianKernel(293.15, 1.81e-5, SVector(1000.0))
        particles = [SVector(1.0e-18), SVector(2.0e-18), SVector(3.0e-18)]
        u0 = make_u0(particles)
        sys = ParticleSystem(Val(1), 3, 1.0, t -> SVector(0.0))
        K_max = compute_majorant(GlobalMajorant(), kernel, u0, sys)
        # K_max must be >= every pair's actual kernel value
        for i in 1:3, j in (i+1):3
            K_ij = kernel(get_particle(u0, i, Val(1)), get_particle(u0, j, Val(1)))
            @test K_max >= K_ij
        end
    end
end
