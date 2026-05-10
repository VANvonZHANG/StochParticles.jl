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

    @testset "GravitationalKernel" begin
        kernel = GravitationalKernel(1.81e-5, 1.225, 1000.0, 9.81, SVector(1000.0))

        # Stagnant air limit: g = 0 -> K = 0
        kernel_zero_g = GravitationalKernel(1.81e-5, 1.225, 1000.0, 0.0, SVector(1000.0))
        μ = SVector(1.0e-15)  # ~12 μm droplet
        @test kernel_zero_g(μ, μ) == 0.0

        # Monodisperse case: identical sizes -> Δv = 0 -> K = 0
        @test kernel(μ, μ) == 0.0

        # Bidisperse case: different sizes -> K > 0
        μ_small = SVector(1.0e-15)   # ~12 μm
        μ_large = SVector(1.0e-12)   # ~124 μm
        K = kernel(μ_small, μ_large)
        @test K > 0.0

        # Symmetry
        @test kernel(μ_small, μ_large) ≈ kernel(μ_large, μ_small)

        # Type stability
        @test @inferred(kernel(μ_small, μ_large)) > 0.0
    end
end

@testset "CoagulationProcess" begin
    @testset "provides no drift" begin
        kernel = BrownianKernel(293.15, 1.81e-5, SVector(1000.0))
        proc = CoagulationProcess(kernel, GlobalMajorant())
        @test provides_drift(proc) == false
    end

    @testset "coagulation jump preserves mass and n_sim" begin
        kernel = BrownianKernel(293.15, 1.81e-5, SVector(1000.0))
        sampling = GlobalMajorant()
        proc = CoagulationProcess(kernel, sampling)

        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(1), 20, 1.0, gas_fn)
        particles = fill(SVector(1.0e-18), 20)
        u0 = make_u0(particles)

        mass_before = total_mass(u0, Val(1), sys.n_active)
        vol_before = sys.volume

        # Manually trigger the affect! function 10 times via a mock integrator
        jump = make_coagulation_jump(kernel, sampling)
        for _ in 1:10
            rate = jump.rate(u0, sys, 0.0)
            if rate > 0
                mock_int = MockIntegrator(u0, sys, 0.0)
                jump.affect!(mock_int)
                u0 = mock_int.u
            end
        end

        @test sys.n_active == sys.n_sim
        mass_after = total_mass(u0, Val(1), sys.n_active)
        # Mass concentration = mass/volume should be conserved
        conc_before = mass_before / vol_before
        conc_after = mass_after / sys.volume
        @test conc_after ≈ conc_before rtol=1e-10
    end
end
