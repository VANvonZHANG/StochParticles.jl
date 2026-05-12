# test/test_coagulation.jl
using StochParticles
using Test
using StaticArrays

@testset "Coagulation Kernels" begin
    @testset "BrownianKernel" begin
        kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))  # single species, ρ=1000
        μ_i = SVector(1.0e-18)   # 1 attogram
        μ_j = SVector(1.0e-18)
        K = kernel(μ_i, μ_j)
        @test K > 0  # kernel must be positive
        # Symmetry
        @test kernel(μ_i, μ_j) ≈ kernel(μ_j, μ_i)
    end

    @testset "CompositeKernel" begin
        k1 = BrownianKernel(293.15, 101325.0, SVector(1000.0))
        k2 = BrownianKernel(293.15, 101325.0, SVector(1000.0))
        comp = CompositeKernel(k1, k2)
        μ = SVector(1.0e-18)
        @test comp(μ, μ) ≈ 2 * k1(μ, μ)  # sum of both kernels
    end

    @testset "compute_majorant" begin
        kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))
        particles = [SVector(1.0e-18), SVector(2.0e-18), SVector(3.0e-18)]
        u0 = make_u0(particles)
        sys = ParticleSystem(Val(1), 3, 1.0, t -> SVector(0.0))
        K_max = compute_majorant(GlobalMajorant(), kernel, u0, sys)
        # K_max must be >= every pair's actual kernel value
        for i in 1:3, j in (i + 1):3

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

    @testset "AyalaTurbulentKernel" begin
        kernel = AyalaTurbulentKernel(
            0.01, 50.0, 1.48e-5, 1.225, 1000.0, 9.81, SVector(1000.0))

        # epsilon <= 0 throws
        @test_throws DomainError AyalaTurbulentKernel(
            -0.01, 50.0, 1.48e-5, 1.225, 1000.0, 9.81, SVector(1000.0))
        @test_throws DomainError AyalaTurbulentKernel(
            0.0, 50.0, 1.48e-5, 1.225, 1000.0, 9.81, SVector(1000.0))

        # Zero-size particles -> K = 0
        μ_zero = SVector(0.0)
        μ_nonzero = SVector(1.0e-12)
        @test kernel(μ_zero, μ_nonzero) == 0.0
        @test kernel(μ_nonzero, μ_zero) == 0.0

        # Monodisperse: same-size droplets have K > 0 (turbulence causes same-size collisions)
        μ_20um = SVector(4.0e-15)   # ~20 μm
        K_mono = kernel(μ_20um, μ_20um)
        @test K_mono > 0.0

        # Bidisperse: K > 0
        μ_10um = SVector(5.0e-16)   # ~10 μm
        K_bi = kernel(μ_10um, μ_20um)
        @test K_bi > 0.0

        # Symmetry
        @test K_bi ≈ kernel(μ_20um, μ_10um)

        # Type stability
        @test @inferred(kernel(μ_10um, μ_20um)) > 0.0

        # Stagnant limit: small epsilon should produce positive K
        K_turb_small = AyalaTurbulentKernel(
            1e-6, 50.0, 1.48e-5, 1.225, 1000.0, 9.81, SVector(1000.0))
        K_small_val = K_turb_small(μ_10um, μ_20um)
        @test K_small_val > 0.0
    end

    @testset "AtmosphericParameters + make_kernel" begin
        params = AtmosphericParameters(293.15, 1.0e5)
        densities = SVector(1000.0)
        epsilon = 0.01
        R_lambda = 50.0

        K_total = make_kernel(params, epsilon, R_lambda, densities)
        @test K_total isa CoagulationKernel

        μ_small = SVector(1.0e-18)  # aerosol scale
        μ_large = SVector(1.0e-12)  # cloud droplet scale

        # Total kernel should be positive for all size pairs
        @test K_total(μ_small, μ_small) > 0.0
        @test K_total(μ_large, μ_large) > 0.0
        @test K_total(μ_small, μ_large) > 0.0

        # Type stability
        @test @inferred(K_total(μ_small, μ_large)) > 0.0
    end

    @testset "Three-kernel CompositeKernel" begin
        params = AtmosphericParameters(293.15, 1.0e5)
        densities = SVector(1000.0)

        K_brown = BrownianKernel(params.T, params.p, densities)
        K_grav = GravitationalKernel(
            params.mu_f, params.rho_f, params.rho_p, params.g, densities)
        K_turb = AyalaTurbulentKernel(
            0.01, 50.0, params.nu, params.rho_f, params.rho_p, params.g, densities)

        K_combined = CompositeKernel(K_brown, CompositeKernel(K_grav, K_turb))

        μ_10um = SVector(5.0e-16)
        μ_20um = SVector(4.0e-15)

        K_total = K_combined(μ_10um, μ_20um)
        K_sum = K_brown(μ_10um, μ_20um) + K_grav(μ_10um, μ_20um) + K_turb(μ_10um, μ_20um)

        @test K_total ≈ K_sum
    end
end

@testset "CoagulationProcess" begin
    @testset "provides no drift" begin
        kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))
        proc = CoagulationProcess(kernel, GlobalMajorant())
        @test provides_drift(proc) == false
    end

    @testset "coagulation jump preserves mass and n_sim" begin
        kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))
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
