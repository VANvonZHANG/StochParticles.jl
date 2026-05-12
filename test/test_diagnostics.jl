# test/test_diagnostics.jl
using OrdinaryDiffEq
using StochParticles
using Test
using StaticArrays

@testset "Diagnostics" begin
    @testset "zeroth moment (total number concentration)" begin
        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(1), 100, 2.0, gas_fn)
        u0 = fill(1.0, 100)

        @test number_concentration(sys) == 100 / 2.0  # N/V = 100/2 = 50
    end

    @testset "first moment (total mass concentration)" begin
        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(2), 3, 1.0, gas_fn)
        u0 = make_u0([SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)])

        M = mass_concentration(u0, Val(2), sys)
        @test M ≈ (1+2+3+4+5+6) / 1.0  # total mass / volume
    end

    @testset "extract_concentrations" begin
        particles = [SVector{1,Float64}(1.0e-18), SVector{1,Float64}(2.0e-18)]
        volume = 1.0e-6
        gas_fn = t -> SVector(0.0)
        prob = ParticleProblem(particles, volume, gas_fn, ();
                               tspan = (0.0, 0.01), n_sim = 2)
        sol = solve(prob, Tsit5(); saveat = 0.005)

        t, N_conc, M_conc = extract_concentrations(sol, prob)
        @test length(t) == length(sol.t)
        @test length(N_conc) == length(sol.t)
        @test length(M_conc) == length(sol.t)
        @test all(N_conc .> 0)
        @test all(M_conc .> 0)

        # Number concentration should be constant (no coagulation)
        @test all(N_conc .≈ N_conc[1])

        # Mass concentration should be constant (no processes)
        @test all(M_conc .≈ M_conc[1])
    end

    @testset "particle_diameters" begin
        particles = [SVector{1,Float64}(1.0e-18), SVector{1,Float64}(8.0e-18)]
        volume = 1.0e-6
        gas_fn = t -> SVector(0.0)
        prob = ParticleProblem(particles, volume, gas_fn, ();
                               tspan = (0.0, 0.01), n_sim = 2)
        sys = prob.prob.p
        u0 = make_u0(particles)
        diams = particle_diameters(u0, sys, 1000.0)
        @test length(diams) == 2
        @test diams[1] ≈ (6.0 * 1.0e-18 / (π * 1000.0))^(1.0 / 3.0)
        @test diams[2] ≈ (6.0 * 8.0e-18 / (π * 1000.0))^(1.0 / 3.0)
    end

    @testset "compute_size_distribution" begin
        particles = [SVector{1,Float64}(1.0e-18), SVector{1,Float64}(8.0e-18)]
        volume = 1.0e-6
        gas_fn = t -> SVector(0.0)
        prob = ParticleProblem(particles, volume, gas_fn, ();
                               tspan = (0.0, 0.01), n_sim = 2)
        sol = solve(prob, Tsit5(); saveat = 0.005)

        bin_edges = [0.0, 1.0e-6, 2.0e-6]
        snapshot_times, bin_centers, matrix = compute_size_distribution(
            sol, prob, bin_edges, 1000.0; n_snapshots = 3)

        @test length(snapshot_times) == 3
        @test length(bin_centers) == 2
        @test size(matrix) == (2, 3)
        @test all(matrix .>= 0)
    end

    @testset "check_mass_conservation" begin
        n_sim = 50
        kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())

        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        tspan = (0.0, 1.0)

        prob = ParticleProblem(particles, 1.0, gas_fn, (coag,); tspan=tspan, n_sim=n_sim)
        sol = solve(prob, Tsit5(); saveat = 0.1)

        passed, rel_error = check_mass_conservation(sol, prob)
        @test passed == true
        @test rel_error < 1e-3
    end
end
