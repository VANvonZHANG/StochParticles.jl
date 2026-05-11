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
end
