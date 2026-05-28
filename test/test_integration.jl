# test/test_integration.jl
using StochParticles
using Test
using StaticArrays
using OrdinaryDiffEq
using JumpProcesses

@testset "Integration: Full PDMP Simulation" begin
    @testset "condensation + coagulation PDMP" begin
        A = 1
        n_sim = 50

        flux = (μ, g, t) -> -0.01 .* μ
        cond = CondensationProcess(flux)

        kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())

        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        tspan = (0.0, 10.0)

        prob = ParticleProblem(
            particles, 1.0, gas_fn, (cond, coag); tspan = tspan, n_sim = n_sim)

        @test prob isa JumpProblem

        sol = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "all four processes PDMP" begin
        A = 1
        n_sim = 30

        cond = CondensationProcess((μ, g, t) -> -0.01 .* μ)
        kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())
        emit = EmissionProcess(0.1, t -> SVector(1.0e-15))
        dil = DilutionProcess(t -> 0.05, t -> SVector(0.5e-15))

        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        tspan = (0.0, 5.0)

        prob = ParticleProblem(particles, 1.0, gas_fn, (cond, coag, emit, dil);
            tspan = tspan, n_sim = n_sim)

        sol = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "coagulation only maintains n_sim" begin
        A = 1
        n_sim = 100

        kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())

        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        tspan = (0.0, 1.0)

        prob = ParticleProblem(
            particles, 1.0, gas_fn, (coag,); tspan = tspan, n_sim = n_sim)

        sol = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success

        sys = prob.prob.p
        @test sys.n_active == sys.n_sim
    end

    @testset "QSSA condensation: no negative masses" begin
        T0 = 293.15
        V = 1.0e-6
        n_sim = 200
        densities = SVector(1770.0, 1000.0)
        h2o_idx = 2

        n_aitken = div(n_sim, 2)
        n_accum = n_sim - n_aitken

        particles_aitken = lognormal_masses(n_aitken, 3.0e-8, 1.6, densities)
        particles_accum = lognormal_masses(n_accum, 1.2e-7, 1.8, densities)
        particles = [SVector{2, Float64}(m[1], 0.0)
                     for m in vcat(particles_aitken, particles_accum)]

        S_target = 0.003
        p_sat_0 = saturation_vapor_pressure(T0)
        p_v0 = p_sat_0 * (1.0 + S_target)
        gas_fn = t -> SVector(T0, p_v0)

        avg_thermo = ThermodynamicsParams(
            SVector(0.455, 0.0), 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)

        # Pre-equilibrate
        pre_equilibrate!(particles, avg_thermo, densities, T0, p_v0; h2o_idx = h2o_idx)

        cond = H2OCondensationProcess(avg_thermo, densities; h2o_idx = h2o_idx, w = 0.0)
        prob = ParticleProblem(particles, V, gas_fn, (cond,);
            tspan = (0.0, 60.0), n_sim = n_sim)

        sol = solve(prob, Tsit5(); saveat = 0.0:10.0:60.0)

        @test sol.retcode == ReturnCode.Success
        for u in sol.u
            for i in 1:n_sim
                @test u[(i - 1) * 2 + h2o_idx] >= 0.0
            end
        end
    end
end
