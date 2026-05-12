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

        prob = ParticleProblem(particles, 1.0, gas_fn, (cond, coag); tspan=tspan, n_sim=n_sim)

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
                               tspan=tspan, n_sim=n_sim)

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

        prob = ParticleProblem(particles, 1.0, gas_fn, (coag,); tspan=tspan, n_sim=n_sim)

        sol = solve(prob, Tsit5())
        @test sol.retcode == ReturnCode.Success

        sys = prob.prob.p
        @test sys.n_active == sys.n_sim
    end
end
