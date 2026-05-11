using StochParticles
using Test

@testset "Plotting" begin
    @testset "plot_concentration_evolution" begin
        t = [0.0, 1.0, 2.0]
        N = [1.0e10, 9.0e9, 8.0e9]
        M = [1.0e-6, 1.0e-6, 1.0e-6]
        pl = plot_concentration_evolution(t, N, M)
        @test pl !== nothing
    end

    @testset "plot_size_distribution_heatmap" begin
        times = [0.0, 1.0]
        centers = [1.0e-6, 2.0e-6]
        matrix = rand(2, 2)
        pl = plot_size_distribution_heatmap(times, centers, matrix)
        @test pl !== nothing
    end

    @testset "plot_kernel_contributions" begin
        labels = ["Brownian", "Gravitational"]
        values = [1.0e-15, 1.0e-14]
        pl = plot_kernel_contributions(labels, values)
        @test pl !== nothing
    end

    @testset "plot_simulation_summary" begin
        n_sim = 50
        kernel = BrownianKernel(293.15, 1.81e-5, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())
        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        prob = ParticleProblem(particles, 1.0, gas_fn, (coag,); tspan=(0.0, 1.0), n_sim=n_sim)
        sol = solve(prob, Tsit5(); saveat=0.1)

        bin_edges = [0.0, 1.0e-6, 2.0e-6]
        pl = plot_simulation_summary(sol, prob, bin_edges, 1000.0)
        @test pl !== nothing

        # Test with time_unit="min"
        pl_min = plot_simulation_summary(sol, prob, bin_edges, 1000.0; time_unit="min")
        @test pl_min !== nothing
    end
end
