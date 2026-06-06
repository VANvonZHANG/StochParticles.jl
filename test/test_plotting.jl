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
        kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())
        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        prob = ParticleProblem(
            particles, 1.0, gas_fn, (coag,); tspan = (0.0, 1.0), n_sim = n_sim)
        sol = solve(prob, Tsit5(); saveat = 0.1)

        bin_edges = [0.0, 1.0e-6, 2.0e-6]
        pl = plot_simulation_summary(sol, prob, bin_edges, 1000.0)
        @test pl !== nothing

        # Test with time_unit="min"
        pl_min = plot_simulation_summary(sol, prob, bin_edges, 1000.0; time_unit = "min")
        @test pl_min !== nothing
    end

    @testset "kde_log_diameter" begin
        using Random
        Random.seed!(42)
        # 1000 particles centered around d ≈ 1e-6 m (1 μm)
        diams = 1e-6 .* exp.(0.3 .* randn(1000))

        bin_edges = 10.0 .^ range(-7, -5; length=11)  # 10 bins
        bin_centers = @. sqrt(bin_edges[1:(end - 1)] * bin_edges[2:end])
        V_t = 1.0  # unit volume

        result = StochParticles.kde_log_diameter(diams, bin_centers, V_t)

        # Result has one value per bin center
        @test length(result) == length(bin_centers)

        # All values non-negative
        @test all(result .>= 0.0)

        # Peak should be near 1e-6 (the center of our distribution)
        peak_bin = argmax(result)
        @test bin_centers[peak_bin] ≈ 1e-6 atol = 5e-6

        # With bandwidth_factor = 0.5 (narrow), result should still be valid
        result_narrow = StochParticles.kde_log_diameter(
            diams, bin_centers, V_t; bandwidth_factor = 0.5)
        @test length(result_narrow) == length(bin_centers)
        @test all(result_narrow .>= 0.0)

        # With bandwidth_factor = 10.0 (very wide), result should be smoother
        result_wide = StochParticles.kde_log_diameter(
            diams, bin_centers, V_t; bandwidth_factor = 10.0)
        @test length(result_wide) == length(bin_centers)
        @test all(result_wide .>= 0.0)

        # Wider bandwidth should produce smaller max value (more spread out)
        @test maximum(result_wide) < maximum(result_narrow)
    end
end
