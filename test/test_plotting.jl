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
        pl = plot_simulation_summary(sol, prob, bin_edges, 1000.0; method = :histogram)
        @test pl !== nothing

        # Test with time_unit="min"
        pl_min = plot_simulation_summary(sol, prob, bin_edges, 1000.0;
            time_unit = "min", method = :histogram)
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

    @testset "smooth_histogram_diameter" begin
        using Random
        Random.seed!(42)
        diams = 1e-6 .* exp.(0.3 .* randn(1000))

        bin_edges = 10.0 .^ range(-7, -5; length=11)  # 10 bins
        V_t = 1.0

        result = StochParticles.smooth_histogram_diameter(diams, bin_edges, V_t)

        # Result has one value per bin (length(edges) - 1)
        @test length(result) == length(bin_edges) - 1

        # All values non-negative
        @test all(result .>= 0.0)

        # With smooth_factor=1, should behave like raw histogram (single fine bin per original bin)
        result_sf1 = StochParticles.smooth_histogram_diameter(
            diams, bin_edges, V_t; smooth_factor = 1)
        @test length(result_sf1) == length(bin_edges) - 1
        @test all(result_sf1 .>= 0.0)

        # With smooth_factor=5, still produces valid results
        result_sf5 = StochParticles.smooth_histogram_diameter(
            diams, bin_edges, V_t; smooth_factor = 5)
        @test length(result_sf5) == length(bin_edges) - 1
        @test all(result_sf5 .>= 0.0)

        # Empty input returns zeros
        result_empty = StochParticles.smooth_histogram_diameter(
            Float64[], bin_edges, V_t)
        @test all(result_empty .== 0.0)
    end

    @testset "compute_size_distribution methods" begin
        n_sim = 200
        kernel = BrownianKernel(293.15, 101325.0, SVector(1800.0))
        coag = CoagulationProcess(kernel, GlobalMajorant())
        gas_fn = t -> SVector(0.0)
        particles = fill(SVector(1.0e-15), n_sim)
        prob = ParticleProblem(
            particles, 1.0, gas_fn, (coag,); tspan = (0.0, 1.0), n_sim = n_sim)
        sol = solve(prob, Tsit5(); saveat = 0.1)
        bin_edges = 10.0 .^ range(-9, -5; length = 26)

        # Test :histogram method (existing behavior)
        t1, c1, m1 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0; n_snapshots = 5, method = :histogram)
        @test length(t1) == 5
        @test length(c1) == 25
        @test size(m1) == (25, 5)
        @test all(m1 .>= 0.0)

        # Test :kde method (new default)
        t2, c2, m2 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0; n_snapshots = 5, method = :kde)
        @test length(t2) == 5
        @test length(c2) == 25
        @test size(m2) == (25, 5)
        @test all(m2 .>= 0.0)

        # Test :histogram_smooth method
        t3, c3, m3 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0; n_snapshots = 5, method = :histogram_smooth)
        @test length(t3) == 5
        @test length(c3) == 25
        @test size(m3) == (25, 5)
        @test all(m3 .>= 0.0)

        # All methods should return the same bin centers and snapshot times
        @test c1 ≈ c2
        @test c1 ≈ c3
        @test t1 ≈ t2
        @test t1 ≈ t3

        # Test invalid method throws
        @test_throws ArgumentError compute_size_distribution(
            sol, prob, bin_edges, 1800.0; method = :invalid)

        # Test that :kde passes bandwidth_factor through
        t4, c4, m4 = compute_size_distribution(
            sol, prob, bin_edges, 1800.0;
            n_snapshots = 3, method = :kde, bandwidth_factor = 2.0)
        @test size(m4) == (25, 3)
        @test all(m4 .>= 0.0)
    end
end
