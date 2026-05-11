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
end
