# Tests for the M2 synthetic-population initialization subsystem.
using StochParticles
using StaticArrays
using SpecialFunctions
using Statistics
using Random
using Test

const EDGES = collect(10.0 .^ range(-8.3, -4.5; length = 96))

@testset "TabulatedSpectrum construction" begin
    tbl = TabulatedSpectrum(EDGES, ones(length(EDGES) - 1))
    @test tbl.bin_edges == EDGES
    @test length(tbl.dNdlogD) == length(EDGES) - 1
    @test_throws ArgumentError TabulatedSpectrum([1.0, 3.0, 2.0], [1.0, 1.0])  # unsorted
    @test_throws ArgumentError TabulatedSpectrum([1.0, 1.0, 2.0], [1.0, 1.0])  # ties
    @test_throws ArgumentError TabulatedSpectrum([-1.0, 1.0, 2.0], [1.0, 1.0]) # non-positive
    @test_throws ArgumentError TabulatedSpectrum([0.0, 1.0, 2.0], [1.0, 1.0]) # zero edge
    @test_throws ArgumentError TabulatedSpectrum([1.0, 2.0], [-0.1])           # negative dNdlogD
    @test_throws ArgumentError TabulatedSpectrum([1.0], Float64[])             # no bin
    @test_throws ArgumentError TabulatedSpectrum([1.0, 2.0, 3.0], [1.0])       # length mismatch
end

@testset "number_concentration and +" begin
    tbl = TabulatedSpectrum([1.0, 10.0, 100.0], [2.0, 4.0])
    # exact trapezoid-in-log: 2.0*log(10) + 4.0*log(10)
    @test number_concentration(tbl) ≈ 6.0 * log(10) rtol = 1e-14
    a = TabulatedSpectrum(EDGES, fill(1.0, length(EDGES) - 1))
    b = TabulatedSpectrum(EDGES, fill(2.0, length(EDGES) - 1))
    s = a + b
    @test s.dNdlogD == fill(3.0, length(EDGES) - 1)
    other = TabulatedSpectrum(EDGES[1:(end - 1)], ones(length(EDGES) - 2))
    @test_throws ArgumentError a + other
end

@testset "lognormal_table exact discretization" begin
    for (dg, sg, n) in [(6.0e-8, 1.45, 8.0e11), (1.6e-7, 1.55, 3.2e11)]
        tbl = lognormal_table(dg, sg, n, EDGES)
        @test number_concentration(tbl) ≈ n rtol = 1e-9
        @test all(>=(0.0), tbl.dNdlogD)
    end
    tbl = lognormal_table(6.0e-8, 1.45, 8.0e11, EDGES) +
          lognormal_table(1.6e-7, 1.55, 3.2e11, EDGES)
    @test number_concentration(tbl) ≈ 1.12e12 rtol = 1e-9
    @test_throws ArgumentError lognormal_table(6.0e-8, 0.5, 100.0, EDGES)
end

function ks_statistic(samples::Vector{Float64}, cdf::Function)
    n = length(samples)
    d = 0.0
    for (i, x) in enumerate(sort(samples))
        f = cdf(x)
        d = max(d, abs(f - i / n), abs(f - (i - 1) / n))
    end
    return d
end

lognormal_cdf(x, dg, sg) = 0.5 * (1.0 + erf((log(x) - log(dg)) / (log(sg) * sqrt(2.0))))

const SPEC = lognormal_table(6.0e-8, 1.45, 8.0e11, EDGES) +
             lognormal_table(1.6e-7, 1.55, 3.2e11, EDGES)

@testset "inverse-CDF sampling" begin
    empty = TabulatedSpectrum([1.0, 2.0, 3.0], [0.0, 0.0])
    @test_throws ArgumentError StochParticles._spectrum_bin_probs(empty)

    rng = MersenneTwister(20260930)
    n = 20_000
    diameters, bins = StochParticles._sample_dry_diameters(rng, SPEC, n)
    @test length(diameters) == n == length(bins)
    @test all(1 .<= bins .<= length(EDGES) - 1)
    # every diameter lies inside its reported bin
    for i in 1:n
        @test EDGES[bins[i]] <= diameters[i] <= EDGES[bins[i] + 1]
    end
    # KS against the bimodal analytic CDF (alpha = 0.01 critical value 1.63)
    w1 = 8.0e11 / 1.12e12
    mix_cdf = x -> w1 * lognormal_cdf(x, 6.0e-8, 1.45) +
                   (1 - w1) * lognormal_cdf(x, 1.6e-7, 1.55)
    dstat = ks_statistic(diameters, mix_cdf)
    @test dstat * sqrt(n) < 1.63
    # bin counts within 5 sigma of the analytic per-bin probabilities
    probs = StochParticles._spectrum_bin_probs(SPEC)
    counts = [sum(bins .== b) for b in 1:(length(EDGES) - 1)]
    for b in eachindex(probs)
        expected = n * probs[b]
        expected >= 5 || continue
        @test abs(counts[b] - expected) <= 5.0 * sqrt(n * probs[b] * (1 - probs[b]))
    end
end
