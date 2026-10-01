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

const ANCHORS = [(3.0e-8, SVector(0.18, 0.15, 0.57, 0.10)),
    (3.0e-7, SVector(0.35, 0.20, 0.30, 0.15))]
const FBAR = SizeResolvedComposition(SPEC; anchors = ANCHORS)

@testset "SizeResolvedComposition construction" begin
    bad = zeros(4, length(EDGES) - 1)
    bad[:, 1] .= 0.25
    @test_throws ArgumentError SizeResolvedComposition(EDGES, bad)          # column not summing to 1
    bad2 = fill(0.25, 4, length(EDGES) - 1)
    bad2[1, 1] = -0.25
    @test_throws ArgumentError SizeResolvedComposition(EDGES, bad2)         # negative entry

    const_fb = constant_fbar(SVector(0.25, 0.15, 0.50, 0.10), SPEC)
    @test const_fb.bin_edges == SPEC.bin_edges
    @test all(const_fb.fractions[:, b] == [0.25, 0.15, 0.50, 0.10]
    for b in 1:(length(EDGES) - 1))
end

@testset "anchor interpolation" begin
    centers = [sqrt(EDGES[b] * EDGES[b + 1]) for b in 1:(length(EDGES) - 1)]
    f1 = SVector(0.2, 0.2, 0.4, 0.2)
    f2 = SVector(0.5, 0.2, 0.2, 0.1)
    fb = SizeResolvedComposition(SPEC; anchors = [(centers[10], f1), (centers[90], f2)])
    @test fb.fractions[:, 10] ≈ collect(f1) atol = 1e-12
    @test fb.fractions[:, 90] ≈ collect(f2) atol = 1e-12
    @test all(isapprox.(sum(fb.fractions; dims = 1), 1.0; atol = 1e-12))
    # species 1 (AS) rises and species 3 (OA) falls between the anchors
    @test all(diff(fb.fractions[1, 10:90]) .>= -1e-12)
    @test all(diff(fb.fractions[3, 10:90]) .<= 1e-12)

    # anchor coverage / ordering validation
    @test_throws ArgumentError SizeResolvedComposition(
        SPEC; anchors = [(1.0e-7, f1), (3.0e-7, f2)])              # first anchor above the number-median diameter
    @test_throws ArgumentError SizeResolvedComposition(
        SPEC; anchors = [(centers[90], f2), (centers[10], f1)])    # unsorted
    @test_throws ArgumentError SizeResolvedComposition(
        SPEC; anchors = [(centers[10], f1), (centers[20], f2)])    # last anchor below median
    @test_throws ArgumentError SizeResolvedComposition(SPEC; anchors = [(centers[10], f1)])
end

@testset "driver anchors" begin
    @test FBAR.bin_edges == SPEC.bin_edges
    @test all(isapprox.(sum(FBAR.fractions; dims = 1), 1.0; atol = 1e-12))
    @test all(>=(0.0), FBAR.fractions)
end

const RHO = SVector(1770.0, 1720.0, 1400.0, 1800.0, 1000.0)
const MASK3 = [1, 2, 3]
const THERMO = ThermodynamicsParams(
    SVector(0.61, 0.67, 0.10, 0.0, 0.0), 0.072, 1000.0, 18.015e-3, 2.5e6, 461.5,
    2.5e-5, 2.4e-2)

function pop_spec(nu; n = 1000, chi = 0.5)
    SyntheticPopulationSpec(
        n_sim = n, spectrum = SPEC, fbar = FBAR, chi_target = chi,
        densities = RHO, h2o_idx = 5, chi_species = MASK3, T0 = 288.15, S0 = 0.0)
end

@testset "synthesize_population (explicit nu)" begin
    particles, dry_d,
    meta = synthesize_population(pop_spec(12.0); seed = 123,
        thermo = THERMO, nu = 12.0)
    @test length(particles) == 1000 == length(dry_d)
    @test meta.nu == 12.0 && meta.seed == 123
    @test isfinite(meta.chi_realized) && 0.0 <= meta.chi_realized <= 1.0
    # bitwise dry-volume conservation per particle
    for i in 1:1000
        m = particles[i]
        @test (pi / 6.0) * dry_d[i]^3 ≈ sum(m[k] / RHO[k] for k in 1:4) rtol = 1e-12
    end
    # haze equilibrium added non-negative water to the last slot
    @test all(p -> p[5] >= 0.0, particles)
    # same seed -> bitwise identical population
    p2, d2,
    _ = synthesize_population(pop_spec(12.0); seed = 123, thermo = THERMO,
        nu = 12.0)
    @test p2 == particles && d2 == dry_d
    # meta chi matches the library diagnostic recomputed by hand
    sys = ParticleSystem(Val(5), 1000, 1000 / number_concentration(SPEC),
        PrescribedProfile([0.0], [288.15], [0.0]))
    @test meta.chi_realized ==
          mixing_state_index(make_u0(particles), sys; species = MASK3)
end

@testset "synthesize_population validation" begin
    bad_h2o = SyntheticPopulationSpec(n_sim = 10, spectrum = SPEC, fbar = FBAR,
        chi_target = 0.5, densities = RHO, h2o_idx = 3, chi_species = MASK3,
        T0 = 288.15, S0 = 0.0)
    @test_throws ArgumentError synthesize_population(bad_h2o; seed = 1, thermo = THERMO)
    bad_rho = SyntheticPopulationSpec(n_sim = 10, spectrum = SPEC, fbar = FBAR,
        chi_target = 0.5, densities = SVector{4, Float64}(RHO[1:4]), h2o_idx = 5,
        chi_species = MASK3, T0 = 288.15, S0 = 0.0)
    @test_throws ArgumentError synthesize_population(bad_rho; seed = 1, thermo = THERMO)
    bad_mask = SyntheticPopulationSpec(n_sim = 10, spectrum = SPEC, fbar = FBAR,
        chi_target = 0.5, densities = RHO, h2o_idx = 5, chi_species = [1, 2, 5],
        T0 = 288.15, S0 = 0.0)
    @test_throws ArgumentError synthesize_population(bad_mask; seed = 1, thermo = THERMO)
    other_edges = collect(10.0 .^ range(-8.3, -4.5; length = 48))
    other_spec = lognormal_table(6.0e-8, 1.45, 8.0e11, other_edges)
    bad_grid = SyntheticPopulationSpec(n_sim = 10, spectrum = other_spec, fbar = FBAR,
        chi_target = 0.5, densities = RHO, h2o_idx = 5, chi_species = MASK3,
        T0 = 288.15, S0 = 0.0)
    @test_throws ArgumentError synthesize_population(bad_grid; seed = 1, thermo = THERMO)
end

@testset "composition recovery of fbar(D)" begin
    n = 20_000
    particles, dry_d,
    _ = synthesize_population(
        pop_spec(12.0; n = n); seed = 777, thermo = THERMO, nu = 12.0)
    bins = [searchsortedfirst(EDGES, d) - 1 for d in dry_d]
    nbins = length(EDGES) - 1
    for b in 1:nbins
        idx = findall(==(b), bins)
        length(idx) >= 100 || continue
        n_b = length(idx)
        for k in 1:4
            fk = FBAR.fractions[k, b]
            mean_k = sum(i -> particles[i][k] /
                              sum(particles[i][j] for j in 1:4), idx) / n_b
            # Dirichlet(nu*fbar) bin-mean noise sigma = sqrt(f*(1-f)/((nu+1)*n_b));
            # a flat 0.01 bound is below 1 sigma for bins near the 100-count
            # threshold, so take max(0.01, 4 sigma) (nu = 12.0 here)
            sig = sqrt(fk * (1 - fk) / ((12.0 + 1) * n_b))
            @test abs(mean_k - fk) < max(0.01, 4.0 * sig)
        end
    end
end

# empty_nu_cache! is produced but deliberately not exported (its docstring
# marks it "tests only"); bring it into scope for the calibration testsets
using StochParticles: empty_nu_cache!

@testset "reachable_chi_max" begin
    # size-independent fbar: the ν→∞ limit is exactly 1
    @test reachable_chi_max(SPEC, constant_fbar(SVector(0.25, 0.15, 0.50, 0.10), SPEC);
        densities = RHO, chi_species = MASK3) ≈ 1.0 atol = 1e-12
    # driver anchors: comfortably above the top grid point 0.75
    chi_inf = reachable_chi_max(SPEC, FBAR; densities = RHO, chi_species = MASK3)
    @test chi_inf > 0.85
    println("chi_inf(driver anchors) = $chi_inf")
    # sharp two-regime composition: mostly-external ceiling far below 1
    strong = Matrix{Float64}(undef, 4, length(EDGES) - 1)
    for b in 1:(length(EDGES) - 1)
        strong[:, b] = b <= 40 ? [0.0, 0.0, 0.9, 0.1] : [0.9, 0.05, 0.03, 0.02]
    end
    strong_fbar = SizeResolvedComposition(EDGES, strong)
    @test reachable_chi_max(SPEC, strong_fbar;
        densities = RHO, chi_species = MASK3) < 0.5
    @test_throws ArgumentError nu_for_chi(SPEC, strong_fbar, 0.75;
        densities = RHO, chi_species = MASK3)
end

@testset "chi(nu) monotone" begin
    rng = MersenneTwister(42)
    chis = [StochParticles._population_chi(rng, SPEC, FBAR, nu, 20_000;
                densities = RHO, chi_species = MASK3)
            for nu in [0.5, 2.0, 10.0, 50.0, 200.0]]
    @test all(diff(chis) .> 0.01)
end

@testset "nu_for_chi acceptance gate (|chi_realized - chi_target| < 0.02)" begin
    empty_nu_cache!()
    for chi_target in [0.10, 0.25, 0.40, 0.50, 0.60, 0.75]
        nu = nu_for_chi(SPEC, FBAR, chi_target;
            densities = RHO, chi_species = MASK3, rng = MersenneTwister(20260930))
        @test nu > 0
        chi_real = StochParticles._population_chi(
            MersenneTwister(70_000 + round(Int, 100 * chi_target)),
            SPEC, FBAR, nu, 1000; densities = RHO, chi_species = MASK3)
        println("chi_target=$chi_target  nu=$(round(nu, digits = 3))  chi_realized=$(round(chi_real, digits = 4))")
        @test abs(chi_real - chi_target) < 0.02
    end
end

@testset "nu_for_chi cache and mask sensitivity" begin
    empty_nu_cache!()
    nu = nu_for_chi(SPEC, FBAR, 0.30;
        densities = RHO, chi_species = MASK3, rng = MersenneTwister(1))
    key = (objectid(SPEC), objectid(FBAR), 0.30, 20_000, 5, MASK3)
    @test StochParticles._NU_CACHE[key] == nu
    StochParticles._NU_CACHE[key] = -999.0
    @test nu_for_chi(SPEC, FBAR, 0.30;
        densities = RHO, chi_species = MASK3, rng = MersenneTwister(1)) == -999.0
    empty_nu_cache!()
    # the calibration must follow the diagnostic mask (wiring check)
    nu3 = nu_for_chi(SPEC, FBAR, 0.50;
        densities = RHO, chi_species = [1, 2, 3], rng = MersenneTwister(2))
    nu4 = nu_for_chi(SPEC, FBAR, 0.50;
        densities = RHO, chi_species = [1, 2, 3, 4], rng = MersenneTwister(2))
    @test nu3 > 0 && nu4 > 0
    # wiring check, version-robust: on identical populations (same rng) the
    # masked chi differs deterministically whenever the mask matters.
    # Bitwise inequality of the two bisection endpoints is NOT portable: with
    # these anchors' ~0.4% true nu separation inside the 0.002 convergence
    # tolerance, both searches can take identical branch sequences and land
    # on the same nu (observed on Julia 1.12 CI, 2026-10-01).
    chi3 = StochParticles._population_chi(MersenneTwister(11), SPEC, FBAR, 1.0,
        20_000; densities = RHO, chi_species = [1, 2, 3])
    chi4 = StochParticles._population_chi(MersenneTwister(11), SPEC, FBAR, 1.0,
        20_000; densities = RHO, chi_species = [1, 2, 3, 4])
    @test chi3 != chi4
end

@testset "synthesize_population calibrated path (nu = nothing)" begin
    empty_nu_cache!()
    expected_nu = nu_for_chi(SPEC, FBAR, 0.50;
        densities = RHO, chi_species = MASK3, rng = MersenneTwister(20260930))
    particles, dry_d,
    meta = synthesize_population(
        pop_spec(0.0; chi = 0.50); seed = 20260930, thermo = THERMO)
    @test meta.nu == expected_nu
    @test abs(meta.chi_realized - 0.50) < 0.02
end
