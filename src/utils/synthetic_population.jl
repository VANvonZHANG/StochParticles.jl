# src/utils/synthetic_population.jl
#
# M2 synthetic-population initialization: tabulated size spectra,
# size-resolved mean composition, Dirichlet(ν·f̄(D)) population synthesis,
# and Monte-Carlo ν(χ) calibration (GCVI closure blueprint contract 2).

using StaticArrays
using Distributions

"""
    TabulatedSpectrum(bin_edges, dNdlogD)

Tabulated dry number spectrum: `dNdlogD[b]` is dN/dlogD [m⁻³] of bin `b`
(geometric grid; uniform-in-log-D bins are assumed for sampling). Future
SMPS measurements drop in directly as the two vectors.
"""
struct TabulatedSpectrum
    bin_edges::Vector{Float64}
    dNdlogD::Vector{Float64}
    function TabulatedSpectrum(bin_edges::Vector{Float64}, dNdlogD::Vector{Float64})
        length(bin_edges) >= 2 ||
            throw(ArgumentError("at least one bin required"))
        length(bin_edges) == length(dNdlogD) + 1 ||
            throw(ArgumentError("bin_edges must have exactly one more entry than dNdlogD " *
                                "(got $(length(bin_edges)) and $(length(dNdlogD)))"))
        all(i -> bin_edges[i] < bin_edges[i + 1], 1:(length(bin_edges) - 1)) ||
            throw(ArgumentError("bin_edges must be strictly increasing"))
        all(>(0.0), bin_edges) ||
            throw(ArgumentError("bin_edges must be positive"))
        all(>=(0.0), dNdlogD) ||
            throw(ArgumentError("dNdlogD must be non-negative"))
        return new(bin_edges, dNdlogD)
    end
end

"""
    number_concentration(s::TabulatedSpectrum) -> Float64

Total number concentration [m⁻³]: the spectrum integral over all bins.
"""
function number_concentration(s::TabulatedSpectrum)
    total = 0.0
    for b in eachindex(s.dNdlogD)
        total += s.dNdlogD[b] * (log(s.bin_edges[b + 1]) - log(s.bin_edges[b]))
    end
    return total
end

"""
    +(a::TabulatedSpectrum, b::TabulatedSpectrum) -> TabulatedSpectrum

Bin-wise sum of two spectra on the same grid (multi-modal spectra).
"""
function Base.:+(a::TabulatedSpectrum, b::TabulatedSpectrum)
    a.bin_edges == b.bin_edges ||
        throw(ArgumentError("bin_edges must match to add spectra"))
    return TabulatedSpectrum(a.bin_edges, a.dNdlogD .+ b.dNdlogD)
end

_stdnorm_cdf(z::Float64) = 0.5 * (1.0 + erf(z / sqrt(2.0)))

"""
    lognormal_table(d_g, sigma_g, N, bin_edges) -> TabulatedSpectrum

Bake a lognormal mode onto the grid: per-bin counts from the analytic CDF
difference (not bin-center evaluation), so `number_concentration` recovers
`N` exactly. `d_g` is the median (geometric mean) diameter [m], `sigma_g`
the geometric standard deviation, `N` the mode number concentration [m⁻³].
"""
function lognormal_table(d_g::Float64, sigma_g::Float64, N::Real,
        bin_edges::Vector{Float64})
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    sigma_g > 0 || throw(ArgumentError("sigma_g must be positive, got $sigma_g"))
    ln_dg = log(d_g)
    ln_sg = log(sigma_g)
    nbins = length(bin_edges) - 1
    dNdlogD = Vector{Float64}(undef, nbins)
    for b in 1:nbins
        count = N * (_stdnorm_cdf((log(bin_edges[b + 1]) - ln_dg) / ln_sg) -
                 _stdnorm_cdf((log(bin_edges[b]) - ln_dg) / ln_sg))
        dNdlogD[b] = count / (log(bin_edges[b + 1]) - log(bin_edges[b]))
    end
    return TabulatedSpectrum(bin_edges, dNdlogD)
end

"""
    _spectrum_bin_probs(s) -> Vector{Float64}

Normalized bin probabilities ∝ dNdlogD·ΔlogD (internal).
"""
function _spectrum_bin_probs(s::TabulatedSpectrum)
    nbins = length(s.dNdlogD)
    probs = Vector{Float64}(undef, nbins)
    total = 0.0
    for b in 1:nbins
        total += s.dNdlogD[b] * (log(s.bin_edges[b + 1]) - log(s.bin_edges[b]))
    end
    total > 0 ||
        throw(ArgumentError("spectrum has zero number concentration"))
    for b in 1:nbins
        probs[b] = s.dNdlogD[b] * (log(s.bin_edges[b + 1]) - log(s.bin_edges[b])) / total
    end
    return probs
end

"""
    _sample_dry_diameters(rng, s, n) -> (diameters, bins)

Inverse-CDF sampling: pick a bin from the cumulative bin probabilities,
then draw log-D uniformly within the bin (internal).
"""
function _sample_dry_diameters(rng::AbstractRNG, s::TabulatedSpectrum, n::Int)
    probs = _spectrum_bin_probs(s)
    cum = cumsum(probs)
    edges = s.bin_edges
    diameters = Vector{Float64}(undef, n)
    bins = Vector{Int}(undef, n)
    for i in 1:n
        b = min(searchsortedfirst(cum, rand(rng)), length(probs))
        bins[i] = b
        ln_lo = log(edges[b])
        diameters[i] = exp(ln_lo + (log(edges[b + 1]) - ln_lo) * rand(rng))
    end
    return diameters, bins
end

"""
    SizeResolvedComposition(bin_edges, fractions)

Size-resolved mean dry composition: `fractions[:, b]` is the mean dry mass
fraction vector (K species) of bin `b`; piecewise-constant within a bin
(same discretization as `TabulatedSpectrum`).
"""
struct SizeResolvedComposition
    bin_edges::Vector{Float64}
    fractions::Matrix{Float64}
    function SizeResolvedComposition(bin_edges::Vector{Float64},
            fractions::Matrix{Float64})
        size(fractions, 2) == length(bin_edges) - 1 ||
            throw(ArgumentError("fractions must have one column per bin " *
                                "(got $(size(fractions, 2)) columns for $(length(bin_edges) - 1) bins)"))
        size(fractions, 1) >= 1 ||
            throw(ArgumentError("at least one species required"))
        all(i -> bin_edges[i] < bin_edges[i + 1], 1:(length(bin_edges) - 1)) ||
            throw(ArgumentError("bin_edges must be strictly increasing"))
        all(>=(0.0), fractions) ||
            throw(ArgumentError("fractions must be non-negative"))
        for b in axes(fractions, 2)
            colsum = sum(fractions[:, b])
            abs(colsum - 1.0) <= 1e-10 ||
                throw(ArgumentError("fraction columns must sum to 1 (column $b sums to $colsum)"))
        end
        return new(bin_edges, fractions)
    end
end

"""
    constant_fbar(fbar, spectrum) -> SizeResolvedComposition

Size-independent mean composition (M0 compatibility).
"""
function constant_fbar(fbar::SVector{K, Float64},
        spectrum::TabulatedSpectrum) where {K}
    nbins = length(spectrum.bin_edges) - 1
    mat = Matrix{Float64}(undef, K, nbins)
    for b in 1:nbins
        mat[:, b] .= fbar
    end
    return SizeResolvedComposition(spectrum.bin_edges, mat)
end

function _anchor_interp(ln_d::Float64, ln_anchors::Vector{Float64},
        values::Vector{Float64})
    # linear in log-D with flat extrapolation (anchors bracket the median, so
    # flat extrapolation covers the far tails of the grid)
    ln_d <= ln_anchors[1] && return values[1]
    ln_d >= ln_anchors[end] && return values[end]
    j = findfirst(i -> ln_anchors[i + 1] >= ln_d, 1:(length(ln_anchors) - 1))
    t = (ln_d - ln_anchors[j]) / (ln_anchors[j + 1] - ln_anchors[j])
    return values[j] + t * (values[j + 1] - values[j])
end

"""
    SizeResolvedComposition(spectrum; anchors) -> SizeResolvedComposition

Build from anchor points `[(D, f̄), …]` (sorted by D, at least two, bracketing
the spectrum's number-median diameter): each species is interpolated linearly
in log-D at the bin geometric centers, then each column is renormalized to
sum to 1.
"""
function SizeResolvedComposition(spectrum::TabulatedSpectrum;
        anchors::Vector{Tuple{Float64, SVector{K, Float64}}}) where {K}
    length(anchors) >= 2 ||
        throw(ArgumentError("at least two anchors required, got $(length(anchors))"))
    ds = [a[1] for a in anchors]
    all(>=(0.0), ds) ||
        throw(ArgumentError("anchor diameters must be positive"))
    issorted(ds) ||
        throw(ArgumentError("anchors must be sorted by diameter"))
    edges = spectrum.bin_edges
    cum = cumsum(_spectrum_bin_probs(spectrum))
    med = findfirst(c -> c >= 0.5, cum)
    med === nothing &&
        throw(ArgumentError("spectrum probability mass never reaches 0.5; " *
                            "cannot locate the number-median bin"))
    d_med = sqrt(edges[med] * edges[med + 1])
    ds[1] <= d_med ||
        throw(ArgumentError("first anchor ($(ds[1]) m) must lie at or below the " *
                            "number-median diameter ($(d_med) m)"))
    ds[end] >= d_med ||
        throw(ArgumentError("last anchor ($(ds[end]) m) must lie at or above the " *
                            "number-median diameter ($(d_med) m)"))
    all(a -> length(a[2]) == K, anchors) ||
        throw(ArgumentError("all anchor fraction vectors must have length $K"))
    ln_anchors = log.(ds)
    nbins = length(edges) - 1
    mat = Matrix{Float64}(undef, K, nbins)
    for b in 1:nbins
        ln_center = 0.5 * (log(edges[b]) + log(edges[b + 1]))
        col = [_anchor_interp(ln_center, ln_anchors, [a[2][k] for a in anchors])
               for k in 1:K]
        mat[:, b] .= col ./ sum(col)
    end
    return SizeResolvedComposition(edges, mat)
end

"""
    SyntheticPopulationSpec(; n_sim, spectrum, fbar, chi_target, densities,
                            h2o_idx, chi_species, T0, S0)

Description of a synthetic initial population (blueprint contract 2).
`densities` includes water in the last slot; `h2o_idx` must equal the number
of dry species + 1; `chi_species` masks the dry species used by the closure
χ (e.g. `[1, 2, 3]` = AS, AN, OA).
"""
Base.@kwdef struct SyntheticPopulationSpec
    n_sim::Int
    spectrum::TabulatedSpectrum
    fbar::SizeResolvedComposition
    chi_target::Float64
    densities::SVector{A, Float64} where {A}
    h2o_idx::Int
    chi_species::Vector{Int}
    T0::Float64
    S0::Float64
end

"""
    _assemble_population(rng, spectrum, fbar, nu, n, dry_densities)
        -> (particles, diameters)

Shared sampling path of `synthesize_population` and the `nu_for_chi`
Monte-Carlo calibration: spectrum inverse-CDF diameters + Dirichlet(ν·f̄(D))
dry fractions + volume-additive mass assembly (water slot = 0). Internal.
"""
function _assemble_population(rng::AbstractRNG, spectrum::TabulatedSpectrum,
        fbar::SizeResolvedComposition, nu::Float64, n::Int,
        dry_densities::SVector{K, Float64}) where {K}
    diameters, bins = _sample_dry_diameters(rng, spectrum, n)
    particles = Vector{SVector{K + 1, Float64}}(undef, n)
    for i in 1:n
        f = rand(rng, Dirichlet(nu .* fbar.fractions[:, bins[i]]))
        rho_eff = 1.0 / sum(f[k] / dry_densities[k] for k in 1:K)
        m_total = (pi / 6.0) * diameters[i]^3 * rho_eff
        particles[i] = SVector{K + 1, Float64}(m_total .* f..., 0.0)
    end
    return particles, diameters
end

"""
    synthesize_population(spec; seed, thermo, nu = nothing)
        -> (particles, dry_diameters, meta)

Generate the initial population: inverse-CDF dry diameters from the
spectrum, Dirichlet(ν·f̄(D)) dry compositions, volume-additive mass
assembly, haze pre-equilibration at (T0, S0). `nu = nothing` calibrates ν
for `spec.chi_target` via `nu_for_chi` (requires a reachable target); an
explicit ν skips calibration. `meta = (chi_realized, nu, seed)`.
"""
function synthesize_population(spec::SyntheticPopulationSpec;
        seed::Integer, thermo::ThermodynamicsParams,
        nu::Union{Nothing, Float64} = nothing)
    K = size(spec.fbar.fractions, 1)
    A = K + 1
    spec.h2o_idx == A ||
        throw(ArgumentError("h2o_idx must be $A (dry species occupy slots 1:$K)"))
    length(spec.densities) == A ||
        throw(ArgumentError("densities must have length $A (dry species + water)"))
    all(k -> 1 <= k <= K, spec.chi_species) ||
        throw(ArgumentError("chi_species must index dry species 1:$K, got $(spec.chi_species)"))
    spec.fbar.bin_edges == spec.spectrum.bin_edges ||
        throw(ArgumentError("fbar bin_edges must match spectrum bin_edges"))

    rng = MersenneTwister(seed)
    nu_used = nu === nothing ?
              nu_for_chi(spec.spectrum, spec.fbar, spec.chi_target;
        densities = spec.densities, chi_species = spec.chi_species,
        rng = rng) :
              Float64(nu)
    dry_densities = SVector{K, Float64}(spec.densities[1:K])
    particles,
    dry_diameters = _assemble_population(
        rng, spec.spectrum, spec.fbar, nu_used, spec.n_sim, dry_densities)

    # dry-only chi is invariant to the equilibration water (masked species),
    # so chi_realized is computed on the final initial state
    pre_equilibrate!(particles, thermo, spec.densities, spec.T0,
        saturation_vapor_pressure(spec.T0) * (1.0 + spec.S0); h2o_idx = A)

    volume = spec.n_sim / number_concentration(spec.spectrum)
    gas = PrescribedProfile([0.0], [spec.T0], [spec.S0])
    sys = ParticleSystem(Val(A), spec.n_sim, volume, gas)
    chi_realized = mixing_state_index(make_u0(particles), sys;
        species = spec.chi_species)
    return particles, dry_diameters,
    (chi_realized = chi_realized, nu = nu_used, seed = Int(seed))
end

_entropy(p) = -sum(x -> x > 0 ? x * log(x) : 0.0, p)

"""
    reachable_chi_max(spectrum, fbar; densities, chi_species) -> Float64

χ at ν → ∞: every particle carries exactly its bin's mean composition
f̄(D). Exact for piecewise-constant f̄ and mirrors the `mixing_state_index`
convention: D_eps with count-share weights (unweighted per-particle
average), D_gamma with masked-dry-mass shares from the within-bin
log-uniform sampling model, `E[D³|b] = (e_hi³ − e_lo³)/(3·Δln b)`.
Size-independent f̄ gives exactly 1.0.
"""
function reachable_chi_max(spectrum::TabulatedSpectrum,
        fbar::SizeResolvedComposition;
        densities::SVector, chi_species::Vector{Int})
    K = size(fbar.fractions, 1)
    edges = spectrum.bin_edges
    nbins = length(edges) - 1
    probs = _spectrum_bin_probs(spectrum)
    fmask = Vector{Vector{Float64}}(undef, nbins)
    mass = Vector{Float64}(undef, nbins)
    for b in 1:nbins
        col = fbar.fractions[:, b]
        masked_sum = sum(col[chi_species])
        masked_sum > 0 ||
            throw(ArgumentError("bin $b has zero masked dry mass; chi is undefined"))
        fmask[b] = col[chi_species] ./ masked_sum
        e3 = (edges[b + 1]^3 - edges[b]^3) /
             (3.0 * (log(edges[b + 1]) - log(edges[b])))
        rho_eff = 1.0 / sum(col[k] / densities[k] for k in 1:K)
        mass[b] = probs[b] * e3 * rho_eff * masked_sum
    end
    d_eps = sum(probs[b] * _entropy(fmask[b]) for b in 1:nbins)
    w = mass ./ sum(mass)
    d_gamma = _entropy(sum(w[b] * fmask[b] for b in 1:nbins))
    d_gamma > 0 || return NaN
    return d_eps / d_gamma
end

"""
    _population_chi(rng, spectrum, fbar, nu, n; densities, chi_species)

χ of one finite population drawn through the exact `synthesize_population`
sampling path (dry masses only — masked χ is water-invariant). Internal.
"""
function _population_chi(rng::AbstractRNG, spectrum::TabulatedSpectrum,
        fbar::SizeResolvedComposition, nu::Float64, n::Int;
        densities::SVector, chi_species::Vector{Int})
    K = size(fbar.fractions, 1)
    dry_densities = SVector{K, Float64}(densities[1:K])
    particles, _ = _assemble_population(rng, spectrum, fbar, nu, n, dry_densities)
    volume = n / number_concentration(spectrum)
    gas = PrescribedProfile([0.0], [288.15], [0.0])
    sys = ParticleSystem(Val(K + 1), n, volume, gas)
    return mixing_state_index(make_u0(particles), sys; species = chi_species)
end

# ν(χ) calibration cache; assumes spectrum/fbar objects are not mutated
# after construction (keys use objectid)
const _NU_CACHE = Dict{
    Tuple{UInt64, UInt64, Float64, Int, Int, Vector{Int}}, Float64}()

"""
    empty_nu_cache!()

Clear the `nu_for_chi` calibration cache (tests only).
"""
empty_nu_cache!() = empty!(_NU_CACHE)

"""
    nu_for_chi(spectrum, fbar, chi_target; densities, chi_species,
               n_mc = 20_000, n_draws = 5, rng) -> Float64

Calibrate the Dirichlet concentration ν reaching `chi_target` in E[χ]:
bisection in log ν (χ(ν) is monotone in expectation), each evaluation the
mean χ of `n_draws` Monte-Carlo populations of `n_mc` particles drawn
through the exact `synthesize_population` sampling path and scored with the
same `mixing_state_index` diagnostic. Results cached per
(spectrum, fbar, chi_target, n_mc, n_draws, chi_species). Throws
`ArgumentError` when the target is unreachable
(`chi_target ≥ reachable_chi_max − 0.01`).
"""
function nu_for_chi(spectrum::TabulatedSpectrum,
        fbar::SizeResolvedComposition, chi_target::Float64;
        densities::SVector, chi_species::Vector{Int},
        n_mc::Int = 20_000, n_draws::Int = 5,
        rng::AbstractRNG = MersenneTwister(20260930))
    key = (objectid(spectrum), objectid(fbar), chi_target, n_mc, n_draws,
        chi_species)
    haskey(_NU_CACHE, key) && return _NU_CACHE[key]
    chi_inf = reachable_chi_max(spectrum, fbar;
        densities = densities, chi_species = chi_species)
    chi_target < chi_inf - 0.01 || throw(ArgumentError(
        "chi_target=$chi_target is unreachable for this spectrum/fbar: " *
        "upper bound chi_inf=$(round(chi_inf, digits = 4)); " *
        "valid range is (0, $(round(chi_inf, digits = 4)))"))
    chi_of(nu::Float64) = begin
        s = 0.0
        for _ in 1:n_draws
            s += _population_chi(rng, spectrum, fbar, nu, n_mc;
                densities = densities, chi_species = chi_species)
        end
        s / n_draws
    end
    lo, hi = 1.0e-2, 1.0e6
    chi_lo = chi_of(lo)
    chi_lo < chi_target || throw(ArgumentError(
        "lower bracket nu=$lo already gives chi=$(round(chi_lo, digits = 4)) " *
        ">= target $chi_target"))
    nu_star = sqrt(lo * hi)
    for _ in 1:100
        nu_star = sqrt(lo * hi)
        chi_mid = chi_of(nu_star)
        if chi_mid < chi_target
            lo = nu_star
        else
            hi = nu_star
        end
        (abs(chi_mid - chi_target) < 0.002 || hi / lo < 1.001) && break
    end
    _NU_CACHE[key] = nu_star
    return nu_star
end
