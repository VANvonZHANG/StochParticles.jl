# src/utils/synthetic_population.jl
#
# M2 synthetic-population initialization: tabulated size spectra,
# size-resolved mean composition, Dirichlet(ν·f̄(D)) population synthesis,
# and Monte-Carlo ν(χ) calibration (GCVI closure blueprint contract 2).

using StaticArrays

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
        all(>=(0.0), bin_edges) ||
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
