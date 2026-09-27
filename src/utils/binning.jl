# src/utils/binning.jl
using StatsBase

"""
    bin_size_distribution(diams, bin_edges) -> Vector{Int}

Histogram particle diameters into bins defined by `bin_edges`.
Each diameter `d` is counted in the first bin where `bin_edges[i] <= d < bin_edges[i+1]`.

Uses `StatsBase.fit(Histogram, ...)` internally.
"""
function bin_size_distribution(diams::Vector{Float64}, bin_edges::Vector{Float64})
    if length(bin_edges) < 2
        throw(ArgumentError("bin_edges must have at least 2 elements"))
    end
    for i in 2:length(bin_edges)
        if bin_edges[i] <= bin_edges[i - 1]
            throw(ArgumentError("bin_edges must be strictly increasing"))
        end
    end
    h = fit(Histogram, diams, bin_edges; closed = :left)
    return h.weights
end

"""
    dNdlogD_from_diameters(diameters, bin_edges, volume) -> Vector{Float64}

dN/dlog₁₀D spectrum [m⁻⁴] from per-particle diameters [m], log-spaced bin
edges [m] (strictly increasing), and computational volume [m³].
"""
function dNdlogD_from_diameters(diameters, bin_edges, volume::Real)
    volume > 0.0 || throw(ArgumentError("volume must be positive"))
    counts = Float64.(bin_size_distribution(
                          Float64.(collect(diameters)), Float64.(collect(bin_edges))))
    dlogD = diff(log10.(Float64.(collect(bin_edges))))
    return counts ./ dlogD ./ volume
end
