# src/utils/binning.jl

"""
    bin_size_distribution(diams, bin_edges) -> Vector{Int}

Histogram particle diameters into bins defined by `bin_edges`.
Each diameter `d` is counted in the first bin where `bin_edges[i] <= d < bin_edges[i+1]`.
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
    counts = zeros(Int, length(bin_edges) - 1)
    for d in diams
        for i in 1:(length(bin_edges) - 1)
            if bin_edges[i] <= d < bin_edges[i + 1]
                counts[i] += 1
                break
            end
        end
    end
    return counts
end
