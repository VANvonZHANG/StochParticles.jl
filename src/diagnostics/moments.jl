# src/diagnostics/moments.jl

"""
    number_concentration(sys::ParticleSystem) -> Float64

Zeroth moment: n₀ = N_sim / V_comp [particles/m³].
"""
number_concentration(sys::ParticleSystem) = sys.n_sim / sys.volume

"""
    mass_concentration(u, ::Val{A}, sys) -> Float64

First moment: total mass concentration = (Σᵢ Σₖ μₖᵢ) / V_comp [kg/m³].
"""
function mass_concentration(u::Vector{Float64}, ::Val{A}, sys::ParticleSystem) where {A}
    M = total_mass(u, Val(A), sys.n_active)
    return M / sys.volume
end

"""
    species_mass_concentration(u, species_idx, ::Val{A}, sys) -> Float64

Mass concentration of a single species: (Σᵢ μ_{species_idx,i}) / V_comp.
"""
function species_mass_concentration(
        u::Vector{Float64}, species_idx::Int, ::Val{A}, sys::ParticleSystem) where {A}
    m = 0.0
    for i in 1:sys.n_active
        μ = get_particle(u, i, Val(A))
        m += μ[species_idx]
    end
    return m / sys.volume
end

"""
    species_fractions(μ::SVector{A, Float64}) -> SVector{A, Float64}

Per-particle species mass fractions, normalized to sum to 1.
If total mass is zero, returns uniform fractions.
"""
function species_fractions(μ::SVector{A, Float64}) where {A}
    m_total = sum(μ)
    if m_total > 0
        return μ / m_total
    else
        return SVector{A, Float64}(ntuple(_ -> 1.0 / A, Val(A)))
    end
end
