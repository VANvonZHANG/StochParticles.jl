# src/diagnostics/mixing_state.jl

"""
    shannon_entropy(p) -> Float64

Shannon entropy of a probability distribution.
H = -Σ p_k * ln(p_k) for p_k > 0. Accepts SVector or Vector.
"""
function shannon_entropy(p::AbstractVector{<:Real})
    h = 0.0
    @inbounds for k in eachindex(p)
        p_k = p[k]
        if p_k > 0
            h -= p_k * log(p_k)
        end
    end
    return h
end

"""
    mixing_state_index(u, sys; species = nothing) -> Float64

Compute the population mixing state parameter χ ∈ [0, 1].

χ = 0  → fully externally mixed (each particle is pure)
χ = 1  → fully internally mixed (all particles have same composition)

With `species` (e.g. `[1, 2, 3]`), χ is computed over the selected species
only — the dry-composition χ when H₂O (and tracers) are excluded. Particles
with zero mass in the selected species are skipped (zero mass weight, no
entropy contribution).

Note: the per-particle diversity uses an unweighted particle average,
following the original implementation; Riemer & West (2013) use mass weights.

Returns 1.0 for single-species systems (A = 1) and NaN for empty populations.
"""
function mixing_state_index(
        u::Vector{Float64}, sys::ParticleSystem{A};
        species::Union{Nothing, AbstractVector{Int}} = nothing) where {A}
    A == 1 && return 1.0
    if species !== nothing
        all(k -> 1 <= k <= A, species) ||
            throw(ArgumentError("species indices must be in 1:$A, got $species"))
        length(unique(species)) == length(species) ||
            throw(ArgumentError("species indices must be unique, got $species"))
    end
    idx = species === nothing ? (1:A) : species

    n = sys.n_active
    m_total = 0.0
    f_bar_acc = zeros(Float64, length(idx))
    ent_sum = 0.0
    n_valid = 0
    for i in 1:n
        μ = get_particle(u, i, Val(A))
        m_i = 0.0
        for k in idx
            m_i += μ[k]
        end
        m_i > 0 || continue
        ent_i = 0.0
        for (j, k) in enumerate(idx)
            f_ik = μ[k] / m_i
            f_bar_acc[j] += μ[k]
            if f_ik > 0
                ent_i -= f_ik * log(f_ik)
            end
        end
        ent_sum += ent_i
        n_valid += 1
        m_total += m_i
    end
    (m_total > 0 && n_valid > 0) || return NaN
    f_bar = f_bar_acc ./ m_total
    D_eps = ent_sum / n_valid
    D_gamma = shannon_entropy(f_bar)
    D_gamma > 0 || return NaN
    chi = D_eps / D_gamma
    return chi == 0.0 ? 0.0 : chi
end

"""
    particle_mixing_entropy(u::Vector{Float64}, sys::ParticleSystem) -> Vector{Float64}

Shannon entropy of species fractions for each active particle.
Higher entropy = more mixed. Zero = pure single-species particle.
"""
function particle_mixing_entropy(u::Vector{Float64}, sys::ParticleSystem{A}) where {A}
    A_val = Val(A)
    [shannon_entropy(species_fractions(get_particle(u, i, A_val)))
     for i in 1:(sys.n_active)]
end
