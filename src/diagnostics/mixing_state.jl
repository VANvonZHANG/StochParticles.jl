# src/diagnostics/mixing_state.jl

"""
    shannon_entropy(p::SVector{A, Float64}) -> Float64

Shannon entropy of a probability distribution.
H = -Σ p_k * ln(p_k) for p_k > 0.
"""
function shannon_entropy(p::SVector{A, Float64}) where {A}
    h = 0.0
    for k in 1:A
        p_k = p[k]
        if p_k > 0
            h -= p_k * log(p_k)
        end
    end
    return h
end

"""
    mixing_state_index(u::Vector{Float64}, sys::ParticleSystem) -> Float64

Compute the population mixing state parameter χ ∈ [0, 1].

χ = 0  → fully externally mixed (each particle is pure)
χ = 1  → fully internally mixed (all particles have same composition)

Returns 1.0 for single-species systems (A = 1).

Reference: Riemer et al. (2019) framework.
"""
function mixing_state_index(u::Vector{Float64}, sys::ParticleSystem{A}) where {A}
    A == 1 && return 1.0
    A_val = Val(A)
    n = sys.n_active

    # Per-particle species fractions
    fractions = [species_fractions(get_particle(u, i, A_val)) for i in 1:n]

    # Population-average species fractions (mass-weighted)
    m_total = 0.0
    f_bar = zeros(SVector{A, Float64})
    for i in 1:n
        μ_i = get_particle(u, i, A_val)
        m_i = sum(μ_i)
        m_total += m_i
        f_bar += species_fractions(μ_i) * m_i
    end
    f_bar /= m_total

    # Actual diversity: average per-particle entropy
    D_eps = sum(shannon_entropy.(fractions)) / n

    # External reference: pure particles have zero entropy
    D_alpha = 0.0

    # Internal reference: all particles have bulk-average composition
    D_gamma = shannon_entropy(f_bar)

    return (D_alpha - D_eps) / (D_alpha - D_gamma)
end

"""
    particle_mixing_entropy(u::Vector{Float64}, sys::ParticleSystem) -> Vector{Float64}

Shannon entropy of species fractions for each active particle.
Higher entropy = more mixed. Zero = pure single-species particle.
"""
function particle_mixing_entropy(u::Vector{Float64}, sys::ParticleSystem{A}) where {A}
    A_val = Val(A)
    [shannon_entropy(species_fractions(get_particle(u, i, A_val))) for i in 1:sys.n_active]
end
