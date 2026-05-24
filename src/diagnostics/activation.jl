"""
    activation_fraction(u, sys, ::Val{A}; mode, kwargs...) -> Float64

Fraction of particles that are activated.

# Modes
- `mode=:radius_threshold` (default) — legacy method: particle is activated if
  wet radius exceeds `threshold` [m]
- `mode=:critical_supersaturation` — physics-based: particle is activated if
  environmental supersaturation `S_env` exceeds its Köhler critical
  supersaturation `Sc`

# Keyword arguments for `:radius_threshold`
- `threshold::Float64 = 1.0e-6` — activation radius threshold [m]
- `densities::SVector{A,Float64}` — per-species densities

# Keyword arguments for `:critical_supersaturation`
- `S_env::Float64` — environmental supersaturation (dimensionless, e.g. 0.002 = 0.2%)
- `thermo::ThermodynamicsParams` — thermodynamic parameters
- `densities::SVector{A,Float64}` — per-species densities
- `T::Float64` — temperature [K]

# Returns
- Fraction ∈ [0, 1]
"""
function activation_fraction(
        u::Vector{Float64},
        sys::ParticleSystem{A},
        ::Val{A};
        mode::Symbol = :radius_threshold,
        kwargs...
) where {A}
    if mode == :radius_threshold
        return _activation_fraction_radius(u, sys, Val(A); kwargs...)
    elseif mode == :critical_supersaturation
        return _activation_fraction_sc(u, sys, Val(A); kwargs...)
    else
        throw(ArgumentError("Unknown activation mode: $mode. Use :radius_threshold or :critical_supersaturation"))
    end
end

function _activation_fraction_radius(
        u::Vector{Float64},
        sys::ParticleSystem{A},
        ::Val{A};
        threshold::Float64 = 1.0e-6,
        densities::SVector{A, Float64}
) where {A}
    n_active = sys.n_active
    count = 0
    for i in 1:n_active
        μ = get_particle(u, i, Val(A))
        V_total = 0.0
        for k in 1:A
            V_total += μ[k] / densities[k]
        end
        R = cbrt(3.0 * V_total / (4.0 * π))
        if R >= threshold
            count += 1
        end
    end
    return n_active > 0 ? count / n_active : 0.0
end

function _activation_fraction_sc(
        u::Vector{Float64},
        sys::ParticleSystem{A},
        ::Val{A};
        S_env::Float64,
        thermo::ThermodynamicsParams{A},
        densities::SVector{A, Float64},
        T::Float64
) where {A}
    n_active = sys.n_active
    count = 0
    h2o_idx = findfirst(==(0.0), thermo.κ_values)  # heuristic: water has κ=0
    h2o_idx === nothing && (h2o_idx = A)

    for i in 1:n_active
        μ = get_particle(u, i, Val(A))

        # Extract dry mass (exclude water)
        m_dry = zero(SVector{A, Float64})
        for k in 1:A
            if k != h2o_idx
                m_dry = setindex(m_dry, μ[k], k)
            end
        end

        # Compute critical supersaturation from dry properties
        Sc = critical_supersaturation(m_dry, thermo, densities, T)

        # Activated if S_env > Sc
        if S_env > Sc
            count += 1
        end
    end
    return n_active > 0 ? count / n_active : 0.0
end

"""
    cloud_droplet_concentration(u, sys, ::Val{A}; mode, kwargs...) -> Float64

Number concentration of activated cloud droplets [m⁻³].

See `activation_fraction` for supported modes and keyword arguments.
"""
function cloud_droplet_concentration(
        u::Vector{Float64},
        sys::ParticleSystem{A},
        ::Val{A};
        mode::Symbol = :radius_threshold,
        kwargs...
) where {A}
    frac = activation_fraction(u, sys, Val(A); mode = mode, kwargs...)
    return frac * sys.n_active / sys.volume
end
