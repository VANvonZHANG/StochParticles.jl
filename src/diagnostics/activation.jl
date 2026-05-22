"""
    activation_fraction(u, sys, ::Val{A}; threshold, densities) -> Float64

Fraction of particles with wet radius exceeding the activation threshold.

# Arguments
- `u` — ODE state vector
- `sys` — ParticleSystem
- `threshold::Float64` — activation radius threshold [m] (default: 1.0e-6)
- `densities::SVector{A,Float64}` — per-species densities

# Returns
- Fraction ∈ [0, 1]
"""
function activation_fraction(
        u::Vector{Float64},
        sys::ParticleSystem{A},
        ::Val{A};
        threshold::Float64 = 1.0e-6,
        densities::SVector{A, Float64},
) where {A}
    n_active = sys.n_active
    count = 0
    for i in 1:n_active
        μ = get_particle(u, i, Val(A))
        # Compute wet radius
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

"""
    cloud_droplet_concentration(u, sys, ::Val{A}; threshold, densities) -> Float64

Number concentration of activated cloud droplets [m⁻³].

# Arguments
- `u` — ODE state vector
- `sys` — ParticleSystem
- `threshold::Float64` — activation radius threshold [m]
- `densities::SVector{A,Float64}` — per-species densities

# Returns
- Droplet number concentration [m⁻³]
"""
function cloud_droplet_concentration(
        u::Vector{Float64},
        sys::ParticleSystem{A},
        ::Val{A};
        threshold::Float64 = 1.0e-6,
        densities::SVector{A, Float64},
) where {A}
    frac = activation_fraction(u, sys, Val(A);
        threshold = threshold, densities = densities)
    return frac * sys.n_active / sys.volume
end
