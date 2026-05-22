"""
    water_activity(m_dry, m_w, κ_values, densities) -> Float64

Compute water activity using κ-Köhler theory (Petters & Kreidenweis, 2007).

# Arguments
- `m_dry::SVector{A}` — dry mass of each species [kg]
- `m_w::Float64` — water mass [kg]
- `κ_values::SVector{A}` — hygroscopicity parameter κ for each species
- `densities::SVector{A}` — density of each species [kg/m³]

# Returns
- Water activity a_w ∈ [0, 1]

# Formula
    a_w = V_w / (V_w + V_dry · κ_mix)

where V_w = m_w / ρ_w, V_dry = Σ m_k / ρ_k, and κ_mix = Σ ε_k · κ_k
with ε_k = V_k / V_dry.
"""
function water_activity(
    m_dry::SVector{A, Float64},
    m_w::Float64,
    κ_values::SVector{A, Float64},
    densities::SVector{A, Float64},
) where {A}
    ρ_w = densities[end]  # last species is water
    V_w = m_w / ρ_w

    # Dry volume and per-species volume fractions
    V_dry = 0.0
    κ_mix = 0.0
    for k in 1:(A - 1)  # exclude water (last species)
        V_k = m_dry[k] / densities[k]
        V_dry += V_k
        κ_mix += V_k * κ_values[k]
    end

    if V_dry ≈ 0.0
        return 1.0  # pure water droplet
    end

    κ_mix /= V_dry  # volume-fraction-weighted average
    denominator = V_w + V_dry * κ_mix

    return V_w / denominator
end
