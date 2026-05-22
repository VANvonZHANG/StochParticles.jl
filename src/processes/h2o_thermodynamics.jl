"""
    ThermodynamicsParams

Physical parameters for H2O thermodynamics calculations.

# Fields
- `κ_values::SVector{A,Float64}` — hygroscopicity κ per species
- `σ::Float64` — surface tension [N/m]
- `ρ_w::Float64` — water density [kg/m³]
- `M_w::Float64` — water molar mass [kg/mol]
- `L_v::Float64` — latent heat of vaporization [J/kg]
- `R_v::Float64` — specific gas constant for water vapor [J/kg/K]
- `D_v::Float64` — molecular diffusivity of water vapor [m²/s]
- `k_a::Float64` — thermal conductivity of air [W/m/K]
"""
struct ThermodynamicsParams{A}
    κ_values::SVector{A, Float64}
    σ::Float64
    ρ_w::Float64
    M_w::Float64
    L_v::Float64
    R_v::Float64
    D_v::Float64
    k_a::Float64
end

"""
    saturation_vapor_pressure(T::Float64) -> Float64

Saturation vapor pressure over liquid water using the Clausius-Clapeyron relation.

Reference: p_sat at T0 = 273.15 K is 611.2 Pa.
"""
function saturation_vapor_pressure(T::Float64)
    T0 = 273.15
    p_sat_0 = 611.2
    L_v = 2.5e6      # [J/kg]
    R_v = 461.5      # [J/kg/K]
    return p_sat_0 * exp((L_v / R_v) * (1.0 / T0 - 1.0 / T))
end

"""
    modified_diffusion_coefficient(thermo, T, p_sat) -> Float64

Effective mass transfer coefficient accounting for latent heat release.
Formula from Seinfeld & Pandis (Eq. 13.58).

D_v' = [1/D_v + (L_v / (k_a·T)) · (L_v/(R_v·T) - 1) · p_sat / p_v ]^(-1)

For the limit p_v → p_sat (near equilibrium), we use p_v ≈ p_sat.
"""
function modified_diffusion_coefficient(
    thermo::ThermodynamicsParams,
    T::Float64,
    p_sat::Float64,
)
    D_v = thermo.D_v
    L_v = thermo.L_v
    k_a = thermo.k_a
    R_v = thermo.R_v

    # Thermal resistance term (latent heat correction)
    # L_v must be in J/mol for unit consistency; convert from J/kg using M_w
    L_v_mol = L_v * thermo.M_w
    thermal_resistance = (L_v_mol / (k_a * T)) * (L_v / (R_v * T) - 1.0)

    # Total resistance: 1/D_v' = 1/D_v + thermal_resistance · p_sat / p_atm
    # Using standard atmospheric pressure p_atm = 101325 Pa
    p_atm = 101325.0
    total_resistance = 1.0 / D_v + thermal_resistance * (p_sat / p_atm)

    return 1.0 / total_resistance
end

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

"""
    particle_wet_radius(m_dry, m_w, densities) -> Float64

Compute wet particle radius from dry masses and water mass.
Assumes volume-additive mixing.
"""
function particle_wet_radius(
    m_dry::SVector{A, Float64},
    m_w::Float64,
    densities::SVector{A, Float64},
) where {A}
    # Total volume = dry volume + water volume
    V_total = m_w / densities[end]  # water volume
    for k in 1:(A - 1)
        V_total += m_dry[k] / densities[k]
    end
    return cbrt(3.0 * V_total / (4.0 * π))
end

"""
    equilibrium_vapor_pressure(m_dry, m_w, thermo, densities, T) -> Float64

Köhler equilibrium vapor pressure over a solution droplet.

p_eq = p_sat(T) · a_w · exp(2σ / (R_v · T · ρ_w · R))

# Arguments
- `m_dry` — dry species masses
- `m_w` — water mass
- `thermo` — ThermodynamicsParams
- `densities` — per-species densities
- `T` — temperature [K]
"""
function equilibrium_vapor_pressure(
    m_dry::SVector{A, Float64},
    m_w::Float64,
    thermo::ThermodynamicsParams{A},
    densities::SVector{A, Float64},
    T::Float64,
) where {A}
    p_sat = saturation_vapor_pressure(T)
    a_w = water_activity(m_dry, m_w, thermo.κ_values, densities)
    R = particle_wet_radius(m_dry, m_w, densities)

    # Kelvin effect
    kelvin = exp(2.0 * thermo.σ / (thermo.R_v * T * thermo.ρ_w * R))

    return p_sat * a_w * kelvin
end
