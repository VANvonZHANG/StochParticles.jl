"""
    ParcelState

0-D air parcel thermodynamic state.

# Fields
- `T::Float64` — temperature [K]
- `p::Float64` — pressure [Pa]
- `qv::Float64` — water vapor mixing ratio [kg/kg]
- `S::Float64` — supersaturation (qv/qsat - 1)
"""
struct ParcelState
    T::Float64
    p::Float64
    qv::Float64
    S::Float64
end

"""
    parcel_drift(parcel, dm_w_rates, w, cp, g, rho_a, volume) -> SVector{4,Float64}

Compute the time derivatives of parcel state variables.

# Arguments
- `parcel::ParcelState` — current parcel state
- `dm_w_rates::Vector{Float64}` — H2O condensation rate for each active particle [kg/s]
- `w::Float64` — updraft velocity [m/s]
- `cp::Float64` — specific heat of air [J/kg/K]
- `g::Float64` — gravitational acceleration [m/s²]
- `rho_a::Float64` — air density [kg/m³]
- `volume::Float64` — computational volume [m³]

# Returns
- `[dT/dt, dp/dt, dqv/dt, dS/dt]`
"""
function parcel_drift(
        parcel::ParcelState,
        dm_w_rates::Vector{Float64},
        w::Float64,
        cp::Float64,
        g::Float64,
        rho_a::Float64,
        volume::Float64,
)
    total_condensation = sum(dm_w_rates)
    L_v = 2.5e6  # [J/kg]

    # Temperature: adiabatic cooling + latent heating
    dT = -g / cp * w + L_v * total_condensation / (rho_a * cp * volume)

    # Pressure: hydrostatic
    dp = -rho_a * g * w

    # Water vapor mixing ratio: depletion by condensation
    dqv = -total_condensation / (rho_a * volume)

    # Supersaturation (simplified: derivative of S = qv/qsat - 1)
    # dS/dt = (1/qsat)·dqv/dt - (qv/qsat²)·(dqsat/dT)·dT/dt
    p_sat = saturation_vapor_pressure(parcel.T)
    qsat = 0.622 * p_sat / parcel.p  # approximate
    dqsat_dT = qsat * 2.5e6 / (461.5 * parcel.T^2)  # Clausius-Clapeyron derivative
    dS = dqv / qsat - parcel.qv * dqsat_dT * dT / qsat^2

    return (T = dT, p = dp, qv = dqv, S = dS)
end
