# src/utils/parameters.jl

"""
    standard_aerosol_atmosphere() -> AtmosphericParameters

Standard atmospheric conditions for near-surface aerosol simulations.
"""
function standard_aerosol_atmosphere()
    return AtmosphericParameters(
        293.15,      # T [K]
        101325.0,    # P [Pa]
        1.225,       # rho_f [kg/m³]
        1.81e-5,     # mu_f [Pa·s]
        1.48e-5,     # nu [m²/s]
        1800.0,      # rho_p [kg/m³]
        9.81         # g [m/s²]
    )
end

"""
    standard_cloud_atmosphere() -> AtmosphericParameters

Standard atmospheric conditions for cumulus cloud simulations (~2 km altitude).
"""
function standard_cloud_atmosphere()
    return AtmosphericParameters(
        288.15,      # T [K]
        80000.0,     # P [Pa]
        1.06,        # rho_f [kg/m³]
        1.75e-5,     # mu_f [Pa·s]
        1.65e-5,     # nu [m²/s]
        1000.0,      # rho_p [kg/m³]
        9.81         # g [m/s²]
    )
end
