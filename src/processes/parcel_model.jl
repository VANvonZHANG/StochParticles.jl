"""
    ParcelState

Live 0-D air-parcel thermodynamic state — the last three slots `[T, p, qv]`
of the ODE state vector when a `ParcelProcess` is assembled (blueprint M3,
spec §2.1). Supersaturation is diagnosed (`parcel_supersaturation`), never
prognostic.

Physical constants (dry-air approximations, spec §2.2 — deltas against
pyrcel are checked via the formulation checklist in the M3 quantification;
S_max discrepancies beyond tolerance are investigated there first):
- `R_DRY_AIR = 287.05` [J/kg/K]
- `CP_DRY_AIR = 1005.0` [J/kg/K]
- `EPSILON_MA = 0.622` (M_w / M_dryair)
"""
mutable struct ParcelState
    T::Float64     # temperature [K]
    p::Float64     # pressure [Pa]
    qv::Float64    # water vapor mixing ratio [kg/kg dry air]
end

const R_DRY_AIR = 287.05
const CP_DRY_AIR = 1005.0
const EPSILON_MA = 0.622

"""
    parcel_supersaturation(parcel) -> Float64

Diagnosed supersaturation S = qv·p / (ε·p_sat(T)) − 1.
"""
function parcel_supersaturation(parcel::ParcelState)
    parcel.qv * parcel.p / (EPSILON_MA * saturation_vapor_pressure(parcel.T)) - 1.0
end

"""
    parcel_drift(parcel, total_cond_rate, w, m_air; thermo, g = 9.81) -> SVector{3}

Parcel tendencies `[dT, dp, dqv]` (spec §2.2):
- `dT = −g·w/c_p + L_v·Σṁ_w / (m_air·c_p)` (adiabatic cooling + latent heat)
- `dp = −ρ_a·g·w` with **diagnosed** ρ_a = p/(R_d·T) (live density, not fixed)
- `dqv = −Σṁ_w / m_air`

`m_air` is the FIXED dry-air mass of the tracked sample (ρ_a0·V, material
invariant of a Lagrangian parcel); `total_cond_rate` is the summed H2O
condensation rate [kg/s] over all active particles (positive = condensation
onto particles).
"""
function parcel_drift(parcel::ParcelState, total_cond_rate::Float64, w::Float64,
        m_air::Float64; thermo::ThermodynamicsParams, g::Float64 = 9.81)
    rho_a = parcel.p / (R_DRY_AIR * parcel.T)
    dT = -g / CP_DRY_AIR * w + thermo.L_v * total_cond_rate / (m_air * CP_DRY_AIR)
    dp = -rho_a * g * w
    dqv = -total_cond_rate / m_air
    return SVector{3, Float64}(dT, dp, dqv)
end

"""
    extract_parcel(u, n_sim, A) -> ParcelState

Read the 3-slot parcel tail `[T, p, qv]` at offset `n_sim*A`.
"""
function extract_parcel(u::Vector{Float64}, n_sim::Int, A::Int)
    off = n_sim * A
    return ParcelState(u[off + 1], u[off + 2], u[off + 3])
end

"""
    set_parcel_state!(u, parcel, n_sim, A)

Write parcel STATE values into the tail (u0 initialization).
"""
function set_parcel_state!(u::Vector{Float64}, parcel::ParcelState, n_sim::Int, A::Int)
    off = n_sim * A
    u[off + 1] = parcel.T
    u[off + 2] = parcel.p
    u[off + 3] = parcel.qv
    nothing
end

"""
    set_parcel_drift!(du, drift, n_sim, A)

Write parcel DERIVATIVES `[dT, dp, dqv]` into the tail of `du`.
"""
function set_parcel_drift!(du::Vector{Float64}, drift::SVector{3, Float64},
        n_sim::Int, A::Int)
    off = n_sim * A
    du[off + 1] = drift[1]
    du[off + 2] = drift[2]
    du[off + 3] = drift[3]
    nothing
end

# Val-form conveniences: make_ode_func holds the species count as Val{A}
function extract_parcel(u::Vector{Float64}, n_sim::Int, ::Val{A}) where {A}
    extract_parcel(u, n_sim, A)
end
function set_parcel_state!(u::Vector{Float64}, parcel::ParcelState, n_sim::Int, ::Val{A}) where {A}
    set_parcel_state!(u, parcel, n_sim, A)
end
function set_parcel_drift!(du::Vector{Float64}, drift::SVector{3, Float64}, n_sim::Int,
        ::Val{A}) where {A}
    set_parcel_drift!(du, drift, n_sim, A)
end

"""
    ParcelProcess(w, h2o_idx, m_air, thermo, T0, p0, qv0) <: PhysicsProcess

Block-level parcel drift (spec §2.1): when present in the process tuple,
`make_ode_func` appends the 3-slot tail `[T, p, qv]` to the state vector.
Per-particle drift contribution is zero — the parcel block is written after
the particle loop using the accumulated Σ dm_w at `h2o_idx`.

Shares the `RefValue{ParcelState}` with the `ParcelCoupled` environment
source, which is synced from `u` at the TOP of each RHS evaluation (before
the particle loop — the flux layer reads it during the loop).
"""
struct ParcelProcess{P} <: PhysicsProcess
    parcel::Base.RefValue{ParcelState}
    w::Float64
    h2o_idx::Int
    m_air::Float64
    thermo::P
end

function ParcelProcess(w::Float64, h2o_idx::Int, m_air::Float64,
        thermo::ThermodynamicsParams, T0::Float64, p0::Float64, qv0::Float64)
    return ParcelProcess(Ref(ParcelState(T0, p0, qv0)), w, h2o_idx, m_air, thermo)
end

# accepted by solve_split's drift filter; per-particle drift contribution is
# zero (the parcel block is assembled by make_ode_func after the particle loop)
provides_drift(::ParcelProcess) = true
apply_drift(::ParcelProcess, μ::SVector{A, Float64}, sys, t) where {A} = zero(μ)
