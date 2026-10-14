# src/utils/env_source.jl

"""
    AbstractEnvSource

Callable environment source for the `gas_phase` hook of `ParticleSystem`:
`g(t) -> SVector{3,Float64}(T, p_v, p_air)` (3-slot form, split-Fuchs: the
third slot feeds the air density of the heat-side transition-regime
correction in the H2O flux).

Invariant: `p_v` is always derived from the supersaturation via
`saturation_vapor_pressure(T) * (1 + S)` — the Clausius–Clapeyron relation
has a single point of definition in the library.
"""
abstract type AbstractEnvSource end

"""
    PrescribedProfile(times, T, S, p_air = 101325.0) <: AbstractEnvSource

Open-loop environment: linear interpolation of T(t) [K] and supersaturation
S(t) (dimensionless, 0.004 = 0.4 %) between knots, flat extrapolation outside
`[times[1], times[end]]`. `times` must be strictly increasing. `p_air` [Pa]
is the (constant) total air pressure of the prescribed environment.
"""
struct PrescribedProfile <: AbstractEnvSource
    times::Vector{Float64}
    T::Vector{Float64}
    S::Vector{Float64}
    p_air::Float64
    function PrescribedProfile(times, T, S, p_air::Float64 = 101325.0)
        (length(times) == length(T) == length(S)) ||
            throw(ArgumentError("times, T, S must have equal length"))
        length(times) >= 1 || throw(ArgumentError("at least one knot required"))
        for i in 2:length(times)
            times[i] > times[i - 1] ||
                throw(ArgumentError("times must be strictly increasing"))
        end
        p_air > 0.0 || throw(ArgumentError("p_air must be positive, got $p_air"))
        return new(times, T, S, p_air)
    end
end

function _interp_flat(t::Float64, times::Vector{Float64}, vals::Vector{Float64})
    t <= times[1] && return vals[1]
    t >= times[end] && return vals[end]
    i = searchsortedfirst(times, t)   # first index with times[i] >= t
    w = (t - times[i - 1]) / (times[i] - times[i - 1])
    return vals[i - 1] * (1.0 - w) + vals[i] * w
end

function (src::PrescribedProfile)(t::Real)
    T = _interp_flat(Float64(t), src.times, src.T)
    S = _interp_flat(Float64(t), src.times, src.S)
    return SVector{3, Float64}(T, saturation_vapor_pressure(T) * (1.0 + S), src.p_air)
end

"""
    ParcelCoupled(ref) <: AbstractEnvSource

Closed-loop environment source (blueprint contract ①, M3): reads the LIVE
parcel state shared with a `ParcelProcess` via `RefValue{ParcelState}` and
returns `SVector(T, p_sat(T)·(1+S), p)` with S diagnosed from (T, p, qv).
Ignores `t` — the state is synced from `u` at the top of each RHS
evaluation, so this source always sees the current solver stage.
"""
struct ParcelCoupled <: AbstractEnvSource
    parcel::Base.RefValue{ParcelState}
end

function (src::ParcelCoupled)(t::Real)
    pr = src.parcel[]
    S = parcel_supersaturation(pr)
    return SVector{3, Float64}(pr.T, saturation_vapor_pressure(pr.T) * (1.0 + S), pr.p)
end
