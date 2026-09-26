# src/utils/env_source.jl

"""
    AbstractEnvSource

Callable environment source for the `gas_phase` hook of `ParticleSystem`:
`g(t) -> SVector{2,Float64}(T, p_v)`.

Invariant: `p_v` is always derived from the supersaturation via
`saturation_vapor_pressure(T) * (1 + S)` — the Clausius–Clapeyron relation
has a single point of definition in the library.
"""
abstract type AbstractEnvSource end

"""
    PrescribedProfile(times, T, S) <: AbstractEnvSource

Open-loop environment: linear interpolation of T(t) [K] and supersaturation
S(t) (dimensionless, 0.004 = 0.4 %) between knots, flat extrapolation outside
`[times[1], times[end]]`. `times` must be strictly increasing.
"""
struct PrescribedProfile <: AbstractEnvSource
    times::Vector{Float64}
    T::Vector{Float64}
    S::Vector{Float64}
    function PrescribedProfile(times, T, S)
        (length(times) == length(T) == length(S)) ||
            throw(ArgumentError("times, T, S must have equal length"))
        length(times) >= 1 || throw(ArgumentError("at least one knot required"))
        for i in 2:length(times)
            times[i] > times[i - 1] ||
                throw(ArgumentError("times must be strictly increasing"))
        end
        return new(times, T, S)
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
    return SVector{2, Float64}(T, saturation_vapor_pressure(T) * (1.0 + S))
end
