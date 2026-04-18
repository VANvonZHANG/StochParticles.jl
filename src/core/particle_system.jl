# src/core/particle_system.jl

"""
    ParticleSystem{A, F}

Mutable parameter struct (the `p` in SciML's `f(u, p, t)`).
Holds simulation metadata that changes during PDMP evolution.

# Unit Convention

All masses are in **kg**, volumes in **m³**, temperatures in **K**,
viscosities in **Pa·s**, and densities in **kg/m³**. Particle
compositions `μ` in the ODE state vector `u` use these SI units.

Fields:
- `n_active::Int` — current number of active particles
- `volume::Float64` — computational volume V_comp(t)
- `gas_phase::F` — external function g(t) returning gas-phase concentrations
- `n_sim::Int` — target particle count for CNMC
- `_mass_total_cache::Float64` — internal: cached total mass for CNMC volume rescale
- `_cached_majorant::Float64` — internal: cached K_max for coagulation accept/reject
"""
mutable struct ParticleSystem{A, F}
    n_active::Int
    volume::Float64
    gas_phase::F
    n_sim::Int
    _mass_total_cache::Float64
    _cached_majorant::Float64
end

"""
    ParticleSystem(::Val{A}, n_sim, volume, gas_phase_fn)

Construct a ParticleSystem with `n_sim` active particles.
"""
ParticleSystem(::Val{A}, n_sim::Int, volume::Float64, gas_phase_fn::F) where {A, F} =
    ParticleSystem{A, F}(n_sim, volume, gas_phase_fn, n_sim, 0.0, 0.0)

"""
    species_val(sys::ParticleSystem{A}) -> Val{A}

Return the compile-time species count as a Val type.
"""
species_val(::ParticleSystem{A}) where {A} = Val(A)

"""
    get_particle(u, i, ::Val{A}) -> SVector{A, Float64}

Extract particle i's composition from flat ODE state vector `u`.
Type-stable: SVector is constructed via ntuple, compiler optimizes to direct load.
"""
@inline function get_particle(u::Vector{Float64}, i::Int, ::Val{A}) where {A}
    offset = (i - 1) * A
    @inbounds SVector(ntuple(k -> u[offset + k], Val(A)))
end

"""
    set_particle!(u, i, ::Val{A}, μ::SVector{A, Float64})

Write particle i's composition into flat ODE state vector `u`.
"""
@inline function set_particle!(u::Vector{Float64}, i::Int, ::Val{A}, μ::SVector{A, Float64}) where {A}
    offset = (i - 1) * A
    @inbounds for k in 1:A
        u[offset + k] = μ[k]
    end
    nothing
end

"""
    make_u0(particles::Vector{SVector{A, Float64}}) -> Vector{Float64}

Create flat ODE state vector from particle composition vectors.
"""
function make_u0(particles::Vector{SVector{A, Float64}}) where {A}
    N = length(particles)
    u0 = Vector{Float64}(undef, N * A)
    for i in 1:N
        set_particle!(u0, i, Val(A), particles[i])
    end
    return u0
end

"""
    total_mass(u, ::Val{A}, n_active) -> Float64

Sum of all component masses across active particles.
"""
function total_mass(u::Vector{Float64}, ::Val{A}, n_active::Int) where {A}
    m = 0.0
    for i in 1:n_active
        μ = get_particle(u, i, Val(A))
        m += sum(μ)
    end
    return m
end
