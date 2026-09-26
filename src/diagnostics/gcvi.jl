# src/diagnostics/gcvi.jl

"""
    GCVIResponse(D50, width, E_max)

Virtual GCVI (ground-based counterflow virtual impactor) transmission model.

Ideal-classifier semantics: applied to the in-cloud endpoint wet diameter;
the "residual" of a CR-classified particle is its dry state (already
tracked) — drying is an instrument operation, not simulated.

Transmission curve (logistic):

    E(D_wet) = E_max / (1 + exp(-(D_wet - D50) / width))

# Fields
- `D50::Float64` — transition (cut) wet diameter [m]; default 7 μm
- `width::Float64` — logistic transition width [m]; default 0.7 μm
- `E_max::Float64` — maximum transmission efficiency; default 0.6
  (Shingler et al., 2012)
"""
Base.@kwdef struct GCVIResponse
    D50::Float64 = 7.0e-6
    width::Float64 = 7.0e-7
    E_max::Float64 = 0.6
end

"""
    transmission(resp::GCVIResponse, D_wet) -> Float64

Transmission efficiency E(D_wet) of the virtual GCVI.
"""
function transmission(resp::GCVIResponse, D_wet::Real)
    return resp.E_max / (1.0 + exp(-(Float64(D_wet) - resp.D50) / resp.width))
end

"""
    classify_cr_ci(u, sys, ::Val{A}, resp, densities; rng) -> Vector{Bool}

Probabilistic CR/CI classification of every active particle: particle `i`
is cloud residual (CR) if `rand(rng) <= E(D_wet,i)`, else interstitial (CI).

Wet diameters are the sphere-equivalent diameters over all species
(including water), i.e. `particle_diameters(u, sys, densities)`.
"""
function classify_cr_ci(
        u::Vector{Float64}, sys::ParticleSystem{A}, ::Val{A},
        resp::GCVIResponse, densities::SVector{A, Float64};
        rng::AbstractRNG = Random.default_rng()) where {A}
    diams = particle_diameters(u, sys, densities)
    return [rand(rng) <= transmission(resp, D) for D in diams]
end
