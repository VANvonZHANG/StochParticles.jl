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

"""
    virtual_smps(cr_flags, dry_diameters, bin_edges, volume) -> NamedTuple(cr, ci)

Virtual-SMPS output: dN/dlog₁₀D spectra of the CR and CI subpopulations on
`bin_edges`, computed from **dry** diameters (residual convention: the
instrument downstream of the GCVI sees dried residuals).
"""
function virtual_smps(
        cr_flags::AbstractVector{Bool}, dry_diameters::AbstractVector{<:Real},
        bin_edges, volume::Real)
    length(cr_flags) == length(dry_diameters) ||
        throw(DimensionMismatch(
                  "cr_flags has $(length(cr_flags)) entries, dry_diameters has $(length(dry_diameters))"))
    cr_d = Float64[dry_diameters[i] for i in eachindex(cr_flags) if cr_flags[i]]
    ci_d = Float64[dry_diameters[i] for i in eachindex(cr_flags) if !cr_flags[i]]
    return (cr = dNdlogD_from_diameters(cr_d, bin_edges, volume),
            ci = dNdlogD_from_diameters(ci_d, bin_edges, volume))
end

"""
    virtual_acsm(u, sys, ::Val{A}, cr_flags; species = 1:(A - 1)) -> NamedTuple(cr, ci)

Virtual-ACSM output: dry mass fractions of the selected `species` within the
CR and CI subpopulations. Water must be excluded from `species` (dry
convention). Returns NaN fractions for an empty group.
"""
function virtual_acsm(
        u::Vector{Float64}, sys::ParticleSystem{A}, ::Val{A},
        cr_flags::AbstractVector{Bool};
        species::AbstractVector{Int} = collect(1:(A - 1))) where {A}
    n = sys.n_active
    length(cr_flags) == n ||
        throw(DimensionMismatch("cr_flags has $(length(cr_flags)) entries, n_active is $n"))
    K = length(species)
    m_cr = zeros(Float64, K)
    m_ci = zeros(Float64, K)
    for i in 1:n
        μ = get_particle(u, i, Val(A))
        target = cr_flags[i] ? m_cr : m_ci
        for (j, k) in enumerate(species)
            target[j] += μ[k]
        end
    end
    _fractions(m) = sum(m) > 0 ? m ./ sum(m) : fill(NaN, K)
    return (cr = _fractions(m_cr), ci = _fractions(m_ci))
end
