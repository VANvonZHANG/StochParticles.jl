# src/utils/preset_species.jl

"""
    Species

Preset aerosol species with physical-chemical properties.

All quantities use SI units: density [kg/m³], molar_mass [kg/mol].
κ (kappa) is the hygroscopicity parameter from κ-Köhler theory, dimensionless.
"""
@kwdef struct Species
    name::Symbol
    density::Float64      # [kg/m³]
    kappa::Float64        # hygroscopicity, dimensionless
    molar_mass::Float64   # [kg/mol]
end

# ---- Inorganic salts ----

"""
    AS :: Species

Ammonium sulfate, (NH₄)₂SO₄.
κ = 0.61 from Petters & Kreidenweis (2007).
"""
const AS = Species(:AS, 1770.0, 0.61, 0.13214)

"""
    AN :: Species

Ammonium nitrate, NH₄NO₃.
κ = 0.67 from Petters & Kreidenweis (2007).
"""
const AN = Species(:AN, 1720.0, 0.67, 0.08004)

# ---- Carbonaceous ----

"""
    BC :: Species

Black carbon (elemental carbon). Hydrophobic, κ = 0.
Molar mass uses atomic carbon (12.01 g/mol) as a reference.
"""
const BC = Species(:BC, 1800.0, 0.0, 0.01201)

"""
    OA :: Species

Organic aerosol (bulk surrogate). κ = 0.1 represents moderately oxidized OA.
Molar mass ≈ 250 g/mol is a representative value for SOA oligomers.
"""
const OA = Species(:OA, 1400.0, 0.1, 0.250)

# ---- Water ----

"""
    H2O :: Species

Liquid water. κ = 0 (not used in κ-mixing; water is handled specially).
"""
const H2O = Species(:H2O, 1000.0, 0.0, 0.018015)

# ---- Combiner ----

"""
    species_vectors(species::Species...)

Build simulation parameter vectors from a list of preset (or custom) species.

Returns a `NamedTuple`:
- `densities::SVector{A,Float64}`      — per-species densities
- `kappas::SVector{A,Float64}`         — per-species hygroscopicity
- `molar_masses::SVector{A,Float64}`   — per-species molar masses
- `names::SVector{A,String}`           — species names for diagnostics
- `h2o_idx::Int`                       — index of H₂O in the list

# Example
```julia
params = species_vectors(AS, BC, H2O)
thermo = ThermodynamicsParams(params.kappas, 0.072, 1000.0, params.molar_masses[params.h2o_idx], 2.5e6, 461.5, 2.5e-5, 2.4e-2)
cond = H2OCondensationProcess(thermo, params.densities; h2o_idx = params.h2o_idx)
```
"""
function species_vectors(species::Species...)
    A = length(species)
    A == 0 && throw(ArgumentError("at least one species required"))

    densities     = SVector{A}(s.density      for s in species)
    kappas        = SVector{A}(s.kappa        for s in species)
    molar_masses  = SVector{A}(s.molar_mass   for s in species)
    names         = SVector{A}(String(s.name) for s in species)

    h2o_idx = findfirst(s -> s.name === :H2O, species)
    h2o_idx === nothing && throw(ArgumentError(
        "H₂O must be included in the species list for H₂O condensation simulations"))

    return (; densities, kappas, molar_masses, names, h2o_idx)
end
