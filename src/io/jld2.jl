using JLD2
using Random
using Dates

"""
    save_checkpoint_jld2(path, u, sys, t; rng=default_rng(), overwrite=false)

Save a simulation checkpoint to a JLD2 file.

# Arguments
- `path::String`: Output file path (`.jld2` suffix appended if missing).
- `u::Vector{Float64}`: Simulation state vector.
- `sys::ParticleSystem`: Particle system containing metadata.
- `t::Float64`: Current simulation time.

# Keywords
- `rng::AbstractRNG`: Random number generator. Only `Xoshiro` RNG state is fully
  serialized (seed is set to 0; the full 4-word state is preserved so the sequence
  can be restored exactly).
- `overwrite::Bool`: If `true`, overwrite an existing file.

# Returns
- `nothing`

# Throws
- `ErrorException` if the file already exists and `overwrite` is `false`.
"""
function save_checkpoint_jld2(
        path::String,
        u::Vector{Float64},
        sys::ParticleSystem,
        t::Float64;
        process_info = nothing,
        rng::AbstractRNG = Random.default_rng(),
        overwrite::Bool = false
)
    if !endswith(path, ".jld2")
        path = path * ".jld2"
    end

    if isfile(path) && !overwrite
        throw(ErrorException("Checkpoint file already exists: $(path)"))
    end

    seed = 0
    state_vec = UInt64[]
    if rng isa Xoshiro
        st = (rng.s0, rng.s1, rng.s2, rng.s3)
        state_vec = UInt64[st...]
    else
        @warn "RNG state serialization only supported for Xoshiro; saving empty state"
    end

    jldopen(path, "w") do file
        file["u"] = u
        file["t"] = t
        file["n_active"] = sys.n_active
        file["volume"] = sys.volume
        file["n_sim"] = sys.n_sim
        file["mass_total_cache"] = sys._mass_total_cache
        file["cached_majorant"] = sys._cached_majorant
        file["rng_seed"] = seed
        file["rng_state"] = state_vec
        file["version"] = CHECKPOINT_VERSION
        file["schema_version"] = SCHEMA_VERSION
        file["created_at"] = string(now())
    end

    return nothing
end

"""
    load_checkpoint_jld2(path)

Load a simulation checkpoint from a JLD2 file.

# Arguments
- `path::String`: Path to the `.jld2` checkpoint file.

# Returns
A tuple `(u, sys_data, t, rng_state)` where:
- `u::Vector{Float64}`: Simulation state vector.
- `sys_data`: Named tuple with fields `n_active`, `volume`, `n_sim`,
  `mass_total_cache`, and `cached_majorant`.
- `t::Float64`: Simulation time.
- `rng_state::Dict`: Dictionary with `:seed` (always 0 for Xoshiro) and `:state`
  (`Vector{UInt64}`). The full Xoshiro RNG state is preserved so the sequence can
  be restored exactly with `restore_rng`.

# Throws
- `SystemError` if the file does not exist.
- `ErrorException` if the schema version major number does not match `SCHEMA_VERSION`.
"""
function load_checkpoint_jld2(path::String)
    if !isfile(path)
        throw(SystemError("Checkpoint file not found: $(path)", 0))
    end

    jldopen(path, "r") do file
        for key in ["schema_version", "u", "t", "n_active", "volume", "n_sim",
            "mass_total_cache", "cached_majorant", "rng_seed", "rng_state"]
            if !haskey(file, key)
                throw(ErrorException("Checkpoint file missing required key: $(key)"))
            end
        end

        file_version = file["schema_version"]
        current = VersionNumber(SCHEMA_VERSION)
        file_v = VersionNumber(file_version)
        if file_v.major != current.major
            throw(ErrorException("Schema version major mismatch: file has $file_version, expected $SCHEMA_VERSION"))
        end
        if file_v.minor != current.minor
            @warn "Schema version minor mismatch: file has $file_version, expected $SCHEMA_VERSION"
        end

        u = file["u"]
        t = file["t"]
        n_active = file["n_active"]
        volume = file["volume"]
        n_sim = file["n_sim"]
        mass_total_cache = file["mass_total_cache"]
        cached_majorant = file["cached_majorant"]

        sys_data = (
            n_active = n_active,
            volume = volume,
            n_sim = n_sim,
            mass_total_cache = mass_total_cache,
            cached_majorant = cached_majorant
        )

        seed = file["rng_seed"]
        state = file["rng_state"]
        rng_state = Dict(:seed => seed, :state => Vector{UInt64}(state))

        return (u, sys_data, t, rng_state)
    end
end
