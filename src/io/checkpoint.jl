using HDF5
using Random
using Dates

const CHECKPOINT_VERSION = "StochParticles.jl v0.1.0"

"""
    save_checkpoint(
        path::String,
        u::Vector{Float64},
        sys::ParticleSystem,
        t::Float64;
        rng::AbstractRNG = Random.default_rng(),
        process_info = nothing,
        overwrite::Bool = false
    )

Save a restart checkpoint to `path`. Appends `.h5` suffix if missing.
"""
function save_checkpoint(
        path::String,
        u::Vector{Float64},
        sys::ParticleSystem,
        t::Float64;
        rng::AbstractRNG = Random.default_rng(),
        process_info = nothing,
        overwrite::Bool = false
)
    if !endswith(path, ".h5")
        path = path * ".h5"
    end

    if isfile(path) && !overwrite
        throw(ErrorException("Checkpoint file already exists: $(path)"))
    end

    mode = overwrite ? "w" : "cw"

    HDF5.h5open(path, mode) do file
        # /checkpoint group
        if !haskey(file, "checkpoint")
            create_group(file, "checkpoint")
        end
        ckpt = file["checkpoint"]
        ckpt["u"] = u
        ckpt["t"] = t
        ckpt["n_active"] = sys.n_active
        ckpt["volume"] = sys.volume
        ckpt["n_sim"] = sys.n_sim
        ckpt["mass_total_cache"] = sys._mass_total_cache
        ckpt["cached_majorant"] = sys._cached_majorant

        # /rng group
        if !haskey(file, "rng")
            create_group(file, "rng")
        end
        rng_group = file["rng"]
        seed = 0
        state_vec = UInt64[]
        if rng isa Xoshiro
            st = (rng.s0, rng.s1, rng.s2, rng.s3)
            state_vec = Vector{UInt64}([st...])
        else
            @warn "RNG state serialization only supported for Xoshiro; saving empty state"
        end
        rng_group["seed"] = seed
        rng_group["state"] = state_vec

        # /meta group
        if !haskey(file, "meta")
            create_group(file, "meta")
        end
        meta = file["meta"]
        attrs(meta)["version"] = CHECKPOINT_VERSION
        attrs(meta)["schema_version"] = SCHEMA_VERSION
        attrs(meta)["created_at"] = string(now())
        attrs(meta)["julia_version"] = string(VERSION)
        git_commit = try
            readchomp(`git rev-parse HEAD`)
        catch
            "unknown"
        end
        attrs(meta)["git_commit"] = git_commit
        attrs(meta)["gas_phase_ref"] = "user_provided"

        # /process_info group (optional)
        if process_info !== nothing
            if !haskey(file, "process_info")
                create_group(file, "process_info")
            end
            # Store as attributes for simplicity
            pi_group = file["process_info"]
            for (k, v) in pairs(process_info)
                attrs(pi_group)[string(k)] = v
            end
        end
    end

    return nothing
end

"""
    load_checkpoint(path::String) -> (u, sys_data, t, rng_state)

Load a checkpoint file.

# Returns
- `u::Vector{Float64}`: State vector.
- `sys_data::NamedTuple`: `(n_active, volume, n_sim, mass_total_cache, cached_majorant)`.
- `t::Float64`: Simulation time.
- `rng_state::Dict`: Keys `:seed` (Int) and `:state` (Vector{UInt64}).

The caller must reconstruct `ParticleSystem` with the original `gas_phase` function.
"""
function load_checkpoint(path::String)
    if !isfile(path)
        throw(SystemError("Checkpoint file not found: $(path)", 0))
    end

    HDF5.h5open(path, "r") do file
        validate_schema_version(file)

        if !haskey(file, "checkpoint")
            throw(ErrorException("Checkpoint file missing /checkpoint group"))
        end
        if !haskey(file, "rng")
            throw(ErrorException("Checkpoint file missing /rng group"))
        end

        ckpt = file["checkpoint"]
        u = read(ckpt["u"])
        t = read(ckpt["t"])
        n_active = read(ckpt["n_active"])
        volume = read(ckpt["volume"])
        n_sim = read(ckpt["n_sim"])
        mass_total_cache = read(ckpt["mass_total_cache"])
        cached_majorant = read(ckpt["cached_majorant"])

        sys_data = (
            n_active = n_active,
            volume = volume,
            n_sim = n_sim,
            mass_total_cache = mass_total_cache,
            cached_majorant = cached_majorant
        )

        rng_group = file["rng"]
        seed = read(rng_group["seed"])
        state = read(rng_group["state"])
        rng_state = Dict(:seed => seed, :state => Vector{UInt64}(state))

        return (u, sys_data, t, rng_state)
    end
end

"""
    restore_rng(rng_state::Dict) -> Xoshiro

Restore an `Xoshiro` RNG from a state dictionary returned by `load_checkpoint`.
"""
function restore_rng(rng_state::Dict)
    state = rng_state[:state]
    if length(state) != 4
        throw(ArgumentError("Invalid RNG state: expected 4 UInt64 values, got $(length(state))"))
    end
    rng = Xoshiro(0)
    rng.s0 = state[1]
    rng.s1 = state[2]
    rng.s2 = state[3]
    rng.s3 = state[4]
    return rng
end

"""
    list_checkpoints(prefix::String) -> Vector{String}

Find and sort all checkpoint files matching `prefix_*.h5`.
"""
function list_checkpoints(prefix::String)
    dir = dirname(prefix)
    if isempty(dir)
        dir = "."
    end
    base = basename(prefix)

    escaped_base = replace(base, r"[.*+?^${}()|[\]\\]" => s"\\\0")
    pattern = Regex("^" * escaped_base * "_\\d+\\.h5\$")
    files = String[]
    for f in readdir(dir)
        if occursin(pattern, f)
            push!(files, joinpath(dir, f))
        end
    end
    sort!(files, by = f -> begin
        m = match(r"_(\d+)\.h5$", f)
        m !== nothing ? parse(Int, m.captures[1]) : 0
    end)
    return files
end
