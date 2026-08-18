using HDF5
using Dates
using Statistics
using StaticArrays
using StochParticles
using DiffEqCallbacks
using OrdinaryDiffEq

const EXAMPLE_SCHEMA_VERSION = "examples-v1"

function example_data_dir()
    path = joinpath(@__DIR__, "data")
    mkpath(path)
    return path
end

function example_fig_dir()
    path = joinpath(@__DIR__, "fig")
    mkpath(path)
    return path
end

function example_replicates(default::Int = 20)
    default >= 1 || throw(ArgumentError("default replicate count must be at least 1"))
    value = get(ENV, "STOCHPARTICLES_EXAMPLE_REPLICATES", string(default))
    n = tryparse(Int, value)
    n === nothing &&
        throw(ArgumentError("STOCHPARTICLES_EXAMPLE_REPLICATES must be an integer, got '$value'"))
    n >= 1 ||
        throw(ArgumentError("STOCHPARTICLES_EXAMPLE_REPLICATES must be at least 1, got $n"))
    return n
end

function _write_attrs!(group, attrs_dict)
    for (key, value) in attrs_dict
        attrs(group)[string(key)] = value
    end
    return nothing
end

function _require_group(parent, key, path)
    obj = parent[key]
    if !(obj isa HDF5.Group)
        throw(ErrorException("HDF5 path '$path' exists but is not a group"))
    end
    return obj
end

function recreate_h5(path; scene_name, n_replicates, notes)
    n_replicates >= 1 ||
        throw(ArgumentError("n_replicates must be at least 1, got $n_replicates"))

    parent = dirname(path)
    if !isempty(parent) && parent != "."
        mkpath(parent)
    end
    if isfile(path)
        rm(path)
    end

    h5open(path, "w") do file
        meta = create_group(file, "meta")
        attrs(meta)["schema_version"] = EXAMPLE_SCHEMA_VERSION
        attrs(meta)["scene_name"] = scene_name
        attrs(meta)["n_replicates"] = n_replicates
        attrs(meta)["notes"] = notes
        attrs(meta)["created_at"] = string(now())
        create_group(file, "cases")
    end
    return path
end

function ensure_case_group(file, case_name; attrs_dict = Dict{String, Any}())
    case_key = string(case_name)
    occursin("/", case_key) &&
        throw(ArgumentError("case_name must not contain '/', got '$case_key'"))

    if !haskey(file, "cases")
        throw(ErrorException("HDF5 path '/cases' does not exist"))
    end
    cases = _require_group(file, "cases", "/cases")
    case_path = "/cases/$case_key"
    case_group = haskey(cases, case_key) ? _require_group(cases, case_key, case_path) :
                 create_group(cases, case_key)
    if !haskey(case_group, "replicates")
        create_group(case_group, "replicates")
    elseif !(case_group["replicates"] isa HDF5.Group)
        throw(ErrorException("HDF5 path '$case_path/replicates' exists but is not a group"))
    end
    _write_attrs!(case_group, attrs_dict)
    return case_group
end

function _assert_new_dataset(parent, name)
    key = string(name)
    if haskey(parent, key)
        throw(ArgumentError("HDF5 dataset or group '$key' already exists"))
    end
    return key
end

function write_vector(parent, name, values)
    key = _assert_new_dataset(parent, name)
    parent[key] = collect(values)
    return parent[key]
end

function write_matrix(parent, name, values::AbstractMatrix)
    key = _assert_new_dataset(parent, name)
    parent[key] = values
    return parent[key]
end

function padded_columns(vectors, width)
    width >= 0 || throw(ArgumentError("width must be non-negative, got $width"))
    matrix = fill(NaN, length(vectors), width)
    for (row_idx, values) in enumerate(vectors)
        row = Float64.(collect(values))
        length(row) <= width ||
            throw(DimensionMismatch("vector $row_idx has length $(length(row)), exceeding width $width"))
        for col_idx in eachindex(row)
            matrix[row_idx, col_idx] = row[col_idx]
        end
    end
    return matrix
end

function diameter_summary(diameters)
    values = Float64.(collect(diameters))
    if isempty(values)
        return (mean = 0.0, median = 0.0, p90 = 0.0)
    end
    return (
        mean = mean(values),
        median = median(values),
        p90 = quantile(values, 0.90)
    )
end

function dNdlogD_from_diameters(diameters, bin_edges, volume)
    volume > 0.0 || throw(ArgumentError("volume must be positive, got $volume"))
    edges = Float64.(collect(bin_edges))
    counts = bin_size_distribution(Float64.(collect(diameters)), edges)
    dlogD = diff(log10.(edges))
    return Float64.(counts) ./ dlogD ./ volume
end

function _density_vector(::Val{A}, densities::SVector{A, Float64}) where {A}
    return densities
end

function _density_vector(::Val{A}, densities::AbstractVector) where {A}
    length(densities) == A ||
        throw(ArgumentError("densities length $(length(densities)) must match species count $A"))
    return SVector{A, Float64}(densities)
end

function _density_vector(::Val{A}, density::Real) where {A}
    A == 1 ||
        throw(ArgumentError("scalar density is only valid for single-species systems"))
    return SVector{1, Float64}(Float64(density))
end

function base_diagnostic_record(t, u, sys, ::Val{A}, densities, bin_edges) where {A}
    A_val = Val(A)
    density_vec = _density_vector(A_val, densities)
    diameters = particle_diameters(u, sys, density_vec)
    summary = diameter_summary(diameters)

    species_mass = Vector{Float64}(undef, A)
    for species_idx in 1:A
        species_mass[species_idx] = species_mass_concentration(u, species_idx, A_val, sys)
    end

    return (
        time = Float64(t),
        volume = Float64(sys.volume),
        active_particles = sys.n_active,
        number_concentration = number_concentration(sys),
        total_mass_concentration = mass_concentration(u, A_val, sys),
        species_mass_concentration = species_mass,
        mean_diameter = summary.mean,
        median_diameter = summary.median,
        p90_diameter = summary.p90,
        diameter_samples = diameters,
        size_distribution_raw = dNdlogD_from_diameters(diameters, bin_edges, sys.volume)
    )
end

function merge_record(record, extras::NamedTuple)
    return merge(record, extras)
end

function solve_with_records(prob, solver; saveat, record_func)
    saved = DiffEqCallbacks.SavedValues(Float64, Any)
    save_func = (u, t, integrator) -> record_func(t, u, integrator.p)
    saving = DiffEqCallbacks.SavingCallback(
        save_func, saved; saveat = saveat, save_start = true, save_end = false)
    sol = solve(prob, solver; callback = saving, saveat = saveat)
    return sol, collect(saved.saveval)
end

const _COMMON_RECORD_FIELDS = Set([
    :time,
    :volume,
    :active_particles,
    :number_concentration,
    :total_mass_concentration,
    :species_mass_concentration,
    :mean_diameter,
    :median_diameter,
    :p90_diameter,
    :diameter_samples,
    :size_distribution_raw
])

function _record_vector(records, field)
    values = Vector{Any}(undef, length(records))
    for i in eachindex(records)
        values[i] = getproperty(records[i], field)
    end
    return values
end

function _scalar_record_vector(::Type{T}, records, field) where {T}
    values = Vector{T}(undef, length(records))
    for i in eachindex(records)
        values[i] = getproperty(records[i], field)
    end
    return values
end

function _fixed_width_record_matrix(records, field, width)
    matrix = zeros(Float64, length(records), width)
    for row_idx in eachindex(records)
        row = Float64.(collect(getproperty(records[row_idx], field)))
        length(row) == width ||
            throw(DimensionMismatch("record $row_idx field $field has length $(length(row)), expected $width"))
        for col_idx in 1:width
            matrix[row_idx, col_idx] = row[col_idx]
        end
    end
    return matrix
end

function _validate_record_keys(records)
    isempty(records) && return ()

    reference_keys = keys(records[1])
    reference_set = Set(reference_keys)
    for record_idx in eachindex(records)
        record_set = Set(keys(records[record_idx]))
        missing = sort!(collect(setdiff(reference_set, record_set)); by = string)
        extra = sort!(collect(setdiff(record_set, reference_set)); by = string)
        if !isempty(missing) || !isempty(extra)
            throw(ArgumentError(
                "record $record_idx has inconsistent keys; " *
                "missing keys: $(collect(string.(missing))); " *
                "extra keys: $(collect(string.(extra)))"))
        end
    end
    return reference_keys
end

function _write_extra_record_field!(rep_group, records, field)
    values = _record_vector(records, field)
    if isempty(values)
        return nothing
    end

    first_value = values[1]
    if first_value isa Bool
        typed_values = Vector{Bool}(undef, length(values))
        for i in eachindex(values)
            typed_values[i] = values[i]
        end
        write_vector(rep_group, string(field), typed_values)
        return nothing
    end

    if first_value isa Number
        typed_values = Vector{Float64}(undef, length(values))
        for i in eachindex(values)
            typed_values[i] = values[i]
        end
        write_vector(rep_group, string(field), typed_values)
        return nothing
    end

    if first_value isa AbstractVector || first_value isa Tuple
        width = 0
        for value in values
            width = max(width, length(value))
        end
        write_matrix(rep_group, string(field), padded_columns(values, width))
        return nothing
    end

    throw(ArgumentError("unsupported extra record field '$field' with value type $(typeof(first_value))"))
end

function write_records_common!(
        rep_group,
        records,
        n_sim,
        bin_edges;
        dry_diameter_initial = Float64[],
        extra_attrs = Dict{String, Any}())
    n_sim >= 1 || throw(ArgumentError("n_sim must be at least 1, got $n_sim"))
    edges = Float64.(collect(bin_edges))
    n_bins = max(length(edges) - 1, 0)
    n_records = length(records)
    record_keys = _validate_record_keys(records)

    attrs(rep_group)["schema_version"] = EXAMPLE_SCHEMA_VERSION
    attrs(rep_group)["n_sim"] = n_sim
    attrs(rep_group)["n_records"] = n_records
    _write_attrs!(rep_group, extra_attrs)

    write_vector(rep_group, "bin_edges", edges)
    if !isempty(dry_diameter_initial)
        write_vector(rep_group, "dry_diameter_initial", dry_diameter_initial)
    end

    if n_records == 0
        write_vector(rep_group, "time", Float64[])
        write_vector(rep_group, "volume", Float64[])
        write_vector(rep_group, "active_particles", Int[])
        write_vector(rep_group, "number_concentration", Float64[])
        write_vector(rep_group, "total_mass_concentration", Float64[])
        write_matrix(rep_group, "species_mass_concentration", zeros(Float64, 0, 0))
        write_vector(rep_group, "mean_diameter", Float64[])
        write_vector(rep_group, "median_diameter", Float64[])
        write_vector(rep_group, "p90_diameter", Float64[])
        write_matrix(rep_group, "diameter_samples", zeros(Float64, 0, n_sim))
        write_matrix(rep_group, "size_distribution_raw", zeros(Float64, 0, n_bins))
        return rep_group
    end

    write_vector(rep_group, "time", _scalar_record_vector(Float64, records, :time))
    write_vector(rep_group, "volume", _scalar_record_vector(Float64, records, :volume))
    write_vector(rep_group, "active_particles", _scalar_record_vector(Int, records, :active_particles))
    write_vector(rep_group, "number_concentration",
        _scalar_record_vector(Float64, records, :number_concentration))
    write_vector(rep_group, "total_mass_concentration",
        _scalar_record_vector(Float64, records, :total_mass_concentration))

    n_species = length(getproperty(records[1], :species_mass_concentration))
    write_matrix(rep_group, "species_mass_concentration",
        _fixed_width_record_matrix(records, :species_mass_concentration, n_species))

    write_vector(rep_group, "mean_diameter",
        _scalar_record_vector(Float64, records, :mean_diameter))
    write_vector(rep_group, "median_diameter",
        _scalar_record_vector(Float64, records, :median_diameter))
    write_vector(rep_group, "p90_diameter",
        _scalar_record_vector(Float64, records, :p90_diameter))

    diameter_vectors = _record_vector(records, :diameter_samples)
    write_matrix(rep_group, "diameter_samples", padded_columns(diameter_vectors, n_sim))
    write_matrix(rep_group, "size_distribution_raw",
        _fixed_width_record_matrix(records, :size_distribution_raw, n_bins))

    for field in record_keys
        if !(field in _COMMON_RECORD_FIELDS)
            _write_extra_record_field!(rep_group, records, field)
        end
    end

    return rep_group
end

function create_replicate_group(case_group, replicate_idx)
    replicate_idx >= 1 ||
        throw(ArgumentError("replicate_idx must be at least 1, got $replicate_idx"))
    if !haskey(case_group, "replicates")
        create_group(case_group, "replicates")
    end
    replicate_name = "replicate_" * lpad(string(replicate_idx), 3, "0")
    replicates = case_group["replicates"]
    if haskey(replicates, replicate_name)
        throw(ArgumentError("replicate group '$replicate_name' already exists"))
    end
    return create_group(replicates, replicate_name)
end
