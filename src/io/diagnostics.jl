using HDF5
using Dates

"""
    init_diagnostics_file(
        path::String,
        n_species::Int,
        bin_edges::Vector{Float64};
        species_names = ["species_\$i" for i in 1:n_species],
        chunk_size::Int = 64
    )

Create a new diagnostics HDF5 file with empty chunked datasets.
"""
function init_diagnostics_file(
        path::String,
        n_species::Int,
        bin_edges::Vector{Float64};
        species_names = ["species_\$i" for i in 1:n_species],
        chunk_size::Int = 64
    )
    if isfile(path)
        throw(ErrorException("Diagnostics file already exists: $(path)"))
    end
    if length(bin_edges) < 2
        throw(ArgumentError("bin_edges must have at least 2 elements"))
    end
    if length(species_names) != n_species
        throw(ArgumentError("species_names length ($(length(species_names))) must match n_species ($n_species)"))
    end

    n_bins = length(bin_edges) - 1

    HDF5.h5open(path, "cw") do file
        h5_create_chunked(file, "time", Float64, (1,); chunk_size = chunk_size)
        h5_create_chunked(file, "number_concentration", Float64, (1,); chunk_size = chunk_size)
        h5_create_chunked(file, "mass_concentration", Float64, (1,); chunk_size = chunk_size)
        h5_create_chunked(file, "species_mass_concentration", Float64, (1, n_species); chunk_size = chunk_size)
        h5_create_chunked(file, "mean_diameter", Float64, (1,); chunk_size = chunk_size)
        h5_create_chunked(file, "volume", Float64, (1,); chunk_size = chunk_size)
        h5_create_chunked(file, "size_distribution", Float64, (1, n_bins); chunk_size = chunk_size)

        h5_write_attrs(file, "meta";
            species_names = species_names,
            bin_edges = bin_edges,
            created_at = string(now()),
            version = CHECKPOINT_VERSION,
            schema_version = SCHEMA_VERSION
        )
    end

    return nothing
end

"""
    save_diagnostics(
        path::String,
        t::Float64,
        u::Vector{Float64},
        sys::ParticleSystem,
        ::Val{A};
        bin_edges = nothing,
        rho::Float64 = 1000.0
    ) where A

Append one timestep of diagnostics to the file at `path`.

# Parameters
- `bin_edges`: diameter bin edges [m] for size distribution. Must match the bin_edges used in `init_diagnostics_file`.
- `rho`: particle density [kg/m³] for diameter computation. Default 1000.0.
"""
function save_diagnostics(
        path::String,
        t::Float64,
        u::Vector{Float64},
        sys::ParticleSystem,
        ::Val{A};
        bin_edges = nothing,
        rho::Float64 = 1000.0
    ) where {A}
    if !isfile(path)
        throw(SystemError("Diagnostics file not found: $(path)", 0))
    end

    HDF5.h5open(path, "r+") do file
        if !haskey(file, "time") || !haskey(file, "meta")
            throw(ErrorException("Invalid diagnostics file: missing required datasets"))
        end

        if bin_edges !== nothing
            stored_bin_edges = attrs(file["meta"])["bin_edges"]
            if length(bin_edges) != length(stored_bin_edges) || !all(bin_edges .≈ stored_bin_edges)
                throw(ArgumentError("bin_edges do not match the bin_edges stored in the diagnostics file"))
            end
        end

        if sys.n_active > 0
            diams = particle_diameters(u, sys, rho)
        end

        h5_append_row!(file, "time", t)
        h5_append_row!(file, "number_concentration", number_concentration(sys))

        M = total_mass(u, Val(A), sys.n_active)
        h5_append_row!(file, "mass_concentration", M / sys.volume)

        species_masses = Vector{Float64}(undef, A)
        for s in 1:A
            species_masses[s] = species_mass_concentration(u, s, Val(A), sys)
        end
        h5_append_row!(file, "species_mass_concentration", species_masses)

        if sys.n_active > 0
            mean_d = sum(diams) / sys.n_active
        else
            mean_d = 0.0
        end
        h5_append_row!(file, "mean_diameter", mean_d)

        h5_append_row!(file, "volume", sys.volume)

        if bin_edges !== nothing
            if sys.n_active > 0
                counts = bin_size_distribution(diams, bin_edges)
                size_dist = Float64.(counts) ./ sys.volume
            else
                n_bins = length(bin_edges) - 1
                size_dist = zeros(Float64, n_bins)
            end
        else
            n_bins = size(file["size_distribution"])[2]
            size_dist = zeros(Float64, n_bins)
        end
        h5_append_row!(file, "size_distribution", size_dist)
    end

    return nothing
end
