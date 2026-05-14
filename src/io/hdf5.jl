using HDF5

const SCHEMA_VERSION = "1.0.0"

"""
    h5_create_chunked(file, name, T, dims; chunk_size=64)

Create an extendable chunked dataset in the HDF5 file.

- For 1D datasets (e.g., `(1,)`): creates with shape `(0,)` and max `(UNLIMITED,)`.
- For 2D datasets (e.g., `(1, n_species)`): creates with shape `(0, n_cols)` and max `(UNLIMITED, n_cols)`.

Note: Due to HDF5/Julia dimension ordering, `dims` is interpreted in Julia column-major order.
"""
function h5_create_chunked(file, name, T, dims; chunk_size=64)
    if length(dims) == 1
        # 1D dataset: initial shape (0,), max (UNLIMITED,)
        space = HDF5.dataspace((0,), max_dims=(signed(HDF5.API.H5S_UNLIMITED),))
        dcpl = HDF5.API.h5p_create(HDF5.API.H5P_DATASET_CREATE)
        HDF5.API.h5p_set_chunk(dcpl, 1, Csize_t[chunk_size])
    elseif length(dims) == 2
        # 2D dataset: dims is (n_rows, n_cols) in Julia
        # HDF5.dataspace takes dims in Julia order and stores in file order.
        # size() reads file order and returns Julia order (reversed).
        # To get Julia initial size (0, n_cols) with first dim unlimited:
        #   Use dataspace((0, n_cols), max_dims=(UNLIMITED, n_cols))
        #   File raw: current=[n_cols, 0], max=[n_cols, UNLIMITED]
        n_rows, n_cols = dims
        space = HDF5.dataspace((0, n_cols), max_dims=(signed(HDF5.API.H5S_UNLIMITED), n_cols))
        dcpl = HDF5.API.h5p_create(HDF5.API.H5P_DATASET_CREATE)
        # Chunk in file order: [n_cols, chunk_size]
        # Julia chunk = reverse([n_cols, chunk_size]) = (chunk_size, n_cols)
        HDF5.API.h5p_set_chunk(dcpl, 2, Csize_t[n_cols, chunk_size])
    else
        error("Only 1D and 2D datasets are supported")
    end

    dtype = HDF5.datatype(T)
    dset_id = HDF5.API.h5d_create(file, name, dtype, space,
                                  HDF5.API.H5P_DEFAULT, dcpl, HDF5.API.H5P_DEFAULT)
    HDF5.API.h5o_close(dset_id)
    HDF5.API.h5p_close(dcpl)
    HDF5.close(dtype)
    return nothing
end

"""
    h5_append_row!(file, name, row)

Append a row (Vector) to a 2D chunked dataset, or a scalar to a 1D chunked dataset,
along the first dimension.
"""
function h5_append_row!(file, name, row::AbstractVector)
    if !haskey(file, name)
        throw(ArgumentError("Dataset '$(name)' not found in HDF5 file"))
    end
    dset = file[name]
    curr_size = size(dset)

    if length(curr_size) == 2
        if length(row) != curr_size[2]
            throw(DimensionMismatch("Row length $(length(row)) does not match dataset width $(curr_size[2])"))
        end
        # 2D dataset: extend dim 1 by 1
        # curr_size is in Julia order: (n_rows, n_cols)
        # To extend to (n_rows+1, n_cols) in Julia:
        #   File raw should be [n_cols, n_rows+1]
        new_n_rows = curr_size[1] + 1
        HDF5.API.h5d_set_extent(dset, Csize_t[curr_size[2], new_n_rows])
        # Write the row at the new last index
        dset[new_n_rows:new_n_rows, :] = reshape(row, 1, length(row))
    else
        error("Only 2D datasets are supported with vector input")
    end
    return nothing
end

function h5_append_row!(file, name, value::T) where {T}
    if !haskey(file, name)
        throw(ArgumentError("Dataset '$(name)' not found in HDF5 file"))
    end
    dset = file[name]
    curr_size = size(dset)

    if length(curr_size) == 1
        # 1D dataset: extend by 1 and write at the end
        new_size = (curr_size[1] + 1,)
        HDF5.API.h5d_set_extent(dset, Csize_t[new_size[1]])
        dset[end] = value
    else
        error("Only 1D datasets are supported with scalar input")
    end
    return nothing
end

"""
    h5_write_attrs(parent, name; kwargs...)

Create a group named `name` under `parent` and write keyword arguments as attributes.
"""
function h5_write_attrs(parent, name; kwargs...)
    if !haskey(parent, name)
        create_group(parent, name)
    end
    g = parent[name]
    for (k, v) in kwargs
        attrs(g)[string(k)] = v
    end
    return nothing
end

"""
    h5_read_attrs(parent, name)

Read all attributes from a group named `name` under `parent` into a Dict.
"""
function h5_read_attrs(parent, name)
    if !haskey(parent, name)
        throw(ArgumentError("Group '$(name)' not found in HDF5 file"))
    end
    g = parent[name]
    a = attrs(g)
    result = Dict{String,Any}()
    for k in keys(a)
        result[k] = a[k]
    end
    return result
end

"""
    validate_schema_version(file)

Read `/meta/schema_version` from the HDF5 file and validate against `SCHEMA_VERSION`.

- Major version mismatch: throws `ErrorException`.
- Minor version mismatch: emits `@warn` but continues.
- Returns the file version string.
"""
function validate_schema_version(file)
    if !haskey(file, "meta")
        error("Missing /meta group in HDF5 file")
    end
    g = file["meta"]
    if !haskey(attrs(g), "schema_version")
        error("Missing schema_version attribute in /meta group")
    end
    file_version_str = attrs(g)["schema_version"]
    file_version = VersionNumber(file_version_str)
    current_version = VersionNumber(SCHEMA_VERSION)

    if file_version.major != current_version.major
        error("Schema version major mismatch: file has $(file_version_str), expected $(SCHEMA_VERSION)")
    end

    if file_version.minor != current_version.minor
        @warn "Schema version minor mismatch: file has $(file_version_str), expected $(SCHEMA_VERSION)"
    end

    return file_version_str
end
