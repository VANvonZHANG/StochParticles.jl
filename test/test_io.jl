using StochParticles
using Test
using HDF5

@testset "IO: HDF5 low-level helpers" begin
    mktempdir() do dir
        path = joinpath(dir, "test_helpers.h5")

        # Test h5_create_chunked (2D)
        HDF5.h5open(path, "w") do file
            StochParticles.h5_create_chunked(file, "data", Float64, (2, 3); chunk_size=4)
            @test haskey(file, "data")
            ds = file["data"]
            @test size(ds) == (0, 3)
            @test HDF5.get_chunk(ds)[1] == 4
        end

        # Test h5_append_row! with 2D data (append along dim 1)
        HDF5.h5open(path, "r+") do file
            StochParticles.h5_append_row!(file, "data", [1.0, 2.0, 3.0])
            StochParticles.h5_append_row!(file, "data", [4.0, 5.0, 6.0])
            ds = file["data"]
            @test size(ds) == (2, 3)
            data = read(ds)
            @test data[1, :] ≈ [1.0, 2.0, 3.0]
            @test data[2, :] ≈ [4.0, 5.0, 6.0]
        end

        # Test h5_write_attrs and h5_read_attrs
        HDF5.h5open(path, "r+") do file
            StochParticles.h5_write_attrs(file, "meta"; version="1.0.0", count=42)
            attrs = StochParticles.h5_read_attrs(file, "meta")
            @test attrs["version"] == "1.0.0"
            @test attrs["count"] == 42
        end

        # Test 1D dataset creation and scalar append
        path1d = joinpath(dir, "test_1d.h5")
        HDF5.h5open(path1d, "w") do file
            StochParticles.h5_create_chunked(file, "times", Float64, (1,); chunk_size=8)
            @test haskey(file, "times")
            @test size(file["times"]) == (0,)
        end
        HDF5.h5open(path1d, "r+") do file
            StochParticles.h5_append_row!(file, "times", 1.0)
            StochParticles.h5_append_row!(file, "times", 2.0)
            ds = file["times"]
            @test size(ds) == (2,)
            @test read(ds) ≈ [1.0, 2.0]
        end

        # Test error paths
        path_err = joinpath(dir, "test_errors.h5")
        HDF5.h5open(path_err, "w") do file
            StochParticles.h5_create_chunked(file, "data", Float64, (2, 3); chunk_size=4)
        end
        HDF5.h5open(path_err, "r+") do file
            # Missing dataset
            @test_throws ArgumentError StochParticles.h5_append_row!(file, "missing", [1.0, 2.0, 3.0])
            # Wrong row length
            @test_throws DimensionMismatch StochParticles.h5_append_row!(file, "data", [1.0, 2.0])
            # Missing group for h5_read_attrs
            @test_throws ArgumentError StochParticles.h5_read_attrs(file, "missing_group")
        end

        # Test schema validation
        path_schema = joinpath(dir, "test_schema.h5")
        HDF5.h5open(path_schema, "w") do file
            StochParticles.h5_write_attrs(file, "meta"; schema_version="1.0.0")
            @test StochParticles.validate_schema_version(file) == "1.0.0"
        end
        HDF5.h5open(path_schema, "r+") do file
            # Minor mismatch warns
            StochParticles.h5_write_attrs(file, "meta"; schema_version="1.1.0")
            @test_logs (:warn, r"Schema version minor mismatch") StochParticles.validate_schema_version(file)
        end
        HDF5.h5open(path_schema, "r+") do file
            # Major mismatch throws
            StochParticles.h5_write_attrs(file, "meta"; schema_version="2.0.0")
            @test_throws ErrorException StochParticles.validate_schema_version(file)
        end
        HDF5.h5open(path_schema, "r+") do file
            # Missing /meta throws
            @test_throws ErrorException StochParticles.validate_schema_version(file)
        end
    end
end
