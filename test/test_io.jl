using StochParticles
using Test
using HDF5
using Random

@testset "IO: HDF5 low-level helpers" begin
    mktempdir() do dir
        path = joinpath(dir, "test_helpers.h5")

        # Test h5_create_chunked (2D)
        HDF5.h5open(path, "w") do file
            StochParticles.h5_create_chunked(file, "data", Float64, (2, 3); chunk_size=4)
            @test haskey(file, "data")
            ds = file["data"]
            @test size(ds) == (3, 0)
            @test HDF5.get_chunk(ds)[2] == 4
        end

        # Test h5_append_row! with 2D data (append along time dim)
        HDF5.h5open(path, "r+") do file
            StochParticles.h5_append_row!(file, "data", [1.0, 2.0, 3.0])
            StochParticles.h5_append_row!(file, "data", [4.0, 5.0, 6.0])
            ds = file["data"]
            @test size(ds) == (3, 2)
            data = read(ds)
            @test data[:, 1] ≈ [1.0, 2.0, 3.0]
            @test data[:, 2] ≈ [4.0, 5.0, 6.0]
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

@testset "IO: checkpoint round-trip" begin
    mktempdir() do dir
        gas_fn(t) = [1.0, 2.0]
        sys = ParticleSystem(Val(2), 10, 1.5, gas_fn)
        sys.n_active = 7
        sys._mass_total_cache = 42.0
        sys._cached_majorant = 3.14
        u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
             11.0, 12.0, 13.0, 14.0]
        t = 0.5
        rng = Random.Xoshiro(1234)

        path = joinpath(dir, "checkpoint")
        StochParticles.save_checkpoint(path, u, sys, t; rng=rng)

        # Verify .h5 suffix appended
        @test isfile(path * ".h5")

        # Load and verify
        u_loaded, sys_data, t_loaded, rng_state = StochParticles.load_checkpoint(path * ".h5")
        @test u_loaded ≈ u
        @test sys_data.n_active == 7
        @test sys_data.volume == 1.5
        @test sys_data.n_sim == 10
        @test sys_data.mass_total_cache == 42.0
        @test sys_data.cached_majorant == 3.14
        @test t_loaded == t
        @test rng_state[:seed] == 0
        @test rng_state[:state] == Vector{UInt64}([rng.s0, rng.s1, rng.s2, rng.s3])

        # Test overwrite=false throws
        @test_throws ErrorException StochParticles.save_checkpoint(path, u, sys, t; rng=rng, overwrite=false)

        # Test overwrite=true succeeds
        StochParticles.save_checkpoint(path, u, sys, t; rng=rng, overwrite=true)

        # Test load_checkpoint throws for missing file
        @test_throws SystemError StochParticles.load_checkpoint(joinpath(dir, "nonexistent.h5"))

        # Test list_checkpoints with actual checkpoint files and numeric sort
        cp1 = joinpath(dir, "run_001.h5")
        cp2 = joinpath(dir, "run_010.h5")
        cp3 = joinpath(dir, "run_100.h5")
        touch(cp1)
        touch(cp2)
        touch(cp3)
        cps = StochParticles.list_checkpoints(joinpath(dir, "run"))
        @test cps == [cp1, cp2, cp3]

        # Test list_checkpoints numeric sort (run_10 after run_2)
        cp4 = joinpath(dir, "run_2.h5")
        cp5 = joinpath(dir, "run_10.h5")
        touch(cp4)
        touch(cp5)
        cps2 = StochParticles.list_checkpoints(joinpath(dir, "run"))
        @test cps2 == [cp1, cp4, cp2, cp5, cp3]

        # Test load_checkpoint throws SystemError on missing file
        @test_throws SystemError StochParticles.load_checkpoint(joinpath(dir, "nonexistent.h5"))

        # Test save_checkpoint throws ErrorException on existing file with overwrite=false
        @test_throws ErrorException StochParticles.save_checkpoint(path, u, sys, t; rng=rng, overwrite=false)

        # Test restore_rng sequence determinism
        rng_orig = Random.Xoshiro(5678)
        # Advance original RNG
        _ = rand(rng_orig, 5)
        StochParticles.save_checkpoint(joinpath(dir, "rng_test"), u, sys, t; rng=rng_orig)
        _, _, _, rng_state_loaded = StochParticles.load_checkpoint(joinpath(dir, "rng_test.h5"))
        rng_restored = StochParticles.restore_rng(rng_state_loaded)
        # Verify same future sequence
        seq_orig = [rand(rng_orig) for _ in 1:10]
        seq_restored = [rand(rng_restored) for _ in 1:10]
        @test seq_orig == seq_restored
    end
end

@testset "IO: diagnostics init and append" begin
    mktempdir() do dir
        gas_fn(t) = [1.0, 2.0]
        sys = ParticleSystem(Val(2), 5, 1.0, gas_fn)
        sys.n_active = 5
        u = [1.0, 2.0, 3.0, 4.0, 5.0,
             6.0, 7.0, 8.0, 9.0, 10.0]
        bin_edges = [0.0, 1.0e-6, 2.0e-6, 3.0e-6]
        n_bins = length(bin_edges) - 1
        path = joinpath(dir, "diagnostics.h5")

        # Test init_diagnostics_file creates file with correct structure
        StochParticles.init_diagnostics_file(path, 2, bin_edges; species_names = ["SO4", "NO3"])
        @test isfile(path)

        HDF5.h5open(path, "r") do file
            @test haskey(file, "time")
            @test haskey(file, "number_concentration")
            @test haskey(file, "mass_concentration")
            @test haskey(file, "species_mass_concentration")
            @test haskey(file, "mean_diameter")
            @test haskey(file, "volume")
            @test haskey(file, "size_distribution")
            @test haskey(file, "meta")

            @test size(file["time"]) == (0,)
            @test size(file["number_concentration"]) == (0,)
            @test size(file["mass_concentration"]) == (0,)
            @test size(file["species_mass_concentration"]) == (2, 0)
            @test size(file["mean_diameter"]) == (0,)
            @test size(file["volume"]) == (0,)
            @test size(file["size_distribution"]) == (n_bins, 0)

            meta_attrs = StochParticles.h5_read_attrs(file, "meta")
            @test meta_attrs["schema_version"] == "1.0.0"
            @test meta_attrs["species_names"] == ["SO4", "NO3"]
            @test meta_attrs["bin_edges"] ≈ bin_edges
        end

        # Test save_diagnostics appends data correctly
        t1 = 0.0
        StochParticles.save_diagnostics(path, t1, u, sys, Val(2); bin_edges = bin_edges, rho = 1000.0)

        t2 = 1.0
        StochParticles.save_diagnostics(path, t2, u, sys, Val(2); bin_edges = bin_edges, rho = 1000.0)

        HDF5.h5open(path, "r") do file
            @test size(file["time"]) == (2,)
            @test read(file["time"]) ≈ [t1, t2]

            @test size(file["number_concentration"]) == (2,)
            @test read(file["number_concentration"]) ≈ [sys.n_sim / sys.volume, sys.n_sim / sys.volume]

            @test size(file["mass_concentration"]) == (2,)
            M = StochParticles.total_mass(u, Val(2), sys.n_active)
            expected_mass_conc = M / sys.volume
            @test read(file["mass_concentration"]) ≈ [expected_mass_conc, expected_mass_conc]

            @test size(file["species_mass_concentration"]) == (2, 2)
            smc = read(file["species_mass_concentration"])
            for s in 1:2
                expected_smc = StochParticles.species_mass_concentration(u, s, Val(2), sys)
                @test smc[s, :] ≈ [expected_smc, expected_smc]
            end

            @test size(file["size_distribution"]) == (n_bins, 2)
        end

        # Test error paths
        @test_throws ErrorException StochParticles.init_diagnostics_file(path, 2, bin_edges)
        @test_throws SystemError StochParticles.save_diagnostics(joinpath(dir, "nonexistent.h5"), 0.0, u, sys, Val(2))

        bad_path = joinpath(dir, "bad.h5")
        HDF5.h5open(bad_path, "w") do file
            file["other"] = 1.0
        end
        @test_throws ErrorException StochParticles.save_diagnostics(bad_path, 0.0, u, sys, Val(2))

        # Test bin_edges validation
        @test_throws ArgumentError StochParticles.init_diagnostics_file(joinpath(dir, "bad_edges.h5"), 2, [1.0])

        # Test species_names length validation
        @test_throws ArgumentError StochParticles.init_diagnostics_file(joinpath(dir, "bad_names.h5"), 2, bin_edges; species_names = ["SO4"])

        # Test bin_edges mismatch in save_diagnostics
        path_mismatch = joinpath(dir, "mismatch.h5")
        StochParticles.init_diagnostics_file(path_mismatch, 2, bin_edges; species_names = ["SO4", "NO3"])
        mismatched_edges = [0.0, 2.0e-6, 4.0e-6]
        @test_throws ArgumentError StochParticles.save_diagnostics(path_mismatch, 0.0, u, sys, Val(2); bin_edges = mismatched_edges)
    end
end

@testset "IO: JLD2 fallback round-trip" begin
    mktempdir() do dir
        gas_fn(t) = [1.0, 2.0]
        sys = ParticleSystem(Val(2), 10, 1.5, gas_fn)
        sys.n_active = 7
        sys._mass_total_cache = 42.0
        sys._cached_majorant = 3.14
        u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
             11.0, 12.0, 13.0, 14.0]
        t = 0.5
        rng = Random.Xoshiro(1234)

        path = joinpath(dir, "checkpoint")
        StochParticles.save_checkpoint_jld2(path, u, sys, t; rng=rng)

        # Verify .jld2 suffix appended
        @test isfile(path * ".jld2")

        # Load and verify
        u_loaded, sys_data, t_loaded, rng_state = StochParticles.load_checkpoint_jld2(path * ".jld2")
        @test u_loaded ≈ u
        @test sys_data.n_active == 7
        @test sys_data.volume == 1.5
        @test sys_data.n_sim == 10
        @test sys_data.mass_total_cache == 42.0
        @test sys_data.cached_majorant == 3.14
        @test t_loaded == t
        @test rng_state[:seed] == 0
        @test rng_state[:state] == Vector{UInt64}([rng.s0, rng.s1, rng.s2, rng.s3])

        # Test overwrite=false throws
        @test_throws ErrorException StochParticles.save_checkpoint_jld2(path, u, sys, t; rng=rng, overwrite=false)

        # Test overwrite=true succeeds
        StochParticles.save_checkpoint_jld2(path, u, sys, t; rng=rng, overwrite=true)

        # Test load_checkpoint_jld2 throws for missing file
        @test_throws SystemError StochParticles.load_checkpoint_jld2(joinpath(dir, "nonexistent.jld2"))

        # Test restore_rng sequence determinism for JLD2
        rng_orig = Random.Xoshiro(5678)
        # Advance original RNG
        _ = rand(rng_orig, 5)
        StochParticles.save_checkpoint_jld2(joinpath(dir, "rng_test"), u, sys, t; rng=rng_orig)
        _, _, _, rng_state_loaded = StochParticles.load_checkpoint_jld2(joinpath(dir, "rng_test.jld2"))
        rng_restored = StochParticles.restore_rng(rng_state_loaded)
        # Verify same future sequence
        seq_orig = [rand(rng_orig) for _ in 1:10]
        seq_restored = [rand(rng_restored) for _ in 1:10]
        @test seq_orig == seq_restored
    end
end

@testset "IO: utility functions" begin
    mktempdir() do dir
        # Test list_checkpoints numeric sorting with ckpt_001, ckpt_010, ckpt_002
        cp1 = joinpath(dir, "ckpt_001.h5")
        cp2 = joinpath(dir, "ckpt_010.h5")
        cp3 = joinpath(dir, "ckpt_002.h5")
        touch(cp1)
        touch(cp2)
        touch(cp3)
        cps = StochParticles.list_checkpoints(joinpath(dir, "ckpt"))
        @test cps == [cp1, cp3, cp2]

        # Test export_diagnostics_to_csv
        gas_fn(t) = [1.0, 2.0]
        sys = ParticleSystem(Val(2), 5, 1.0, gas_fn)
        sys.n_active = 5
        u = [1.0, 2.0, 3.0, 4.0, 5.0,
             6.0, 7.0, 8.0, 9.0, 10.0]
        bin_edges = [0.0, 1.0e-6, 2.0e-6, 3.0e-6]
        diag_path = joinpath(dir, "export_diagnostics.h5")
        csv_dir = joinpath(dir, "csv_output")

        StochParticles.init_diagnostics_file(diag_path, 2, bin_edges; species_names = ["SO4", "NO3"])
        StochParticles.save_diagnostics(diag_path, 0.0, u, sys, Val(2); bin_edges = bin_edges, rho = 1000.0)
        StochParticles.save_diagnostics(diag_path, 1.0, u, sys, Val(2); bin_edges = bin_edges, rho = 1000.0)

        StochParticles.export_diagnostics_to_csv(diag_path, csv_dir)

        @test isdir(csv_dir)
        @test isfile(joinpath(csv_dir, "number_concentration.csv"))
        @test isfile(joinpath(csv_dir, "mass_concentration.csv"))
        @test isfile(joinpath(csv_dir, "all_diagnostics.csv"))

        # Read back number_concentration.csv and verify number of rows (header + 2 data rows)
        nc_lines = readlines(joinpath(csv_dir, "number_concentration.csv"))
        @test length(nc_lines) == 3
        @test nc_lines[1] == "time,number_concentration"
        @test startswith(nc_lines[2], "0.0,")
        @test startswith(nc_lines[3], "1.0,")

        # Read back all_diagnostics.csv and verify header includes 1D datasets
        all_lines = readlines(joinpath(csv_dir, "all_diagnostics.csv"))
        @test length(all_lines) == 3
        @test startswith(all_lines[1], "time,")
    end
end
