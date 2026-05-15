using StochParticles
using Test
using StaticArrays

@testset "Mixing State" begin
    @testset "species_fractions" begin
        # Normal case
        μ = SVector(3.0, 7.0)
        f = @inferred(species_fractions(μ))
        @test f ≈ SVector(0.3, 0.7)
        @test sum(f) ≈ 1.0

        # Uniform fallback for zero mass
        μ_zero = SVector(0.0, 0.0)
        f_zero = @inferred(species_fractions(μ_zero))
        @test f_zero ≈ SVector(0.5, 0.5)

        # Single species
        μ1 = SVector(5.0)
        f1 = @inferred(species_fractions(μ1))
        @test f1 ≈ SVector(1.0)

        # Very small total mass
        μ_tiny = SVector(1e-300, 2e-300)
        @test @inferred(species_fractions(μ_tiny)) ≈ SVector(1.0 / 3.0, 2.0 / 3.0)

        # Three species
        μ3 = SVector(1.0, 2.0, 3.0)
        @test @inferred(species_fractions(μ3)) ≈ SVector(1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0)
    end

    @testset "shannon_entropy" begin
        # Pure particle has zero entropy
        f_pure = SVector(1.0, 0.0)
        @test @inferred(shannon_entropy(f_pure)) ≈ 0.0 atol = 1e-15

        # Uniform binary mixture
        f_uniform = SVector(0.5, 0.5)
        @test @inferred(shannon_entropy(f_uniform)) ≈ log(2.0) rtol = 1e-10

        # Uniform ternary mixture
        f_3 = SVector(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        @test @inferred(shannon_entropy(f_3)) ≈ log(3.0) rtol = 1e-10
    end

    @testset "mixing_state_index" begin
        gas_fn = t -> SVector(0.0)

        # Fully external mixture: 100 pure SO4 + 100 pure BC
        sys_ext = ParticleSystem(Val(2), 200, 1.0, gas_fn)
        u_ext = make_u0(vcat(
            [SVector(1.0, 0.0) for _ in 1:100],
            [SVector(0.0, 1.0) for _ in 1:100]
        ))
        chi_ext = @inferred mixing_state_index(u_ext, sys_ext)
        @test chi_ext ≈ 0.0 atol = 0.05

        # Fully internal mixture: all particles have same composition
        sys_int = ParticleSystem(Val(2), 200, 1.0, gas_fn)
        u_int = make_u0([SVector(0.5, 0.5) for _ in 1:200])
        chi_int = @inferred mixing_state_index(u_int, sys_int)
        @test chi_int ≈ 1.0 atol = 1e-10

        # Single species returns 1.0
        sys_1 = ParticleSystem(Val(1), 100, 1.0, gas_fn)
        u_1 = make_u0([SVector(1.0) for _ in 1:100])
        @test @inferred(mixing_state_index(u_1, sys_1)) ≈ 1.0

        # Partially mixed: 2 pure + 2 mixed (50/50)
        sys_partial = ParticleSystem(Val(2), 4, 1.0, gas_fn)
        u_partial = make_u0([
            SVector(1.0, 0.0),
            SVector(0.0, 1.0),
            SVector(0.5, 0.5),
            SVector(0.5, 0.5)
        ])
        chi_partial = @inferred mixing_state_index(u_partial, sys_partial)
        @test 0.0 < chi_partial < 1.0

        # System with one zero-mass particle
        sys_zero = ParticleSystem(Val(2), 3, 1.0, gas_fn)
        u_zero = make_u0([
            SVector(1.0, 0.0),
            SVector(0.0, 1.0),
            SVector(0.0, 0.0)
        ])
        chi_zero = mixing_state_index(u_zero, sys_zero)
        @test isfinite(chi_zero)  # should not be NaN or Inf
    end

    @testset "particle_mixing_entropy" begin
        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(2), 4, 1.0, gas_fn)
        u = make_u0([
            SVector(1.0, 0.0),   # pure, entropy = 0
            SVector(0.0, 1.0),   # pure, entropy = 0
            SVector(0.5, 0.5),   # mixed, entropy = ln(2)
            SVector(0.5, 0.5)   # mixed, entropy = ln(2)
        ])
        entropies = @inferred particle_mixing_entropy(u, sys)
        @test length(entropies) == 4
        @test entropies[1] ≈ 0.0 atol = 1e-15
        @test entropies[2] ≈ 0.0 atol = 1e-15
        @test entropies[3] ≈ log(2.0) rtol = 1e-10
        @test entropies[4] ≈ log(2.0) rtol = 1e-10
    end

    @testset "particle_diameters multi-species" begin
        gas_fn = t -> SVector(0.0)

        # Two particles with only species 1 (density 1000)
        sys1 = ParticleSystem(Val(2), 2, 1.0, gas_fn)
        u1 = make_u0([SVector(1.0e-18, 0.0), SVector(8.0e-18, 0.0)])
        diams1 = @inferred particle_diameters(u1, sys1, SVector(1000.0, 2000.0))
        @test diams1[1] ≈ (6.0 * 1.0e-18 / (π * 1000.0))^(1.0 / 3.0)
        @test diams1[2] ≈ (6.0 * 8.0e-18 / (π * 1000.0))^(1.0 / 3.0)

        # Same particle but use backward-compatible single-species call
        diams1_compat = @inferred particle_diameters(u1, sys1, 1000.0)
        @test diams1_compat ≈ diams1

        # Mixed particle: total volume = 0.5e-18/1000 + 0.5e-18/2000
        sys2 = ParticleSystem(Val(2), 1, 1.0, gas_fn)
        u2 = make_u0([SVector(0.5e-18, 0.5e-18)])
        diams2 = @inferred particle_diameters(u2, sys2, SVector(1000.0, 2000.0))
        V_expected = 0.5e-18 / 1000.0 + 0.5e-18 / 2000.0
        d_expected = (6.0 * V_expected / π)^(1.0 / 3.0)
        @test diams2[1] ≈ d_expected
    end

    @testset "lognormal_masses multi-species" begin
        densities = SVector(1770.0, 1800.0)

        # Pure SO4 mode (fractions = [1, 0])
        so4_particles = @inferred lognormal_masses(
            1000, 1.0e-7, 1.5, densities; fractions = SVector(1.0, 0.0))
        @test length(so4_particles) == 1000
        @test all(p -> p[1] > 0 && iszero(p[2]), so4_particles)

        # Mixed mode (50/50)
        mixed_particles = @inferred lognormal_masses(
            1000, 1.0e-7, 1.5, densities; fractions = SVector(0.5, 0.5))
        @test length(mixed_particles) == 1000
        @test all(p -> p[1] > 0 && p[2] > 0, mixed_particles)
        @test all(p -> p[1] ≈ p[2], mixed_particles)

        # Default fractions (pure species 1)
        default_particles = @inferred lognormal_masses(
            100, 1.0e-7, 1.5, densities)
        @test length(default_particles) == 100
        @test all(p -> p[1] > 0 && iszero(p[2]), default_particles)

        # Invalid fractions (do not sum to 1)
        @test_throws ArgumentError lognormal_masses(
            10, 1.0e-7, 1.5, densities; fractions = SVector(0.3, 0.3))

        # Backward compatible single-species call
        single_particles = @inferred lognormal_masses(100, 1.0e-7, 1.5, 1000.0)
        @test length(single_particles) == 100
        @test all(p -> p[1] > 0, single_particles)
        @test eltype(single_particles) == SVector{1, Float64}
    end

    @testset "diameters_from_masses multi-species" begin
        densities = SVector(1000.0, 2000.0)

        # Pure species 1 particle
        masses = [SVector(1.0e-18, 0.0)]
        diams = diameters_from_masses(masses, densities)
        @test diams[1] ≈ (6.0 * 1.0e-18 / (π * 1000.0))^(1.0 / 3.0)

        # Mixed particle
        masses2 = [SVector(0.5e-18, 0.5e-18)]
        diams2 = diameters_from_masses(masses2, densities)
        V_expected = 0.5e-18 / 1000.0 + 0.5e-18 / 2000.0
        @test diams2[1] ≈ (6.0 * V_expected / π)^(1.0 / 3.0)

        # Backward compatible single-species call
        masses1 = [SVector{1, Float64}(1.0e-18)]
        diams1 = diameters_from_masses(masses1, 1000.0)
        @test diams1[1] ≈ (6.0 * 1.0e-18 / (π * 1000.0))^(1.0 / 3.0)
    end

    @testset "SpeciesDependentCondensation" begin
        rates = SVector(1.0e-20, 2.0e-20)
        proc = SpeciesDependentCondensation(rates)

        μ = SVector(1.0e-18, 2.0e-18)
        gas_fn = t -> SVector(0.0, 0.0)
        sys = ParticleSystem(Val(2), 10, 1.0, gas_fn)

        dμ = @inferred apply_drift(proc, μ, sys, 0.0)
        @test dμ ≈ rates
        @test provides_drift(proc) == true
    end

    @testset "IO mixing state integration" begin
        using HDF5
        mktempdir() do dir
            h5_path = joinpath(dir, "test_mixing.h5")
            bin_edges = [1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6]

            # Create diagnostics file
            init_diagnostics_file(h5_path, 2, bin_edges;
                species_names = ["SO4", "BC"])

            gas_fn = t -> SVector(0.0, 0.0)
            sys = ParticleSystem(Val(2), 4, 1.0, gas_fn)
            u = make_u0([
                SVector(1.0, 0.0),
                SVector(0.0, 1.0),
                SVector(1.0, 0.0),
                SVector(0.0, 1.0)
            ])

            # Save one timestep
            save_diagnostics(h5_path, 0.0, u, sys, Val(2);
                bin_edges = bin_edges, rho = 1000.0)

            # Verify mixing_state dataset exists and has correct value
            HDF5.h5open(h5_path, "r") do file
                @test haskey(file, "mixing_state")
                chi_data = read(file["mixing_state"])
                @test length(chi_data) == 1
                # External mixture should have χ ≈ 0
                @test chi_data[1] ≈ 0.0 atol = 1e-10
            end
        end
    end
end
