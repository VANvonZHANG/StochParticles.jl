using StochParticles
using Test
using StaticArrays

@testset "Species struct" begin
    s = Species(:TEST, 1500.0, 0.2, 0.200)
    @test s.name == :TEST
    @test s.density == 1500.0
    @test s.kappa == 0.2
    @test s.molar_mass == 0.200

    # keyword constructor
    s2 = Species(name = :KWD, density = 1600.0, kappa = 0.3, molar_mass = 0.180)
    @test s2.name == :KWD
    @test s2.density == 1600.0
end

@testset "Preset species values" begin
    @test AS.name == :AS
    @test AS.density == 1770.0
    @test AS.kappa == 0.61
    @test AS.molar_mass == 0.13214

    @test AN.name == :AN
    @test AN.density == 1720.0
    @test AN.kappa == 0.67
    @test AN.molar_mass == 0.08004

    @test BC.name == :BC
    @test BC.density == 1800.0
    @test BC.kappa == 0.0
    @test BC.molar_mass == 0.01201

    @test OA.name == :OA
    @test OA.density == 1400.0
    @test OA.kappa == 0.1
    @test OA.molar_mass == 0.250

    @test H2O.name == :H2O
    @test H2O.density == 1000.0
    @test H2O.kappa == 0.0
    @test H2O.molar_mass == 0.018015
end

@testset "species_vectors basic" begin
    params = species_vectors(AS, BC, H2O)

    @test params.densities == SVector(1770.0, 1800.0, 1000.0)
    @test params.kappas == SVector(0.61, 0.0, 0.0)
    @test params.molar_masses == SVector(0.13214, 0.01201, 0.018015)
    @test params.names == ["AS", "BC", "H2O"]
    @test params.h2o_idx == 3
end

@testset "species_vectors h2o_idx position" begin
    params = species_vectors(H2O, AS)
    @test params.h2o_idx == 1
    @test params.densities == SVector(1000.0, 1770.0)

    params2 = species_vectors(AS, H2O)
    @test params2.h2o_idx == 2
    @test params2.densities == SVector(1770.0, 1000.0)
end

@testset "species_vectors type stability" begin
    params = @inferred species_vectors(AS, BC, H2O)
    @test params.densities isa SVector{3, Float64}
    @test params.kappas isa SVector{3, Float64}
    @test params.molar_masses isa SVector{3, Float64}
end

@testset "species_vectors missing H2O" begin
    @test_throws ArgumentError species_vectors(AS, BC)
end

@testset "species_vectors empty" begin
    @test_throws ArgumentError species_vectors()
end

@testset "species_vectors with custom species" begin
    custom = Species(:CUSTOM, 1500.0, 0.2, 0.300)
    params = species_vectors(AS, custom, H2O)
    @test params.densities == SVector(1770.0, 1500.0, 1000.0)
    @test params.kappas == SVector(0.61, 0.2, 0.0)
    @test params.names == ["AS", "CUSTOM", "H2O"]
    @test params.h2o_idx == 3
end
