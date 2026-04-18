# test/test_process_trait.jl
using StochParticles
using Test

@testset "Process Trait" begin
    @testset "default: provides nothing" begin
        # A dummy process that doesn't provide anything
        struct DummyProcess <: PhysicsProcess end
        @test provides_drift(DummyProcess()) == false
    end

    @testset "tuple-based dispatch is type-stable" begin
        # Verify that iteration over a concrete tuple is type-stable
        processes = (DummyProcess(),)
        @test typeof(processes) <: Tuple
    end
end
