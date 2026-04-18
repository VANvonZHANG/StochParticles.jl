# test/runtests.jl
using StochParticles
using Test

@testset "StochParticles.jl" begin
    include("test_particle_system.jl")
    include("test_process_trait.jl")
end
