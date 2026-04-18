# test/runtests.jl
using StochParticles
using Test

@testset "StochParticles.jl" begin
    include("test_particle_system.jl")
    include("test_process_trait.jl")
    include("test_condensation.jl")
    include("test_coagulation.jl")
    include("test_cnmc.jl")
    include("test_emission.jl")
    include("test_dilution.jl")
end
