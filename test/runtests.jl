# test/runtests.jl
using StochParticles
using Test
using Aqua

# Shared test helper: minimal mock integrator for testing jump affect! functions
struct MockIntegrator
    u::Vector{Float64}
    p::ParticleSystem
    t::Float64
end

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(StochParticles)
end

@testset "StochParticles.jl" begin
    include("test_particle_system.jl")
    include("test_process_trait.jl")
    include("test_condensation.jl")
    include("test_coagulation.jl")
    include("test_cnmc.jl")
    include("test_emission.jl")
    include("test_dilution.jl")
    include("test_integration.jl")
    include("test_diagnostics.jl")
    include("test_plotting.jl")
end
