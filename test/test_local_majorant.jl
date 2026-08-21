# test/test_local_majorant.jl
using StochParticles
using Test
using StaticArrays

@testset "LocalMajorant bounds" begin
    kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))
    particles = [SVector(1.0e-18), SVector(2.0e-18), SVector(3.0e-18), SVector(5.0e-17)]
    u0 = make_u0(particles)
    sys = ParticleSystem(Val(1), 4, 1.0, t -> SVector(0.0))
    bounds = StochParticles.local_majorant_bounds(kernel, u0, sys)
    @test length(bounds) == 4
    for i in 1:4, j in 1:4
        i == j && continue
        K_ij = kernel(get_particle(u0, i, Val(1)), get_particle(u0, j, Val(1)))
        @test bounds[i] >= K_ij
    end
    # exactness: the extreme particle's bound equals its true row max
    row_max = maximum(kernel(get_particle(u0, 4, Val(1)), get_particle(u0, j, Val(1)))
                      for j in 1:3)
    @test bounds[4] == row_max
    # LocalMajorant is a CoagulationSampling usable in process constructors
    proc = NonCNMCCoagulationProcess(kernel, LocalMajorant())
    @test proc isa NonCNMCCoagulationProcess
end
