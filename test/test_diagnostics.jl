# test/test_diagnostics.jl
using StochParticles
using Test
using StaticArrays

@testset "Diagnostics" begin
    @testset "zeroth moment (total number concentration)" begin
        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(1), 100, 2.0, gas_fn)
        u0 = fill(1.0, 100)

        @test number_concentration(sys) == 100 / 2.0  # N/V = 100/2 = 50
    end

    @testset "first moment (total mass concentration)" begin
        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(2), 3, 1.0, gas_fn)
        u0 = make_u0([SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)])

        M = mass_concentration(u0, Val(2), sys)
        @test M ≈ (1+2+3+4+5+6) / 1.0  # total mass / volume
    end
end
