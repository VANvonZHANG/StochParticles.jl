# test/test_cnmc.jl
using StochParticles
using Test
using StaticArrays
using Random

@testset "CNMC" begin
    @testset "merge conserves mass" begin
        A = 2
        u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # 4 particles
        gas_fn = t -> SVector(0.0, 0.0)
        sys = ParticleSystem(Val(A), 4, 1.0, gas_fn)
        mass_before = total_mass(u, Val(A), sys.n_active)

        cnmc_merge!(u, sys, 1, 2)

        mass_after = total_mass(u, Val(A), sys.n_active)
        # Merge: particle 1 absorbs particle 2, particle 2 becomes zero
        @test get_particle(u, 1, Val(A)) == SVector(4.0, 6.0)
        @test sys.n_active == 3
        @test mass_before ≈ mass_after  # mass conserved after merge
    end

    @testset "clone copies and increments" begin
        A = 2
        u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 3 particles, only 2 active
        gas_fn = t -> SVector(0.0, 0.0)
        sys = ParticleSystem(Val(A), 2, 1.0, gas_fn)

        cnmc_clone!(u, sys, 2, 1)  # clone particle 1 into slot 2

        @test sys.n_active == 3
        @test get_particle(u, 2, Val(A)) == get_particle(u, 1, Val(A))
    end

    @testset "volume rescale conserves concentration" begin
        A = 1
        u = [1.0, 2.0, 3.0]
        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(A), 3, 1.0, gas_fn)
        sys._mass_total_cache = 6.0  # 1.0 + 2.0 + 3.0

        cnmc_volume_rescale!(sys, get_particle(u, 2, Val(A)))
        # V_new = V_old × (1 + |μ_clone| / M_total)
        expected_V = 1.0 * (1.0 + 2.0 / 6.0)  # = 1.333...
        @test sys.volume ≈ expected_V

        # Concentration n = N_sim / V should decrease
        n_before = 3 / 1.0
        n_after = 3 / sys.volume
        @test n_after < n_before
    end

    @testset "cnmc_coagulate! maintains n_sim" begin
        A = 1
        u = fill(1.0, 10)  # 10 particles
        gas_fn = t -> SVector(0.0)
        sys = ParticleSystem(Val(A), 10, 1.0, gas_fn)

        for _ in 1:50
            i = rand(1:sys.n_active)
            j = rand(1:sys.n_active)
            while j == i
                j = rand(1:sys.n_active)
            end
            cnmc_coagulate!(u, sys, Val(A), i, j)
        end

        @test sys.n_active == sys.n_sim  # always maintains n_sim
        @test sys.volume > 1.0  # volume grows with coagulation
    end
end
