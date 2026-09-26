using StochParticles
using Test
using Random
using StaticArrays

@testset "Virtual GCVI" begin
    @testset "GCVIResponse defaults and transmission" begin
        resp = GCVIResponse()
        @test resp.D50 == 7.0e-6
        @test resp.width == 7.0e-7
        @test resp.E_max == 0.6
        @test transmission(resp, 7.0e-6) ≈ 0.3
        # logistic 两端
        @test transmission(resp, 0.0) < 1e-4
        @test abs(transmission(resp, 1.0e-3) - 0.6) < 1e-9
        # 单调
        @test transmission(resp, 5.0e-6) < transmission(resp, 7.0e-6) <
              transmission(resp, 1.0e-5)
    end

    @testset "classify_cr_ci" begin
        gas_fn = t -> SVector(0.0, 0.0)
        densities = SVector(1000.0, 1000.0)
        # 小霾滴（~0.6 μm）与大云滴（~12.4 μm）：A=2，物种2=水
        μ_small = SVector(1.0e-16, 0.0)
        μ_big = SVector(0.0, 1.0e-12)
        particles = [μ_small, μ_small, μ_small, μ_big, μ_big, μ_big]
        sys = ParticleSystem(Val(2), 6, 1.0, gas_fn)
        u = make_u0(particles)

        # 窄 transition + E_max=1 → 确定性分类（两端概率 ≈ 1e-34 / 1）
        resp = GCVIResponse(D50 = 7.0e-6, width = 7.0e-8, E_max = 1.0)
        flags = classify_cr_ci(u, sys, Val(2), resp, densities; rng = MersenneTwister(42))
        @test flags == [false, false, false, true, true, true]

        # 同 RNG 种子 → 逐位可复现
        flags2 = classify_cr_ci(u, sys, Val(2), resp, densities; rng = MersenneTwister(42))
        @test flags == flags2

        # 统计性：大滴 E_max=0.5 → 约半数入选
        resp_half = GCVIResponse(D50 = 7.0e-6, width = 7.0e-8, E_max = 0.5)
        n_big = 4000
        sys_big = ParticleSystem(Val(2), n_big, 1.0, gas_fn)
        u_big = make_u0([μ_big for _ in 1:n_big])
        f_big = classify_cr_ci(u_big, sys_big, Val(2), resp_half, densities)
        @test 0.44 < count(f_big) / n_big < 0.56
    end
end
