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

    @testset "virtual_smps complementarity" begin
        bin_edges = [1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5]
        dry_diams = [5.0e-9, 5.0e-8, 5.0e-8, 5.0e-7, 5.0e-7, 5.0e-7]
        flags = [false, true, false, true, true, false]
        smps = virtual_smps(flags, dry_diams, bin_edges, 1.0)
        total = dNdlogD_from_diameters(dry_diams, bin_edges, 1.0)
        @test smps.cr + smps.ci ≈ total
        # CR 只含 flag=true 的颗粒：bin2 一颗(5e-8)，bin3 两颗(5e-7,5e-7)
        @test smps.cr[1] ≈ 0.0 atol = 1e-30
        @test smps.cr[2] ≈ 1.0 / log10(1.0e-7 / 1.0e-8)
        @test smps.cr[3] ≈ 2.0 / log10(1.0e-6 / 1.0e-7)
    end

    @testset "virtual_acsm dry fractions" begin
        gas_fn = t -> SVector(0.0, 0.0, 0.0, 0.0, 0.0)
        # A=5: [AS, AN, OA, BC, H2O]; mask=[1,2,3]
        p1 = SVector(3.0, 1.0, 1.0, 2.0, 0.0)   # flag=true
        p2 = SVector(1.0, 1.0, 2.0, 0.0, 5.0)   # flag=false（含水，mask 外）
        p3 = SVector(0.0, 2.0, 2.0, 1.0, 0.0)   # flag=true
        sys = ParticleSystem(Val(5), 3, 1.0, gas_fn)
        u = make_u0([p1, p2, p3])
        acsm = virtual_acsm(u, sys, Val(5), [true, false, true]; species = [1, 2, 3])
        # CR: p1+p3 masked [3,3,3]/9 -> [1/3,1/3,1/3]
        @test acsm.cr ≈ [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        # CI: [1,1,2]/4
        @test acsm.ci ≈ [0.25, 0.25, 0.5]
        # 全员 CI → cr = NaN（空组）
        acsm2 = virtual_acsm(u, sys, Val(5), [false, false, false]; species = [1, 2, 3])
        @test all(isnan, acsm2.cr)
    end
end
