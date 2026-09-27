using StochParticles
using Test
using StaticArrays

@testset "PrescribedProfile" begin
    src = PrescribedProfile(
        [0.0, 120.0, 600.0], [288.15, 288.15, 288.15], [0.0, 0.004, 0.004])

    # 节点处取精确值
    @test src(0.0)[1] ≈ 288.15
    @test src(0.0)[2] ≈ saturation_vapor_pressure(288.15)
    @test src(120.0)[2] ≈ saturation_vapor_pressure(288.15) * 1.004

    # 节点间线性插值（t=60 → S=0.002）
    @test src(60.0)[2] ≈ saturation_vapor_pressure(288.15) * 1.002

    # 两端平推外插
    @test src(-10.0)[2] ≈ saturation_vapor_pressure(288.15)
    @test src(1.0e4)[2] ≈ saturation_vapor_pressure(288.15) * 1.004

    # p_v 单点定义不变量 + 返回类型
    T, p_v = src(60.0)
    @test p_v ≈ saturation_vapor_pressure(T) * (1.0 + 0.002)
    @test src(60.0) isa SVector{2, Float64}

    # 构造校验
    @test_throws ArgumentError PrescribedProfile([0.0, 0.0], [288.0, 288.0], [0.0, 0.0])
    @test_throws ArgumentError PrescribedProfile([0.0], [288.0, 288.0], [0.0])
end
