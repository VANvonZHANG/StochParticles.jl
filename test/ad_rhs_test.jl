using StochParticles
using StaticArrays
using ForwardDiff
using OrdinaryDiffEq
using SparseArrays: nnz
using Test

# PR3 (M4 ki-enabler) acceptance: the FULL parcel-coupled RHS — particle
# slots + 3-slot tail, no Ref-box indirection — is AD-transparent, and
# implicit solvers (Rodas5/QNDF) reproduce the explicit reference on the
# kinetic scenario that motivated the whole stack.

const THERMO_RHS = ThermodynamicsParams(SVector(0.61, 0.0), 0.072, 1000.0,
    18.015e-3, 2.5e6, 461.5, 2.5e-5, 2.4e-2)
const DENS_RHS = SVector(1770.0, 1000.0)

function kinetic_setup(; sizes_nm = (50.0, 80.0, 120.0))
    T0, p0, S0 = 285.0, 9.0e4, -0.002
    p_sat = saturation_vapor_pressure(T0)
    particles = [SVector(4.0 / 3.0 * π * (0.5e-9 * d)^3 * 1770.0, 0.0)
                 for d in sizes_nm]
    pre_equilibrate!(particles, THERMO_RHS, DENS_RHS, T0, p_sat * (1.0 + S0);
        h2o_idx = 2)
    qv0 = (1.0 + S0) * (0.622 * p_sat) / p0    # S0-consistent qv from S = qv·p/(ε·p_sat) − 1
    pp = ParcelProcess(1.0, 2, 1.0e-10, THERMO_RHS, T0, p0, qv0)
    cond = H2OCondensationProcess(THERMO_RHS, DENS_RHS; h2o_idx = 2, w = 0.0,
        activation_gate = :kinetic)
    return particles, pp, cond
end

@testset "full-RHS Jacobian: ForwardDiff vs finite differences" begin
    particles, pp, cond = kinetic_setup()
    procs = (cond, pp)
    f! = make_ode_func(procs)
    sys = ParticleSystem(Val(2), 3, 1.0e-6, ParcelCoupled(pp.parcel))
    u0 = vcat([v for p in particles for v in p]..., pp.parcel[].T,
        pp.parcel[].p, pp.parcel[].qv)
    @test length(u0) == 9                       # 3 particles x 2 + tail

    F(u) = (du = similar(u); f!(du, u, sys, 0.0); du)
    J = ForwardDiff.jacobian(F, u0)
    @test all(isfinite, J)

    # structure: AD zeros are EXACT zeros — dry rows, cross-particle blocks
    @test all(iszero, J[1, :]) && all(iszero, J[3, :]) && all(iszero, J[5, :])
    @test all(iszero, J[2, 3:6])                # particle 1 vs particles 2/3
    @test all(iszero, J[4, 1:2]) && all(iszero, J[4, 5:6])
    @test !all(iszero, J[2, 7:9])               # ... vs the parcel tail
    @test !iszero(J[7, 2]) && !iszero(J[7, 4]) && !iszero(J[7, 6])  # tail vs Σṁ_w

    # values: central finite differences on representative entries
    # (rows: 2 = dm₁/dt; 7,8,9 = dT,dp,dqv; cols: 1 = m_dry₁, 2 = m_w₁,
    #  4 = m_w₂, 7,8,9 = T, p, qv)
    fd(j) = begin
        h = 1e-6 * max(abs(u0[j]), 1e-24)
        up, um = copy(u0), copy(u0)
        up[j] += h
        um[j] -= h
        (F(up) .- F(um)) ./ (2h)
    end
    for (r,
        c) in ((2, 1), (2, 2), (2, 7), (2, 8), (2, 9),
        (7, 2), (7, 4), (8, 7), (8, 8), (9, 4))
        col = fd(c)
        @test J[r, c] ≈ col[r] rtol = 1e-4
    end
end

@testset "rhs_jac_prototype matches the analytic Jacobian support" begin
    B = rhs_jac_prototype(3, 2)
    @test size(B) == (9, 9)
    particles, pp, cond = kinetic_setup()
    f! = make_ode_func((cond, pp))
    sys = ParticleSystem(Val(2), 3, 1.0e-6, ParcelCoupled(pp.parcel))
    u0 = vcat([v for p in particles for v in p]..., pp.parcel[].T,
        pp.parcel[].p, pp.parcel[].qv)
    F(u) = (du = similar(u); f!(du, u, sys, 0.0); du)
    J = ForwardDiff.jacobian(F, u0)
    # every significant AD entry must fall inside the prototype support
    sig = abs.(J) .> 1e-9 * maximum(abs.(J))
    @test all(sig .<= Bool.(Matrix(B)))
    @test nnz(B) == 3 * 2 * (2 + 3) + 3 * 9     # n·A·(A+3) + 3·N
end

@testset "implicit solvers reproduce Tsit5 (kinetic closed loop)" begin
    rec = (t,
        u,
        sys) -> (
        t = t,
        S = parcel_supersaturation(extract_parcel(u, sys.n_sim, Val(2))),
        mw = sum(get_particle(u, i, Val(2))[2] for i in 1:(sys.n_active)))
    function run(solver; jac = nothing)
        particles, pp, cond = kinetic_setup()
        solve_split(particles, 1.0e-6, ParcelCoupled(pp.parcel), (cond, pp),
            solver; tspan = (0.0, 120.0), n_sim = 3, dt_split = 5.0,
            saveat = 5.0, record_func = rec, abstol = 1.0e-24,
            reltol = 1.0e-8, jac_prototype = jac)
    end
    sol_ref, ref = run(Tsit5())
    @test sol_ref.retcode == ReturnCode.Success
    for (solver,
        tag,
        jac) in ((Rodas5(), "Rodas5-dense", nothing),
        (Rodas5(), "Rodas5-sparse", rhs_jac_prototype(3, 2)),
        (QNDF(), "QNDF-sparse", rhs_jac_prototype(3, 2)))
        sol, out = run(solver; jac)
        @test sol.retcode == ReturnCode.Success
        @test out[end].S ≈ ref[end].S rtol = 1e-5
        @test out[end].mw ≈ ref[end].mw rtol = 1e-5
        @test out[5].S ≈ ref[5].S rtol = 1e-4   # mid-trajectory agreement
    end
end

@testset "parcel box mirrors the accepted state after solve_split" begin
    particles, pp, cond = kinetic_setup()
    sol,
    recs = solve_split(particles, 1.0e-6, ParcelCoupled(pp.parcel),
        (cond, pp), Tsit5(); tspan = (0.0, 60.0), n_sim = 3, dt_split = 5.0,
        saveat = 60.0, record_func = (t, u, sys) -> (t = t,),
        abstol = 1.0e-24, reltol = 1.0e-8)
    pr = extract_parcel(sol.u[end], 3, Val(2))
    @test pp.parcel[].T == pr.T && pp.parcel[].p == pr.p &&
          pp.parcel[].qv == pr.qv
end
