# test/test_local_majorant.jl
using StochParticles
using Test
using StaticArrays
using Random
using Statistics
using OrdinaryDiffEq

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

@testset "step_coagulation! bound maintenance" begin
    kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))
    particles = [SVector(1.0e-18 * k) for k in 1:6]
    u = make_u0(particles)
    sys = ParticleSystem(Val(1), 6, 1.0e-6, t -> SVector(0.0))
    proc = NonCNMCCoagulationProcess(kernel, LocalMajorant())
    bounds = StochParticles.local_majorant_bounds(kernel, u, sys)
    m_before = StochParticles.total_mass(u, Val(1), sys.n_active)
    StochParticles.non_cnmc_coagulate!(u, sys, Val(1), 2, 4)
    StochParticles.refresh_local_bounds!(kernel, u, sys, bounds, proc, 2, 4)
    fresh = StochParticles.local_majorant_bounds(kernel, u, sys)
    @test sys.n_active == 5
    @test StochParticles.total_mass(u, Val(1), sys.n_active) ≈ m_before
    for k in 1:(sys.n_active)
        @test bounds[k] >= fresh[k] - 1.0e-15 * max(1.0, fresh[k])
    end
end

@testset "step_coagulation! frozen SSA statistics" begin
    kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))
    masses = vcat(fill(1.0e-19, 10), fill(1.0e-16, 10))
    particles = [SVector(m) for m in masses]
    volume = 1.0e-6
    Λ = 0.0
    for i in 1:19, j in (i + 1):20

        Λ += kernel(particles[i], particles[j])
    end
    Λ /= volume
    dt = 10.0 / Λ                      # ~6-7 expected events (rates decay as N drops)
    proc = NonCNMCCoagulationProcess(kernel, LocalMajorant())
    events = Int[]
    for rep in 1:300
        u = make_u0(particles)         # fresh copy (particles untouched)
        sys = ParticleSystem(Val(1), 20, volume, t -> SVector(0.0))
        Random.seed!(20260821 + rep)
        push!(events, step_coagulation!(u, sys, proc, dt))
        @test sys.n_active == 20 - events[end]              # NonCNMC removes one per event
        m_now = StochParticles.total_mass(u, Val(1), sys.n_active)
        @test m_now ≈ StochParticles.total_mass(make_u0(particles), Val(1), 20)
    end
    # reference: exact pair-sum SSA (direct method) on the same ensemble
    ref = Int[]
    for rep in 1:300
        u = make_u0(particles)
        sys = ParticleSystem(Val(1), 20, volume, t -> SVector(0.0))
        Random.seed!(907070 + rep)
        t = 0.0
        c = 0
        while sys.n_active >= 2
            n = sys.n_active
            pairs = Tuple{Int, Int, Float64}[]
            tot = 0.0
            for i in 1:(n - 1), j in (i + 1):n

                K = kernel(get_particle(u, i, Val(1)), get_particle(u, j, Val(1)))
                if K > 0.0
                    push!(pairs, (i, j, K))
                    tot += K
                end
            end
            tot <= 0.0 && break
            t += randexp() / (tot / volume)
            t >= dt && break
            r = rand() * tot
            acc = 0.0
            for (i, j, K) in pairs
                acc += K
                if r <= acc
                    StochParticles.non_cnmc_coagulate!(u, sys, Val(1), i, j)
                    break
                end
            end
            c += 1
        end
        push!(ref, c)
    end
    # sampler mean must match the exact SSA mean within 4 standard errors
    se = sqrt(var(events) / 300 + var(ref) / 300)
    @test abs(mean(events) - mean(ref)) < 4 * se
    # edge: n_active < 2 is a no-op
    u2 = make_u0([SVector(1.0e-18)])
    sys2 = ParticleSystem(Val(1), 1, 1.0e-6, t -> SVector(0.0))
    @test step_coagulation!(u2, sys2, proc, 100.0) == 0
end

# --- solve_split ---

# Minimal custom drift for the degenerate-ODE equivalence test (1%/s mass growth)
struct ConstDrift <: StochParticles.PhysicsProcess end
StochParticles.provides_drift(::ConstDrift) = true
StochParticles.apply_drift(::ConstDrift, μ, sys, t) = μ .* 0.01

# Constant-kernel coagulation for the Smoluchowski analytic test (K0 = 1.0e-12)
struct ConstKernel <: StochParticles.CoagulationKernel{1} end
(::ConstKernel)(μ_i::SVector{1, Float64}, μ_j::SVector{1, Float64}) = 1.0e-12

@testset "solve_split degenerate ODE path" begin
    particles = [SVector(1.0e-18), SVector(2.0e-18), SVector(3.0e-18)]
    records = solve_split(particles, 1.0e-6, t -> SVector(0.0), (ConstDrift(),),
        Tsit5(); tspan = (0.0, 100.0), dt_split = 10.0,
        saveat = 10.0,
        record_func = (t, u,
            sys) -> (t = t,
            mass = StochParticles.total_mass(u, Val(1), sys.n_active)))[2]
    @test length(records) == 11                      # t = 0, 10, ..., 100
    @test records[1].t == 0.0
    @test records[end].t == 100.0
    # masses grow e^{0.01 t}: total mass ratio = e^1 within ODE tolerance
    @test records[end].mass / records[1].mass ≈ exp(1.0) rtol = 1.0e-3
end

@testset "solve_split contract errors" begin
    particles = [SVector(1.0e-18), SVector(2.0e-18)]
    kernel = BrownianKernel(293.15, 101325.0, SVector(1000.0))
    proc = NonCNMCCoagulationProcess(kernel, LocalMajorant())
    # saveat not an integer multiple of dt_split
    @test_throws ArgumentError solve_split(particles, 1.0e-6, t -> SVector(0.0),
        (ConstDrift(),), Tsit5();
        tspan = (0.0, 10.0), dt_split = 3.0,
        saveat = 10.0, record_func = (t, u, sys) -> t)
    # two coagulation processes
    @test_throws ArgumentError solve_split(particles, 1.0e-6, t -> SVector(0.0),
        (proc, proc), Tsit5();
        tspan = (0.0, 10.0), dt_split = 5.0,
        saveat = 10.0, record_func = (t, u, sys) -> t)
    # unsupported process kind (Emission; sampler never invoked, type is what matters)
    @test_throws ArgumentError solve_split(particles, 1.0e-6, t -> SVector(0.0),
        (EmissionProcess(1.0, (args...) -> SVector(0.0)),), Tsit5();
        tspan = (0.0, 10.0), dt_split = 5.0,
        saveat = 10.0, record_func = (t, u, sys) -> t)
end

@testset "solve_split Smoluchowski analytic (constant kernel)" begin
    n0 = 200
    volume = 1.0e-12
    particles = [SVector(1.0e-18) for _ in 1:n0]
    proc = NonCNMCCoagulationProcess(ConstKernel(), LocalMajorant())
    # N0 = n0/V = 2e14; K0*N0/2 = 100 /s -> t = 0.01 s gives N/N0 = 1/2
    finals = Float64[]
    for rep in 1:20
        Random.seed!(777000 + rep)
        _,
        recs = solve_split(particles, volume, t -> SVector(0.0), (proc,), Tsit5();
            tspan = (0.0, 0.01), dt_split = 0.002, saveat = 0.01,
            record_func = (t, u, sys) -> sys.n_active)
        push!(finals, recs[end] / n0)
    end
    # analytic N/N0 = 1/(1 + K0*N0*t/2) = 0.5; 20 reps -> 10% band
    @test abs(mean(finals) - 0.5) < 0.05
end
