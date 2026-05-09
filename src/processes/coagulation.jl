# src/processes/coagulation.jl

# ---- Kernel implementations ----

"""
    BrownianKernel{A} <: CoagulationKernel

Brownian diffusion coagulation kernel:
    K = (2 k_B T) / (3 μ_f) × (d_i + d_j)² / (d_i × d_j)

# Fields
- `T::Float64` — temperature [K]
- `mu_f::Float64` — dynamic viscosity of carrier fluid [Pa·s]
- `densities::SVector{A, Float64}` — per-species densities [kg/m³]
- `kb::Float64` — Boltzmann constant
"""
struct BrownianKernel{A} <: CoagulationKernel
    T::Float64
    mu_f::Float64
    densities::SVector{A, Float64}
    kb::Float64
end

BrownianKernel(T::Float64, mu_f::Float64, densities::SVector{A, Float64}) where {A} =
    BrownianKernel{A}(T, mu_f, densities, 1.380649e-23)

"""
    CompositeKernel{A, K1, K2} <: CoagulationKernel

Sum of two kernels: K(μ_i, μ_j) = K1(μ_i, μ_j) + K2(μ_i, μ_j).
"""
struct CompositeKernel{A, K1<:CoagulationKernel, K2<:CoagulationKernel} <: CoagulationKernel
    k1::K1
    k2::K2
end

CompositeKernel(k1::BrownianKernel{A}, k2::BrownianKernel{A}) where {A} =
    CompositeKernel{A, BrownianKernel{A}, BrownianKernel{A}}(k1, k2)

# ---- Kernel evaluation ----

function particle_diameter(μ::SVector{A, Float64}, densities::SVector{A, Float64}) where {A}
    V_p = sum(μ[k] / densities[k] for k in 1:A)
    return (6.0 * V_p / π)^(1.0 / 3.0)
end

function (kernel::BrownianKernel{A})(μ_i::SVector{A, Float64}, μ_j::SVector{A, Float64}) where {A}
    d_i = particle_diameter(μ_i, kernel.densities)
    d_j = particle_diameter(μ_j, kernel.densities)
    coeff = 2.0 * kernel.kb * kernel.T / (3.0 * kernel.mu_f)
    return coeff * (d_i + d_j)^2 / (d_i * d_j)
end

function (kernel::CompositeKernel{A})(μ_i::SVector{A, Float64}, μ_j::SVector{A, Float64}) where {A}
    return kernel.k1(μ_i, μ_j) + kernel.k2(μ_i, μ_j)
end

# ---- Sampling strategy implementations ----

"""
    GlobalMajorant <: CoagulationSampling

Use a single global upper bound K_max over all particle pairs.
Simple but can have low acceptance rate for broad size distributions.
"""
struct GlobalMajorant <: CoagulationSampling end

"""
    compute_majorant(sampling, kernel, u, sys) -> Float64

Compute the majorant kernel value K_max ≥ K(μ_i, μ_j) for all active pairs.
"""
function compute_majorant(sampling::GlobalMajorant, kernel, u::Vector{Float64}, sys::ParticleSystem{A}) where {A}
    K_max = 0.0
    for i in 1:sys.n_active
        μ_i = get_particle(u, i, Val(A))
        for j in (i + 1):sys.n_active
            μ_j = get_particle(u, j, Val(A))
            K_max = max(K_max, kernel(μ_i, μ_j))
        end
    end
    return K_max
end

"""
    majorant_rate(sampling, kernel, u, sys) -> Float64

Compute the total coagulation event rate using the majorant method:
    Λ = (K_max / V) × N × (N-1) / 2
"""
function majorant_rate(sampling, kernel, u, sys)
    K_max = compute_majorant(sampling, kernel, u, sys)
    N = sys.n_active
    return K_max / sys.volume * N * (N - 1) / 2
end

# ---- Coagulation Process ----

"""
    CoagulationProcess{K, S} <: PhysicsProcess

Stochastic coagulation process that merges particle pairs via Majorant/Null-event sampling.

# Fields
- `kernel::K` — coagulation rate kernel (e.g. `BrownianKernel`)
- `sampling::S` — pair-selection strategy (e.g. `GlobalMajorant`)
"""
struct CoagulationProcess{K<:CoagulationKernel, S<:CoagulationSampling} <: PhysicsProcess
    kernel::K
    sampling::S
end

provides_drift(::CoagulationProcess) = false

"""
    make_coagulation_jump(kernel, sampling) -> ConstantRateJump

Create a SciML ConstantRateJump for the coagulation process using
the Majorant/Null-event method.
"""
function make_coagulation_jump(kernel, sampling)
    rate = (u, p, t) -> begin
        if p.n_active < 2
            return 0.0
        end
        K_max = compute_majorant(sampling, kernel, u, p)
        p._cached_majorant = K_max
        return K_max / p.volume * p.n_active * (p.n_active - 1) / 2
    end

    affect! = (integrator) -> begin
        u = integrator.u
        p = integrator.p
        N = p.n_active
        if N < 2
            return nothing
        end

        # Select random pair
        i = rand(1:N)
        j = rand(1:(N-1))
        j = j >= i ? j + 1 : j

        A_val = species_val(p)
        μ_i = get_particle(u, i, A_val)
        μ_j = get_particle(u, j, A_val)

        # Accept/reject using cached K_max from rate evaluation
        K_actual = kernel(μ_i, μ_j)
        K_max = p._cached_majorant
        if K_max > 0 && rand() < K_actual / K_max
            cnmc_coagulate!(u, p, A_val, i, j)
        end
        nothing
    end

    return ConstantRateJump(rate, affect!)
end
