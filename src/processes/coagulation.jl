# src/processes/coagulation.jl

# ---- Kernel implementations ----

"""
    compute_air_properties(T, p) -> NamedTuple

Compute air properties from temperature [K] and pressure [Pa].

References:
- Air density: ideal gas law (Jacobson 2005, Eq. 4.54)
- Viscosity: Sutherland formula (Jacobson 2005, Eq. 4.48)
- Mean free path: kinetic theory derivation
"""
function compute_air_properties(T::Float64, p::Float64)
    M_air = 0.0289647       # kg/mol, molar mass of dry air
    R_univ = 8.314462618   # J/(mol*K), universal gas constant
    N_A = 6.02214076e23    # mol^-1, Avogadro's number
    kb = 1.380649e-23      # J/K, Boltzmann constant

    rho_air = p * M_air / (R_univ * T)
    mu_f = 1.8325e-5 * (416.16 / (T + 120.0)) * (T / 296.16)^1.5
    nu_f = mu_f / rho_air
    gasspeed = sqrt((8.0 * kb * T * N_A) / (pi * M_air))
    gasfreepath = 2.0 * nu_f / gasspeed

    return (rho_air=rho_air, mu_f=mu_f, nu_f=nu_f,
            gasspeed=gasspeed, gasfreepath=gasfreepath)
end

"""
    BrownianKernel{A} <: CoagulationKernel

Brownian diffusion coagulation kernel:
    K = (2 k_B T) / (3 μ_f) × (d_i + d_j)² / (d_i × d_j)

# Fields
- `T::Float64` — temperature [K]
- `p::Float64` — pressure [Pa]
- `rho_air::Float64` — air density [kg/m³]
- `mu_f::Float64` — dynamic viscosity of carrier fluid [Pa·s]
- `nu_f::Float64` — kinematic viscosity [m²/s]
- `gasspeed::Float64` — mean molecular speed [m/s]
- `gasfreepath::Float64` — mean free path [m]
- `densities::SVector{A, Float64}` — per-species densities [kg/m³]
- `kb::Float64` — Boltzmann constant
"""
struct BrownianKernel{A} <: CoagulationKernel{A}
    T::Float64
    p::Float64
    rho_air::Float64
    mu_f::Float64
    nu_f::Float64
    gasspeed::Float64
    gasfreepath::Float64
    densities::SVector{A, Float64}
    kb::Float64
end

BrownianKernel(T::Float64, p::Float64, densities::SVector{A, Float64}) where {A} =
    BrownianKernel{A}(T, p, compute_air_properties(T, p)..., densities, 1.380649e-23)

"""
    CompositeKernel{A, K1, K2} <: CoagulationKernel

Sum of two kernels: K(μ_i, μ_j) = K1(μ_i, μ_j) + K2(μ_i, μ_j).
"""
struct CompositeKernel{A, K1<:CoagulationKernel{A}, K2<:CoagulationKernel{A}} <: CoagulationKernel{A}
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

"""
    terminal_velocity_stokes(a, rho_p, rho_f, mu_f, g) -> Float64

Stokes terminal settling velocity for a spherical particle [m/s].
Formula: v_g = 2 ρ_p g a² (1 - ρ_f/ρ_p) / (9 μ_f)
"""
function terminal_velocity_stokes(a::Float64, rho_p::Float64, rho_f::Float64, mu_f::Float64, g::Float64)
    return 2.0 * rho_p * g * a^2 * (1.0 - rho_f / rho_p) / (9.0 * mu_f)
end

"""
    terminal_velocity_nld(a, rho_p, rho_f, mu_f, g, nu) -> Float64

Terminal velocity with Nonlinear Drag (NLD) correction (Ayala 2008 Part 1, Eq. 3-4).
For a < 30 μm uses Stokes velocity; for a ≥ 30 μm applies NLD factor.
"""
function terminal_velocity_nld(a::Float64, rho_p::Float64, rho_f::Float64, mu_f::Float64, g::Float64, nu::Float64)
    v_g = terminal_velocity_stokes(a, rho_p, rho_f, mu_f, g)
    if a < 30.0e-6
        return v_g
    end
    Re_p0 = 2.0 * a * abs(v_g) / nu
    f_Re = 1.0 + 0.15 * Re_p0^0.687
    return v_g / f_Re
end

"""
    GravitationalKernel{A} <: CoagulationKernel

Geometric collision kernel due to differential gravitational settling.

K_grav = π (a_i + a_j)² |v_i - v_j|

# Fields
- `mu_f::Float64` — dynamic viscosity [Pa·s]
- `rho_f::Float64` — air density [kg/m³]
- `rho_p::Float64` — particle density [kg/m³]
- `g::Float64` — gravity [m/s²]
- `densities::SVector{A, Float64}` — per-species densities [kg/m³]
"""
struct GravitationalKernel{A} <: CoagulationKernel{A}
    mu_f::Float64
    rho_f::Float64
    rho_p::Float64
    g::Float64
    densities::SVector{A, Float64}
end

CompositeKernel(k1::GravitationalKernel{A}, k2::GravitationalKernel{A}) where {A} =
    CompositeKernel{A, GravitationalKernel{A}, GravitationalKernel{A}}(k1, k2)

function (kernel::GravitationalKernel{A})(μ_i::SVector{A, Float64}, μ_j::SVector{A, Float64}) where {A}
    a_i = particle_diameter(μ_i, kernel.densities) / 2.0
    a_j = particle_diameter(μ_j, kernel.densities) / 2.0
    if a_i <= 0.0 || a_j <= 0.0
        return 0.0
    end
    nu = kernel.mu_f / kernel.rho_f
    v_i = terminal_velocity_nld(a_i, kernel.rho_p, kernel.rho_f, kernel.mu_f, kernel.g, nu)
    v_j = terminal_velocity_nld(a_j, kernel.rho_p, kernel.rho_f, kernel.mu_f, kernel.g, nu)
    R = a_i + a_j
    return π * R^2 * abs(v_i - v_j)
end

# ---- Ayala turbulent kernel helpers ----

"""
    particle_relaxation_time(a, rho_p, rho_f, mu_f) -> Float64

Particle relaxation time τ_p [s] from Stokes drag.
"""
function particle_relaxation_time(a::Float64, rho_p::Float64, rho_f::Float64, mu_f::Float64)
    return 2.0 * rho_p * a^2 * (1.0 - rho_f / rho_p) / (9.0 * mu_f)
end

"""
    nonlinear_drag_factor(Re_p0) -> Float64

Nonlinear drag correction factor f_Re (Ayala 2008 Part 1, Eq. 3).
"""
function nonlinear_drag_factor(Re_p0::Float64)
    return 1.0 + 0.15 * Re_p0^0.687
end

"""
    AyalaFlowParams

Pre-computed turbulence flow scales for Ayala 2008 parameterization.
All fields are Float64 in SI units.
"""
struct AyalaFlowParams
    tau_k::Float64    # Kolmogorov time scale [s]
    eta::Float64      # Kolmogorov length scale [m]
    v_k::Float64      # Kolmogorov velocity scale [m/s]
    u_prime::Float64  # RMS velocity fluctuation [m/s]
    T_L::Float64      # Lagrangian integral time scale [s]
    L_e::Float64      # Eulerian integral length scale [m]
    lambda::Float64   # Taylor microscale [m]
    tau_T::Float64    # Turbulence time scale [s]
    a_0::Float64      # Low-Reynolds parameter
    R_lambda::Float64 # Taylor microscale Reynolds number (passed through)
    g::Float64        # gravity [m/s²] (passed through)
end

"""
    compute_flow_params(epsilon, R_lambda, nu, g) -> AyalaFlowParams

Compute turbulence scales following Ayala 2008 Part 2, Table 2.
"""
function compute_flow_params(epsilon::Float64, R_lambda::Float64, nu::Float64, g::Float64)
    tau_k = sqrt(nu / epsilon)
    eta = (nu^3 / epsilon)^(1.0 / 4.0)
    v_k = (nu * epsilon)^(1.0 / 4.0)
    u_prime = sqrt(R_lambda) * v_k / 15.0^(1.0 / 4.0)
    T_L = u_prime^2 / epsilon
    L_e = 0.5 * u_prime^3 / epsilon
    lambda = u_prime * sqrt(15.0 * nu / epsilon)
    a_0 = (11.0 + 7.0 * R_lambda) / (205.0 + R_lambda)
    tau_T = sqrt(2.0 * R_lambda / (sqrt(15.0) * a_0)) * tau_k
    return AyalaFlowParams(tau_k, eta, v_k, u_prime, T_L, L_e, lambda, tau_T, a_0, R_lambda, g)
end

"""
    longitudinal_correlation_f2(R, Le, a_0) -> Float64

Longitudinal velocity correlation f₂(R) (Ayala 2008 Part 2, Eq. 63).
"""
function longitudinal_correlation_f2(R::Float64, Le::Float64, a_0::Float64)
    R_over_L = R / Le
    return 1.0 - a_0 * R_over_L^2 + (a_0 / 2.0) * R_over_L^4
end

"""
    compute_variance_sigma2(tau_p1, tau_p2, v_p1, v_p2, R, fp::AyalaFlowParams) -> Float64

Compute variance σ² of radial relative velocity (Ayala 2008 Part 2, Eq. 69-78).

Reference: Ayala et al. (2008) Part 2, New Journal of Physics 10, 075016.
"""
function compute_variance_sigma2(tau_p1::Float64, tau_p2::Float64, v_p1::Float64, v_p2::Float64,
                                  R::Float64, fp::AyalaFlowParams)
    # Stokes numbers
    St_1 = tau_p1 / fp.tau_k
    St_2 = tau_p2 / fp.tau_k

    # Correlation parameter
    f2 = longitudinal_correlation_f2(R, fp.L_e, fp.a_0)

    # Fluid-phase contribution (shear + acceleration)
    sigma2_fluid = (2.0 / 3.0) * fp.u_prime^2 * (1.0 - f2)

    # Particle-phase correction for finite inertia
    beta_st = sqrt(St_1 * St_2) / (1.0 + sqrt(St_1 * St_2))
    sigma2_particle = beta_st * sigma2_fluid

    # Gravity-shear coupling (cross term)
    v_diff = abs(v_p1 - v_p2)
    sigma2_grav = v_diff^2 / 3.0

    # Total variance
    return sigma2_fluid + sigma2_particle + sigma2_grav
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

"""
    compute_rdf_g12(St_1, St_2, R, eta, fp::AyalaFlowParams) -> Float64

Radial distribution function g₁₂ (Ayala 2008 Part 2, Eq. 87-90).
"""
function compute_rdf_g12(St_1::Float64, St_2::Float64, R::Float64, eta::Float64, fp::AyalaFlowParams)
    St = max(St_1, St_2)

    # y_St polynomial fit (Part 2, p.38)
    y_St = -0.1988 * St^4 + 1.5275 * St^3 - 4.2942 * St^2 + 5.3406 * St
    if y_St < 0.0
        y_St = 0.0
    end

    # f3_R and C1 (Eq. 88-89)
    f3_R = 0.1886 * exp(20.306 / fp.R_lambda)
    g_ratio = fp.g / (fp.v_k / fp.tau_k)
    C1 = y_St / g_ratio^f3_R

    # rc computation (Eq. 90)
    a_0g = fp.a_0 + (π / 8.0) * g_ratio^2
    F_val = 20.115 * sqrt(a_0g / fp.R_lambda)
    rc_over_eta_sq = abs(St_2 - St_1) * F_val
    rc = eta * sqrt(rc_over_eta_sq)

    # g12 (Eq. 87)
    R_sq = R^2
    return ((eta^2 + rc^2) / (R_sq + rc^2))^(C1 / 2.0)
end

"""
    f_ayala(b) -> Float64

Auxiliary function for folded-normal mean (Ayala 2008 Part 2, Eq. 80).
f(b) = 0.5 √π (b + 0.5/b) erf(b) + 0.5 exp(-b²)
"""
function f_ayala(b::Float64)
    if b < 1e-8
        # Series expansion for small b: f(b) ≈ 1 + b²/3 + O(b⁴)
        return 1.0 + b^2 / 3.0
    end
    return 0.5 * sqrt(π) * (b + 0.5 / b) * erf(b) + 0.5 * exp(-b^2)
end

"""
    AyalaTurbulentKernel{A} <: CoagulationKernel

Turbulence-enhanced geometric collision kernel following Ayala et al. (2008).

Γ = 2π R² ⟨|w_r|⟩ g₁₂

# Fields
- `epsilon::Float64` — turbulent dissipation rate [m²/s³]
- `R_lambda::Float64` — Taylor microscale Reynolds number
- `nu::Float64` — kinematic viscosity [m²/s]
- `rho_f::Float64` — air density [kg/m³]
- `rho_p::Float64` — particle (water) density [kg/m³]
- `g::Float64` — gravity [m/s²]
- `densities::SVector{A, Float64}` — per-species densities [kg/m³]
"""
struct AyalaTurbulentKernel{A} <: CoagulationKernel{A}
    epsilon::Float64
    R_lambda::Float64
    nu::Float64
    rho_f::Float64
    rho_p::Float64
    g::Float64
    densities::SVector{A, Float64}

    function AyalaTurbulentKernel{A}(epsilon, R_lambda, nu, rho_f, rho_p, g, densities::SVector{A, Float64}) where {A}
        if epsilon <= 0.0
            throw(DomainError(epsilon, "epsilon must be positive"))
        end
        new{A}(epsilon, R_lambda, nu, rho_f, rho_p, g, densities)
    end
end

AyalaTurbulentKernel(epsilon, R_lambda, nu, rho_f, rho_p, g, densities::SVector{A, Float64}) where {A} =
    AyalaTurbulentKernel{A}(epsilon, R_lambda, nu, rho_f, rho_p, g, densities)

CompositeKernel(k1::AyalaTurbulentKernel{A}, k2::AyalaTurbulentKernel{A}) where {A} =
    CompositeKernel{A, AyalaTurbulentKernel{A}, AyalaTurbulentKernel{A}}(k1, k2)

function (kernel::AyalaTurbulentKernel{A})(μ_i::SVector{A, Float64}, μ_j::SVector{A, Float64}) where {A}
    # Error handling
    if kernel.epsilon <= 0.0
        throw(DomainError(kernel.epsilon, "epsilon must be positive"))
    end
    if kernel.R_lambda < 1.0
        @warn "R_lambda < 1 is outside DNS validation range" R_lambda=kernel.R_lambda
    end

    a_1 = particle_diameter(μ_i, kernel.densities) / 2.0
    a_2 = particle_diameter(μ_j, kernel.densities) / 2.0
    if a_1 <= 0.0 || a_2 <= 0.0
        return 0.0
    end

    R = a_1 + a_2

    # Step 1: Particle parameters (with NLD)
    mu_f = kernel.nu * kernel.rho_f
    tau_p1_stokes = particle_relaxation_time(a_1, kernel.rho_p, kernel.rho_f, mu_f)
    tau_p2_stokes = particle_relaxation_time(a_2, kernel.rho_p, kernel.rho_f, mu_f)

    v_g1 = terminal_velocity_stokes(a_1, kernel.rho_p, kernel.rho_f, mu_f, kernel.g)
    v_g2 = terminal_velocity_stokes(a_2, kernel.rho_p, kernel.rho_f, mu_f, kernel.g)

    Re_p0_1 = 2.0 * a_1 * abs(v_g1) / kernel.nu
    Re_p0_2 = 2.0 * a_2 * abs(v_g2) / kernel.nu
    f_Re1 = nonlinear_drag_factor(Re_p0_1)
    f_Re2 = nonlinear_drag_factor(Re_p0_2)

    tau_p1 = tau_p1_stokes / f_Re1
    tau_p2 = tau_p2_stokes / f_Re2
    v_p1 = v_g1 / f_Re1
    v_p2 = v_g2 / f_Re2

    # Step 2: Turbulence scales
    fp = compute_flow_params(kernel.epsilon, kernel.R_lambda, kernel.nu, kernel.g)

    # Step 3: Radial relative velocity
    St_1 = tau_p1 / fp.tau_k
    St_2 = tau_p2 / fp.tau_k

    if max(St_1, St_2) > 10.0
        @warn "St > 10 is outside weak-inertia assumption" St_1=St_1 St_2=St_2
    end

    sigma2 = compute_variance_sigma2(tau_p1, tau_p2, v_p1, v_p2, R, fp)
    if sigma2 <= 0.0
        # Fallback: pure gravitational settling (no turbulence)
        return π * R^2 * abs(v_p1 - v_p2)
    end

    b = kernel.g * abs(tau_p1 - tau_p2) / (sqrt(sigma2) * sqrt(2.0))
    wr = sqrt(2.0 / π) * sqrt(sigma2) * f_ayala(b)

    # Step 4: RDF
    g12 = compute_rdf_g12(St_1, St_2, R, fp.eta, fp)

    # Step 5: Return kernel value
    return 2.0 * π * R^2 * wr * g12
end

# ---- Atmospheric parameters and kernel factory ----

"""
    AtmosphericParameters

Container for consistent fluid and environmental parameters used
across all coagulation kernels.

# Fields
- `T::Float64` — temperature [K]
- `p::Float64` — pressure [Pa]
- `rho_f::Float64` — air density [kg/m³]
- `mu_f::Float64` — dynamic viscosity [Pa·s]
- `nu::Float64` — kinematic viscosity [m²/s]
- `rho_p::Float64` — particle density [kg/m³]
- `g::Float64` — gravity [m/s²]
"""
struct AtmosphericParameters
    T::Float64
    p::Float64
    rho_f::Float64
    mu_f::Float64
    nu::Float64
    rho_p::Float64
    g::Float64
end

"""
    AtmosphericParameters(T, p; rho_f=1.225, mu_f=1.81e-5, nu=1.48e-5, rho_p=1000.0, g=9.81)

Convenience constructor with standard atmospheric defaults.
"""
function AtmosphericParameters(T::Float64, p::Float64;
                                rho_f::Float64=1.225,
                                mu_f::Float64=1.81e-5,
                                nu::Float64=1.48e-5,
                                rho_p::Float64=1000.0,
                                g::Float64=9.81)
    return AtmosphericParameters(T, p, rho_f, mu_f, nu, rho_p, g)
end

"""
    make_kernel(params, epsilon, R_lambda, densities) -> CompositeKernel

Create a composite kernel combining Brownian, gravitational, and turbulent
coagulation with consistent physical parameters.

Returns: `CompositeKernel(K_brown, CompositeKernel(K_grav, K_turb))`
"""
function make_kernel(params::AtmosphericParameters, epsilon::Float64, R_lambda::Float64,
                      densities::SVector{A, Float64}) where {A}
    K_brown = BrownianKernel(params.T, params.mu_f, densities)
    K_grav = GravitationalKernel(params.mu_f, params.rho_f, params.rho_p, params.g, densities)
    K_turb = AyalaTurbulentKernel(epsilon, R_lambda, params.nu, params.rho_f, params.rho_p, params.g, densities)
    return CompositeKernel(K_brown, CompositeKernel(K_grav, K_turb))
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
struct CoagulationProcess{K<:CoagulationKernel{<:Any}, S<:CoagulationSampling} <: PhysicsProcess
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
