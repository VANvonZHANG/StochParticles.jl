# src/diagnostics/reconstruction.jl

"""
    reconstruct_volumes(sol, prob) -> Vector{Float64}

Reconstruct computational volumes at each saved time point using mass conservation.

V(t) = V₀ × M(t) / M₀

where M₀ is the total mass at t=0 and M(t) is the total mass at time t.
"""
function reconstruct_volumes(sol, prob)
    sys = prob.prob.p
    A = species_val(sys)
    u0 = sol.u[1]
    M_total_0 = total_mass(u0, A, sys.n_active)
    n = length(sol.t)
    volumes = Vector{Float64}(undef, n)
    for i in 1:n
        u = sol.u[i]
        M_total_t = total_mass(u, A, sys.n_active)
        volumes[i] = M_total_t * sys.volume / M_total_0
    end
    return volumes
end

"""
    extract_concentrations(sol, prob) -> (t, N_conc, M_conc)

Extract time-series diagnostics from a SciML solution.

Returns:
- `t`: time points
- `N_conc`: number concentration [particles/m³] at each time
- `M_conc`: mass concentration [kg/m³] at each time

Uses volume reconstruction to account for CNMC volume rescaling.
"""
function extract_concentrations(sol, prob)
    sys = prob.prob.p
    A = species_val(sys)
    t = sol.t
    n = length(t)
    volumes = reconstruct_volumes(sol, prob)
    N_conc = Vector{Float64}(undef, n)
    M_conc = Vector{Float64}(undef, n)
    for i in 1:n
        u = sol.u[i]
        M_total_t = total_mass(u, A, sys.n_active)
        N_conc[i] = sys.n_sim / volumes[i]
        M_conc[i] = M_total_t / volumes[i]
    end
    return t, N_conc, M_conc
end
