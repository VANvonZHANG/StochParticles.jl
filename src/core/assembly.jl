# src/core/assembly.jl

"""
    apply_all_drifts!(dμ, μ, sys, t, processes::Tuple)

Apply drift from all drift-providing processes. Type-stable because tuple
element types are known at compile time (concrete NTuple).
"""
function apply_all_drifts!(dμ, μ, sys, t, processes::Tuple)
    for proc in processes
        if provides_drift(proc)
            dμ = dμ + apply_drift(proc, μ, sys, t)
        end
    end
    return dμ
end

"""
    make_ode_func(processes::Tuple)

Create the ODE right-hand side function for SciML's ODEProblem.
Iterates over active particles, applies drift from all drift-providing processes.
The returned closure captures the concrete process tuple type for full specialization.
"""
function make_ode_func(processes::Tuple)
    return function (du, u, p, t)
        A = species_val(p)
        n = p.n_active
        for i in 1:n
            μ = get_particle(u, i, A)
            dμ = apply_all_drifts!(zero(μ), μ, p, t, processes)
            set_particle!(du, i, A, dμ)
        end
        # Zero out inactive particle slots
        if n < p.n_sim
            zero_μ = zero(get_particle(u, 1, A))
            for i in (n + 1):p.n_sim
                set_particle!(du, i, A, zero_μ)
            end
        end
        nothing
    end
end
