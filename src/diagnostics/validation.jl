# src/diagnostics/validation.jl

"""
    check_mass_conservation(sol, prob; tolerance=1e-3) -> (passed::Bool, rel_error::Float64)

Check that mass concentration is conserved over the simulation.

For coagulation-only simulations, mass should be exactly conserved.
For simulations with emission/dilution, mass conservation is not expected.

Returns a tuple of (whether the check passed, the relative error).
"""
function check_mass_conservation(sol, prob; tolerance::Float64 = 1e-3)
    t, N_conc, M_conc = extract_concentrations(sol, prob)
    M0 = M_conc[1]
    max_rel_error = 0.0
    for i in 1:length(M_conc)
        rel_error = abs(M_conc[i] - M0) / M0
        if rel_error > max_rel_error
            max_rel_error = rel_error
        end
    end
    return (max_rel_error < tolerance, max_rel_error)
end
