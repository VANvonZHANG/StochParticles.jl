# src/core/cnmc.jl

"""
    cnmc_merge!(u, sys, i, j)

Merge particle j into particle i: μᵢ ← μᵢ + μʲ.
Then swap-delete particle j (move last active to slot j, decrement n_active).
"""
function cnmc_merge!(u::Vector{Float64}, sys::ParticleSystem{A}, i::Int, j::Int) where {A}
    A_val = Val(A)
    μ_i = get_particle(u, i, A_val)
    μ_j = get_particle(u, j, A_val)
    set_particle!(u, i, A_val, μ_i + μ_j)
    # Swap-delete: move last active particle to slot j
    if j < sys.n_active
        μ_last = get_particle(u, sys.n_active, A_val)
        set_particle!(u, j, A_val, μ_last)
    end
    sys.n_active -= 1
    nothing
end

"""
    cnmc_clone!(u, sys, target_slot, source_idx)

Copy particle source_idx into target_slot. Increment n_active.
Used after merge to restore n_sim count.
"""
function cnmc_clone!(u::Vector{Float64}, sys::ParticleSystem{A}, target_slot::Int, source_idx::Int) where {A}
    A_val = Val(A)
    μ_source = get_particle(u, source_idx, A_val)
    set_particle!(u, target_slot, A_val, μ_source)
    sys.n_active += 1
    nothing
end

"""
    cnmc_volume_rescale!(sys, μ_cloned)

Update computational volume to conserve mass concentration after cloning.
    V_new = V_old × (1 + |μ_cloned| / M_total)
"""
function cnmc_volume_rescale!(sys::ParticleSystem, μ_cloned::SVector{A, Float64}) where {A}
    m_cloned = sum(μ_cloned)
    sys.volume *= (1.0 + m_cloned / sys._mass_total_cache)
    nothing
end

"""
    cnmc_coagulate!(u, sys, ::Val{A}, i, j)

Full CNMC coagulation step:
1. Merge particles i and j (mass conserving, n_active decrements)
2. Clone a random particle into the vacated slot (n_active increments)
3. Rescale volume to conserve mass concentration (n_sim maintained)
"""
function cnmc_coagulate!(u::Vector{Float64}, sys::ParticleSystem{A}, ::Val{A}, i::Int, j::Int) where {A}
    A_val = Val(A)

    # Cache total mass before merge
    M_total = total_mass(u, A_val, sys.n_active)
    sys._mass_total_cache = M_total

    # Step 1: Merge j into i, swap-delete j
    cnmc_merge!(u, sys, i, j)
    vacated_slot = j

    # Step 2: Clone a random particle into vacated slot
    clone_source = rand(1:sys.n_active)
    cnmc_clone!(u, sys, vacated_slot, clone_source)

    # Step 3: Volume rescale
    μ_cloned = get_particle(u, vacated_slot, A_val)
    cnmc_volume_rescale!(sys, μ_cloned)

    nothing
end
