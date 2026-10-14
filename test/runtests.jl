# test/runtests.jl
using StochParticles
using Test
using Aqua

# Shared test helper: minimal mock integrator for testing jump affect! functions
struct MockIntegrator
    u::Vector{Float64}
    p::ParticleSystem
    t::Float64
end

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(StochParticles; persistent_tasks = false)
end

# SP_TEST_TIER: "fast" (default, local dev) | "full" (CI PR job + nightly).
# SLOW = real simulations (coagulation suite, parcel loops, campaigns);
# FAST = unit/AD/io/diagnostics. See plans/2026-10-14-test-tiers.md.
const TIER = lowercase(get(ENV, "SP_TEST_TIER", "fast"))
TIER in ("fast", "full") || error("SP_TEST_TIER must be fast|full, got $TIER")
const FAST_FILES = ["test_particle_system.jl", "test_process_trait.jl",
    "test_preset_species.jl", "test_condensation.jl", "test_kohler_critical_point.jl",
    "test_cnmc.jl", "test_emission.jl", "test_dilution.jl", "test_diagnostics.jl",
    "test_io.jl", "test_env_source.jl", "parcel_test.jl",
    "activation_diagnostics_test.jl", "ad_flux_test.jl", "ad_rhs_test.jl"]
const SLOW_FILES = ["test_coagulation.jl", "test_local_majorant.jl",
    "test_integration.jl", "test_mixing_state.jl", "test_gcvi.jl",
    "test_synthetic_population.jl", "test_parcel_coupling.jl",
    "test_activation_comparison_example_config.jl", "test_plotting.jl",
    "test_brownian_precision.jl"]

@testset "StochParticles.jl" begin
    for f in (TIER == "full" ? vcat(FAST_FILES, SLOW_FILES) : FAST_FILES)
        include(f)
    end
end
