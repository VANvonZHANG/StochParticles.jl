using Documenter, StochParticles

DocMeta.setdocmeta!(StochParticles, :DocTestSetup, :(using StochParticles); recursive = true)

makedocs(;
    modules = [StochParticles],
    checkdocs = :none,
    authors = "Fan Zhang",
    sitename = "StochParticles.jl",
    remotes = nothing,
    format = Documenter.HTML(;
        canonical = "https://VANvonZHANG.github.io/StochParticles.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API Reference" => [
            "Core Types" => "api/core.md",
            "Physics Processes" => "api/processes.md",
            "Coagulation Kernels" => "api/kernels.md",
            "CNMC Operations" => "api/cnmc.md",
            "Diagnostics" => "api/diagnostics.md"
        ]
    ]
)

deploydocs(;
    repo = "github.com/VANvonZHANG/StochParticles.jl",
    devbranch = "main",
    push_preview = true
)
