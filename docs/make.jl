# Build the package documentation (#1).
#
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
#   julia --project=docs docs/make.jl
#
# Output goes to docs/build/ (gitignored). The build fails on an exported name
# without a docstring (`checkdocs = :exports`) and on a failing doctest. Doctests
# use the offline Ergast fixture in test/fixtures; nothing here fits a model.

using Documenter, Diomedes

DocMeta.setdocmeta!(Diomedes, :DocTestSetup,
                    :(using Diomedes; FIXTURE = joinpath(pkgdir(Diomedes), "test", "fixtures", "ergast"));
                    recursive = true)

makedocs(; sitename = "Diomedes.jl", modules = [Diomedes], checkdocs = :exports, doctest = true,
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
         pages = ["Home" => "index.md",
                  "Getting started" => "getting_started.md",
                  "Data" => "data.md",
                  "Models" => ["Overview" => "models/overview.md",
                               "Z-score model" => "models/zscore.md",
                               "Gap and pace + loss models" => "models/gap.md",
                               "Retirement model" => "models/retirement.md",
                               "Lap-time model" => "models/laps.md"],
                  "Running long fits" => "fitting.md",
                  "Model comparison" => "comparison.md",
                  "Scripts" => "scripts.md",
                  "API reference" => ["Data sources and schemas" => "api/data.md",
                                           "Preparation and z-score model" => "api/zscore.md",
                                           "Gap and pace + loss models" => "api/gap.md",
                                           "Model comparison" => "api/comparison.md",
                                           "Retirement model" => "api/retirement.md",
                                           "Lap-time model" => "api/laps.md"]])
