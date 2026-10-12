# Build the package documentation (#1).
#
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
#   julia --project=docs docs/make.jl
#
# Output goes to docs/build/ (gitignored). The build fails on an exported name
# without a docstring (`checkdocs = :exports`) and on a failing doctest. Doctests
# use the offline Ergast fixture in test/fixtures; nothing here fits a model.
#
# The site (#45) adds Methodology and Findings sections. Their figures are Vega-Lite
# specs embedded in the pages (docs/src/assets/figures.js), drawing on posterior
# summaries committed under docs/src/assets/data/ (scripts/export_site.jl).
# Deployment is a separate step (docs/deploy.jl), so CI can check that the built
# site's figures render (docs/test/render_figures.js) before publishing it.

using Documenter, Diomedes

DocMeta.setdocmeta!(Diomedes, :DocTestSetup,
                    :(using Diomedes; FIXTURE = joinpath(pkgdir(Diomedes), "test", "fixtures", "ergast"));
                    recursive = true)

makedocs(; sitename = "Diomedes.jl", modules = [Diomedes], checkdocs = :exports, doctest = true,
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://aleccrowell.github.io/diomedes/stable/",
                                  size_threshold_ignore = ["findings/drivers.md"],
                                  # Vega itself is loaded by figures.js through RequireJS (see there)
                                  assets = ["assets/figures.js"]),
         pages = ["Home" => "index.md",
                  "Methodology" => ["Overview" => "methodology/index.md",
                                    "Identification" => "methodology/identification.md",
                                    "Data handling" => "methodology/data_handling.md",
                                    "Noise" => "methodology/noise.md",
                                    "Model progression" => "methodology/progression.md",
                                    "Retirements" => "methodology/retirements.md",
                                    "Lap-time model" => "methodology/laps.md",
                                    "Limitations" => "methodology/limitations.md"],
                  "Findings" => ["Overview" => "findings/index.md",
                                 "Drivers" => "findings/drivers.md",
                                 "Driver vs car" => "findings/explorer.md",
                                 "Cars" => "findings/cars.md",
                                 "Careers" => "findings/careers.md",
                                 "Eras" => "findings/eras.md",
                                 "Retirements" => "findings/retirements.md",
                                 "Lap times" => "findings/laps.md",
                                 "Rallying (WRC)" => "findings/wrc.md",
                                 "Data downloads" => "findings/data.md"],
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

