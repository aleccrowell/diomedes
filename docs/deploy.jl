# Publish the site built by docs/make.jl to the gh-pages branch (GitHub Pages).
#
#   julia --project=docs docs/deploy.jl
#
# Run by CI after the build and the figure check (docs/test/render_figures.js).
# Pushes to master update /dev; a tag vX.Y.Z publishes /stable and /vX.Y.Z.
# Outside CI, deploydocs detects no deploy environment and does nothing.

using Documenter

deploydocs(; repo = "github.com/aleccrowell/diomedes.git", devbranch = "master", push_preview = false)
