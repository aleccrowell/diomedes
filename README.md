# diomedes

[![CI](https://github.com/aleccrowell/diomedes/actions/workflows/CI.yml/badge.svg)](https://github.com/aleccrowell/diomedes/actions/workflows/CI.yml)
[![Slow tests](https://github.com/aleccrowell/diomedes/actions/workflows/Slow.yml/badge.svg)](https://github.com/aleccrowell/diomedes/actions/workflows/Slow.yml)

Bayesian models of driver vs machine performance in motorsport, in Julia with
[Turing.jl](https://turinglang.org).

The question: how much of a result is the driver and how much is the car? Every
model has crossed random effects for drivers and car-seasons, fitted to public
timing data: Formula 1 from 1950 (Jolpica / Ergast) and the WRC and ERC rally
championships.

| Model | Outcome | Fit |
|---|---|---|
| Pace + loss (main model) | % gap to the winner, lapped cars as intervals; Gaussian pace noise plus incident losses; optional era, career, big-loss and in-season development terms | `fit_paceloss` |
| Gap model | the same outcome with Student-t noise | `fit_gaps` |
| Retirement | competing-risks hazards: car reliability, driver incidents, collisions | `fit_retirement` |
| Lap time (F1, 1996–) | lap time, % of the race's median clean lap (in development) | `fit_laps` |
| Z-score (legacy port) | stage time standardised within stage | `fit_effects` |

## Documentation

The manual and API reference are built with Documenter.jl:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=docs docs/make.jl                         # then open docs/build/index.html
```

It covers getting started, the data sources and schema, one page per model
(outcome, likelihood, identification, options), running long fits
(checkpoints, parallel chains, warm starts), model comparison, and the scripts.

## Quick start

```julia
julia> ]activate .
julia> ]instantiate
julia> using Diomedes

julia> res = fetch_results(JolpicaF1(), 1950:2024)       # or ErgastCSV("dir"), WRCTiming()
julia> g = prepare_gaps(res)                               # % gaps, lapped cars as intervals
julia> chain = fit_paceloss(g; loss_duration = true, era = :rw, pace_scale = true, n_chains = 4)
julia> identified_convergence(chain, g, "pl_dur_rw_kappa")
julia> gap_effects_table(chain, g).competitors             # negative = faster, % of race time
```

API responses are cached under `data/cache/` (override with `DIOMEDES_CACHE`).
Full-history fits take from about an hour to most of a day on a Raspberry Pi 5;
the scripts in `scripts/` run them as one process per chain with resumable
checkpoints (see the manual's "Running long fits").

## Layout

```
src/
  Diomedes.jl        module entry, exports
  schema.jl          results / lap / pit-stop schemas
  cache.jl           on-disk cache for API responses + rate limiting
  sources/           adapters: public data -> schema
    jolpica.jl         F1, 1950-present (Jolpica API, the Ergast successor)
    ergast_csv.jl      F1 from a local Ergast CSV dump (+ lap times, pit stops)
    wrc.jl             WRC 2018- / ERC 2022- stage times (wrc.com timing API)
    suspensions.jl     red-flag suspension detection from lap times
  prepare.jl         z-scores per stage -> ModelData
  models.jl          z-score model; samplers, checkpointed NUTS, progress, diagnostics
  noise.jl           noise families for the gap model
  gap_model.jl       % gap to winner, lapped cars as intervals -> GapData; Student-t model
  paceloss.jl        pace + incident-loss model and its optional terms
  likelihoods.jl     fused likelihoods with hand-written reverse-mode rules
  loo.jl             per-row log-likelihoods (PSIS-LOO), model names, convergence
  kfold.jl           grouped K-fold cross-validation
  retirement.jl      competing-risks retirement model
  partial.jl         partial-distance gaps of retirements (selection check)
  laps.jl            lap-time model
scripts/             full-data fits and analyses (arguments in each header)
docs/                Documenter manual and API reference
test/                offline tests (fixtures in test/fixtures)
legacy/              original TensorFlow Probability implementation (reference only)
```

Adding a series means writing one `DataSource` adapter that returns
`RESULT_SCHEMA` rows; nothing downstream is series-specific (manual: "Data").

## Tests

```
julia --project test/runtests.jl                              # fast, offline
DIOMEDES_SLOW_TESTS=1 julia --project test/runtests.jl        # + NUTS model fit
DIOMEDES_NETWORK_TESTS=1 julia --project test/runtests.jl     # + live APIs
```

Running `test/runtests.jl` directly rather than `Pkg.test()` reuses the normal
precompile cache; `Pkg.test()` forces `--check-bounds=yes`, which needs a
separate, full recompile of the dependency tree. `Pkg.test()` still works (CI).

GitHub Actions runs the fast suite on every push (`CI.yml`), and the slow and
network groups weekly and on demand (`Slow.yml`).

## Releases

To release, bump `version` in `Project.toml` in a PR. When CI passes on the
merge commit to master, `Release.yml` tags it `vX.Y.Z` and creates a GitHub
Release with notes generated from the merged PRs. A version that is already
tagged is skipped, so other merges don't release.

## Data sources

- F1: [Jolpica F1 API](https://github.com/jolpica/jolpica-f1)
- WRC/ERC: the undocumented timing API behind wrc.com; endpoints as used by
  [OpenWRC](https://github.com/jixy2012/OpenWRC) and described by
  [rallydatajunkie](https://rallydatajunkie.com/visualising-wrc-rally-results/accessing-data-from-the-wrc-live-timing-api.html).
  It is not a public contract and may change.
