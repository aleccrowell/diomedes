# diomedes

Bayesian models of driver vs machine performance in motorsport, in Julia with
[Turing.jl](https://turinglang.org).

The question: how much of a result is the driver and how much is the car? Each
timed stage (an F1 race, a WRC special stage) gives a standardised time per
competitor, modelled as

    z = α + driver effect + machine-season effect + noise

with crossed random effects for drivers and car-seasons.

## Layout

```
src/
  Diomedes.jl        module entry
  schema.jl          common long-format results schema (one row per competitor per stage)
  cache.jl           on-disk cache for API responses + rate limiting
  sources/           adapters: public data -> schema
    jolpica.jl         F1, 1950-present (Jolpica API, the Ergast successor)
    ergast_csv.jl      F1 from a local Ergast CSV dump
    wrc.jl             WRC 2018-present / ERC 2022-present stage times (wrc.com timing API)
  prepare.jl         standardise times per stage, index levels -> ModelData
  models.jl          z-score model (legacy port), fitting, progress logging, diagnostics
  gap_model.jl       % gap to winner, lapped cars as intervals -> GapData; Student-t model
  paceloss.jl        pace + incident-loss noise model with race-duration / era terms
  likelihoods.jl     fused likelihoods with hand-written reverse-mode rules
  loo.jl             per-row log-likelihoods for PSIS-LOO
scripts/
  legacy_parity.jl   check against the original Python pipeline
  gap_fit.jl         fit a gap model (4 chains, as parallel processes: `chain k` + `combine`)
  loo_compare.jl     PSIS-LOO comparison of fitted gap models
  incident_diagnostic.jl, era_paths.jl   per-season incident rate / loss size
test/                offline tests (fixtures in test/fixtures)
legacy/              original TensorFlow Probability implementation (reference only)
```

## Usage

```julia
julia> ]activate .
julia> ]instantiate
julia> using Diomedes

julia> res = fetch_results(JolpicaF1(), 2010:2024)      # or WRCTiming(), ErgastCSV("dir")
julia> d = prepare(res)                                  # ModelData
julia> chain = fit_effects(d; n_chains = 4)              # NUTS; σ_y = nothing to learn the noise scale
julia> convergence_summary(chain)                        # worst R-hat / ESS
julia> eff = effects_table(chain, d)
julia> first(eff.competitors, 10)
```

Responses are cached under `data/cache/` (override with `DIOMEDES_CACHE`), so
repeat runs make no requests. Jolpica allows 500 unauthenticated requests per
hour; a full 1950-present history needs about 300.

A full 4-chain fit on all F1 data (6,400 results) takes under 2 minutes on a
Raspberry Pi 5. Chains run one after another by default: on that machine,
threaded chains were slower in total (see `fit_effects`).

### Gap model with lapped cars (F1)

The z-score model uses lead-lap finishers only, so it silently drops the slower
half of every field. The gap model uses every classified finisher:

```julia
julia> g = prepare_gaps(fetch_results(JolpicaF1(), 1950:2024))   # timed + lapped rows
julia> chain = fit_paceloss(g; era = :rw, n_chains = 4)           # best by PSIS-LOO so far
julia> gap_effects_table(chain, g).competitors
```

- Outcome: % gap to the winner, `100·log(T / T_winner)`. A car k laps down is an
  interval, `[100·log(L/l), 100·log(L/(l-1)))`.
- Noise: small Gaussian pace noise plus an incident loss (with probability π,
  an exponential loss with mean λ) that only ever makes results slower.
- `era = :rw` lets π and λ drift by season (random walks with heavy-tailed
  steps). `:decade`, `:regime` (regulation eras) and `loss_duration` are the
  alternatives.
- Race intercepts; driver effects sum to zero; car-season effects sum to zero
  within each season. The Indianapolis 500 (1950–60) is excluded by default.

A full 4-chain fit on all F1 data takes 40–70 min on a Raspberry Pi 5:
`scripts/gap_fit.jl all pl_rw chain k` for k = 1..4 in parallel, then `combine`.

### Adding a series

Write a `DataSource` subtype and a `fetch_results(src, seasons)` method that
returns a `DataFrame` matching `RESULT_SCHEMA` (check it with `validate_results`).
Nothing downstream is series-specific.

## Tests

```
julia --project test/runtests.jl                              # fast, offline
DIOMEDES_SLOW_TESTS=1 julia --project test/runtests.jl        # + NUTS model fit
DIOMEDES_NETWORK_TESTS=1 julia --project test/runtests.jl     # + live APIs
```

Running `test/runtests.jl` directly rather than `Pkg.test()` reuses the normal
precompile cache; `Pkg.test()` forces `--check-bounds=yes`, which needs a
separate, full recompile of the dependency tree. `Pkg.test()` still works (CI).

## Data sources

- F1: [Jolpica F1 API](https://github.com/jolpica/jolpica-f1)
- WRC/ERC: the undocumented timing API behind wrc.com; endpoints as used by
  [OpenWRC](https://github.com/jixy2012/OpenWRC) and described by
  [rallydatajunkie](https://rallydatajunkie.com/visualising-wrc-rally-results/accessing-data-from-the-wrc-live-timing-api.html).
  It is not a public contract and may change.
