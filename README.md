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
  models.jl          Turing models, fitting, effect summaries
scripts/
  legacy_parity.jl   check against the original Python pipeline
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
julia> chain = fit_effects(d; σ_y = nothing)             # NUTS; σ_y=1.0 matches the legacy model
julia> eff = effects_table(chain, d)
julia> first(eff.competitors, 10)
```

Responses are cached under `data/cache/` (override with `DIOMEDES_CACHE`), so
repeat runs make no requests. Jolpica allows 500 unauthenticated requests per
hour; a full 1950-present history needs about 300.

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
