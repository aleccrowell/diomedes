```@meta
DocTestSetup = :(using Diomedes; FIXTURE = joinpath(pkgdir(Diomedes), "test", "fixtures", "ergast"))
```

# Getting started

## Installation

`Diomedes` isn't registered. Clone the repository and instantiate its environment:

```
git clone https://github.com/aleccrowell/diomedes
cd diomedes
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## From data to effects

Each model follows the same three steps: fetch a results table from a source,
prepare it for the model, and fit. The examples below use a small offline
fixture: two 2019 Grands Prix in Ergast CSV format, from the test suite.

```jldoctest start
julia> res = fetch_results(ErgastCSV(FIXTURE), 2019);    # or JolpicaF1(), WRCTiming()

julia> size(res)
(40, 22)

julia> g = prepare_gaps(res)
GapData(17 timed + 16 lapped obs, 17 competitors, 10 machine-seasons, 2 races)
```

`res` has one row per driver per race ([`RESULT_SCHEMA`](@ref)). `g` holds the %
gap to the winner of each finisher, with lapped cars as intervals.

A fit is a Turing `Chains` object. On real data this is a long job (see
[Running long fits](@ref)):

```julia
chain = fit_paceloss(g; loss_duration = true, era = :rw, pace_scale = true, n_chains = 4)
identified_convergence(chain, g, "pl_dur_rw_kappa")   # R-hat / ESS of what the results use
tab = gap_effects_table(chain, g)
first(tab.competitors, 10)                            # fastest drivers: mean, sd, 90% interval
tab.machines                                          # car-season effects
```

## Fetching real data

```julia
res = fetch_results(JolpicaF1(), 1950:2024)                  # F1, Jolpica API
res = fetch_results(ErgastCSV("path/to/ergast_csv"), 2010:2019)
res = wrc_top_tiers(fetch_results(WRCTiming(), 2018:2025))   # WRC, Rally1 + Rally2
```

API responses are cached under `data/cache/` (override with the
`DIOMEDES_CACHE` environment variable), so a repeat run makes no requests. Jolpica
allows 500 unauthenticated requests per hour, and a full 1950–present history
needs about 300. The WRC API needs about 300 requests per season.

Lap times and pit stops ([`fetch_laps`](@ref), [`fetch_pit_stops`](@ref)) come
from `JolpicaF1()` (lap times 1996–, pit stops 2011–; about 11 requests per race,
cached) or from an Ergast CSV dump (`lap_times.csv`, `pit_stops.csv`).

## Tests

```
julia --project test/runtests.jl                              # fast, offline
DIOMEDES_SLOW_TESTS=1 julia --project test/runtests.jl        # + NUTS fits
DIOMEDES_NETWORK_TESTS=1 julia --project test/runtests.jl     # + live APIs
```

Running `test/runtests.jl` directly reuses the normal precompile cache.
`Pkg.test()` forces `--check-bounds=yes`, which recompiles the dependency tree;
it still works, and CI uses it.
