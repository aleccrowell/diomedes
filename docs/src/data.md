# Data

## The results table

Every source returns a `DataFrame` in one long format, [`RESULT_SCHEMA`](@ref):
**one row per competitor per timed stage**.

| Series | Event | Stage |
|---|---|---|
| F1 | Grand Prix | the race (one stage per event) |
| WRC / ERC | rally | special stage (about 15–25 per event) |

The columns that matter most to the models:

- `competitor_id`: the driver, stable across seasons. `competitor_birth` (if
  known) is used for the career-age terms.
- `machine_id`: whatever the series treats as "the car". In F1 that is the
  constructor; in rallying it is the vehicle model as typed by the organisers,
  so `manufacturer`, `class` and `entrant` give cleaner groupings. The models
  pair it with `season` to form a **machine-season**, since cars change from year
  to year. Every `prepare*` function takes a `machine_key(row)` to change the
  grouping; for WRC, [`wrc_machine_key`](@ref) gives manufacturer × tier × season.
- `time_ms`, `laps`, `status`, `classified`: the outcome. `time_ms` is missing for
  untimed rows (lapped cars, retirements). `status` is the source's own text
  ("Finished", "+1 Lap", "Engine", ...), which the gap and retirement models
  parse.
- `suspended_ms`: red-flag suspension time included in the official race time.
  Since 2005, a red-flagged race resumes and its official time includes the
  stoppage (Canada 2011: about 123 of the winner's 245 minutes). The gap model
  subtracts it. It is detected from lap times; `missing` means unknown.

[`validate_results`](@ref) checks a table against the schema, and
[`empty_results`](@ref) gives a zero-row table to build one from.

The lap-time model also uses per-lap tables: [`LAP_SCHEMA`](@ref) (one row per
car per lap) and [`PIT_SCHEMA`](@ref) (one row per pit stop).

## Sources

| Source | Coverage | Notes |
|---|---|---|
| [`JolpicaF1`](@ref) | F1, 1950–present | The maintained successor to the Ergast API. Rate-limited; cached. |
| [`ErgastCSV`](@ref) | F1, from a local Ergast CSV dump | Offline. Also provides lap times (1996–) and pit stops (2011–). |
| [`WRCTiming`](@ref) | WRC 2018–, ERC 2022– | The undocumented timing API behind wrc.com. Not a public contract and may change. |

Both F1 sources drop the Indianapolis 500 (1950–1960) by default
(`include_indy500 = true` keeps it): it counted towards the championship but was
effectively a separate series, with a field of drivers who mostly raced only there.

For WRC, [`wrc_top_tiers`](@ref) keeps the top two tiers: Rally1 (including WRC
cars before 2022) and Rally2 (including R5 before 2020), using [`wrc_tier`](@ref).
Only stages with status "Completed" are included; cancelled and interrupted
stages get notional times and are skipped.

### Caching and rate limits

HTTP sources cache every response on disk under `data/cache/` (or
`DIOMEDES_CACHE`), keyed by URL, and back off on HTTP 429 and 5xx responses. Pass
`refresh = true` to a source to re-fetch, for example for a season still in progress.

## Adding a series

1. Define a subtype of [`DataSource`](@ref).
2. Add a method `fetch_results(src::MySource, seasons)` that returns a table
   matching [`RESULT_SCHEMA`](@ref). End it with
   `validate_results(df)`.
3. If the series' machine identifier isn't a clean "car", pass a `machine_key`
   when preparing (as [`wrc_machine_key`](@ref) does for WRC).
4. Check that its priors suit the data's scale. The pace models set their
   priors from the data's own gap scale ([`PriorScale`](@ref)), so F1 priors
   aren't reused on a series with much larger gaps (WRC's are about 4× F1's).

`src/sources/wrc.jl` is a complete worked example: it walks the WRC API's
season → rally → stage → times endpoints, normalises free-text vehicle names
and maps entry classes to tiers.

Nothing downstream is series-specific. A circuit series with lap counts can
also use the lapped-car intervals and the retirement model; stage-based series
(rallying) use timed rows only.
