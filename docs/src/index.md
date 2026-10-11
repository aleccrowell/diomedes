# Diomedes.jl

Bayesian models of driver vs machine performance in motorsport, in Julia with
[Turing.jl](https://turinglang.org).

The question: how much of a result is the driver and how much is the car? Every
model here has **crossed random effects** for competitors (drivers) and
machine-seasons (a car in one season), fitted to public timing data: Formula 1
from 1950 (Jolpica / Ergast) and the WRC and ERC rally championships (the wrc.com
timing API).

## How the package is organised

Data flows through three layers, and only the first knows which series it is
looking at:

1. **Sources** fetch raw data from a public API or local files and normalise it
   into a common long-format table, one row per competitor per timed stage
   ([`RESULT_SCHEMA`](@ref)). See [Data](@ref).
2. **Preparation** (`prepare*` functions) turns that table into the inputs of one
   model: an outcome per row and integer indices for competitors, machine-seasons
   and races.
3. **Models** are Turing models over those inputs, each with a `fit_*` function
   and summary helpers.

| Model | Outcome | Prepare | Fit | Page |
|---|---|---|---|---|
| Z-score (legacy port) | stage time standardised within stage | [`prepare`](@ref) | [`fit_effects`](@ref) | [Z-score model](@ref) |
| Gap, Student-t noise | % gap to winner, lapped cars as intervals | [`prepare_gaps`](@ref) | [`fit_gaps`](@ref) | [Gap and pace + loss models](@ref) |
| Pace + loss (main model) | as above, with incident losses | [`prepare_gaps`](@ref) | [`fit_paceloss`](@ref) | [Gap and pace + loss models](@ref) |
| Retirement | laps completed and cause of retirement | [`prepare_retirements`](@ref) | [`fit_retirement`](@ref) | [Retirement model](@ref) |
| Lap time (F1, 1996–) | lap time, % of the race's median clean lap | [`prepare_laps`](@ref) | [`fit_laps`](@ref) | [Lap-time model](@ref) |

Adding a series means writing one source adapter; nothing downstream changes
(see [Adding a series](@ref)).

```@docs
Diomedes
```

## Conventions

- **Units.** Pace effects are in % of race (or lap) time, on a log scale:
  `100·log(T / T_ref)`. Negative is faster. The retirement model's effects are
  on the log-hazard scale.
- **Identification.** Driver effects sum to zero, so they are relative to the
  average driver in the data. Car-season effects sum to zero *within each
  season*: with race intercepts, only differences between cars in the same
  season are identified. Drivers who change teams link the seasons together.
- **Issue numbers** (#9, #15, ...) in docstrings and comments refer to the
  [GitHub issues](https://github.com/aleccrowell/diomedes/issues) where each
  piece was designed and evaluated.
