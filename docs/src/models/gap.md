```@meta
DocTestSetup = :(using Diomedes; FIXTURE = joinpath(pkgdir(Diomedes), "test", "fixtures", "ergast"))
```

# Gap and pace + loss models

## Outcome: % gap to the winner

```
y = 100·log(T / T_winner)
```

This is the percentage gap to the stage winner, on a log scale. Times are
*racing* times: the official time minus any red-flag suspension
(`suspended_ms`).

**Lapped cars** have no time, but their lap count bounds it. Assume constant
pace, and take a car that completed `l` of the winner's `L` laps. It finished
its `l`-th lap after the winner finished, but less than one of its own laps
later. So its full-distance gap is interval-censored:

```
y ∈ [100·log(L/l), 100·log(L/(l−1)))
```

Lapped cars are identified by status ("+N Laps", "Lapped"), not by position.
Retirements are dropped (the selection check below shows this barely moves the
effects).

```jldoctest
julia> g = prepare_gaps(fetch_results(ErgastCSV(FIXTURE), 2019))
GapData(17 timed + 16 lapped obs, 17 competitors, 10 machine-seasons, 2 races)
```

## Mean structure

```
μ = γ[race] + c − mean(c over the race),     c = driver effect + car-season effect
```

- **Race intercepts γ** are the expected gap of the race's *average* entrant. A
  dominant winner therefore doesn't shift the whole field.
- **Driver effects** sum to zero.
- **Car-season effects** sum to zero within each season. Shifting every car in a
  season one way and every race intercept in that season the other leaves
  predictions unchanged, so only within-season car differences are identified.
  Drivers changing teams link the seasons.
- **Priors** for quantities in % are set in units of the dataset's own gap scale
  ([`PriorScale`](@ref)): about 1.15 for F1 and 4.9 for WRC.

## Gap model: Student-t noise

[`gap_effects`](@ref) / [`fit_gaps`](@ref): the mean above plus symmetric noise,
[`StudentTNoise`](@ref)`(4)` (default) or [`NormalNoise`](@ref). Heavy tails
matter: with Gaussian noise, adding lapped cars inflated σ from 0.48% to 3.7%;
Student-t(4) brings it to about 1.0%.

## Pace + loss model

[`paceloss_effects`](@ref) / [`fit_paceloss`](@ref) split the noise into two
sources:

```
y = μ + ε + L,    ε ~ Normal(0, σ)                       pace noise
                  L = 0 with probability 1 − π            clean race
                  L ~ Exponential(mean λ) with prob. π    incident loss (only ever slower)
```

Spins, damage and slow stops can only lose time, which symmetric noise can only
approximate. The density is a mixture of a Gaussian and an exponentially modified
Gaussian, with numerically stable closed forms for both the density and lapped
cars' interval probabilities (`paceloss_logpdf`, `paceloss_loginterval`).

### Optional terms

All are keyword arguments of [`fit_paceloss`](@ref) and model-name suffixes for
the scripts (see [`model_spec`](@ref)):

| Option | Suffix | What it adds |
|---|---|---|
| `loss_duration` | `_dur` | slopes of logit π and log λ on standardised log race duration |
| `era = :decade` / `:regime` / `:rw` | `_decade` / `_regime` / `_rw` | era effects on π and λ: hierarchical by decade or regulation regime ([`REGIME_STARTS`](@ref)), or random walks over seasons with Student-t(3) steps ([`LossCovariates`](@ref)) |
| `pace_scale` | `_kappa` | per-season pace scale κ (a random walk), so gaps can be larger in some eras; effects in average-season units ([`race_pace_scale`](@ref)). Needed by the terms below |
| `age` | `_age` | population career curve over driver age ([`AgeCurveBasis`](@ref), [`age_curve`](@ref)) |
| `slopes` | `_slope` | per-driver linear career slopes, centred across drivers ([`CareerRows`](@ref), [`career_slopes`](@ref)) |
| `big_loss` | `_big` | a second, larger incident-loss component (repairs, many laps down) |
| `dev` | `_dev` | per-car linear in-season development trend, centred within season ([`DevRows`](@ref), [`dev_trends`](@ref)) |
| `circ` | `_circ` | per-car slopes on the race's average speed, roughness (surface score) and temperature, standardised; covariates centred over each car-season's races, slopes centred within season ([`RaceCircuits`](@ref), [`CircuitRows`](@ref), [`circuit_slopes`](@ref)) |
| `track` | `_track` | slopes of logit π and log λ on the circuit's walls score and on mixed and wet races (from the documented race conditions) |
| `walk` | `_walk` | chassis and engine random walks over seasons: each row's car part adds the walk of the works that built its chassis and of the builder of its engine, in average-season units scaled by κ and centred within season; `b_mach` becomes a per-car-season residual. Steps widen with gaps and, by learned multipliers, at changes of ownership and major regulation changes ([`CarWalks`](@ref), [`WalkStates`](@ref), [`car_walk_states`](@ref), [`car_table`](@ref); tables in [reference/cars](https://github.com/aleccrowell/diomedes/tree/master/reference/cars)) |
| `centred_drivers` | `_cdrv` | driver effects sampled in % units rather than non-centred |
| `driver_ν = 3.0` | `_tdrv` | Student-t driver prior |
| `race_mean` (default `true`) | (default; `_mgam` accepted) | race intercepts γ ~ N(μ_γ, 4s) with a learned common level μ_γ |
| `race_mean = false` | `_fgam` | the older fixed race-intercept prior |
| `race_hier` | `_hgam` | learned level and spread for the race intercepts |

### Identification notes

- **Age curve:** only its shape up to a linear tilt is identified (the
  age–period–cohort problem). A linear age term is absorbed by the race
  centring and the driver effects. Read the curve's bends, not its overall slope
  or the age of its minimum.
- **Career slopes** are centred across drivers. They rank trajectories against
  the average driver's; they don't measure whether drivers improve with age.
- **Development trends** are centred within each season. A trend shared by the
  whole field is absorbed by the race intercepts, so a dominant car can "fade"
  while still winning.
- **Circuit slopes** are centred twice. Each car-season's covariates are centred
  over its own races, so a slope is orthogonal to the car-season effect, and a
  car that raced only one kind of circuit has none. The slopes are then centred
  within season, since a slope shared by the field is absorbed by the race
  intercepts. They say which conditions suited a car relative to the field.
- **Chassis and engine walks** are centred within season. Chassis and engine
  separate only where they cross (customer engines, engine changes, 1960s
  chassis with several engines); elsewhere the split rests on the walks'
  priors. Car-season effects reported for a walk model are totals: residual +
  chassis + engine.
- **Track terms** are measured against a dry race at an open circuit, so `a_π`
  and `a_λ` refer to that race. The circuit data come from the hand-checked
  [reference tables](https://github.com/aleccrowell/diomedes/tree/master/reference/circuits)
  ([`circuit_table`](@ref)).

### Results

```julia
tab = gap_effects_table(chain, g)    # competitors and machines: mean, sd, 90% interval, rows
```

With `pace_scale`, driver effects are in average-season units, and car effects
are in their own season's units. Per-row career offsets are
[`driver_offsets`](@ref); per-season incident rates come from
`scripts/era_paths.jl`.

## Retirements and selection

The pace models drop retirements. If cars that retire were running slower than
their effects predict, the finishers are a selected sample. `src/partial.jl` and
`scripts/selection_check.jl` test this using lap times (F1, 1996–):

- [`ergast_cumulative_laps`](@ref) and [`partial_gaps`](@ref) build each
  retiree's gap at its last lap, calibrated within the race to a full-distance
  scale.
- [`with_partial_rows`](@ref) adds them to the results as timed rows, and
  [`partial_mask`](@ref) marks them.

Result (#13): retirees *were* running about 0.35 points slower before they
stopped, but adding them doesn't materially move the driver or car effects.
