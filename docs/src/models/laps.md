# Lap-time model

[`lap_effects`](@ref) / [`fit_laps`](@ref): per-lap pace instead of whole-race
times (#12). It uses partial races before a retirement and has far more data
per driver–car pair. It is **in development**: the published fit has the #108 and #113 terms, and per-lap conditions are in progress in #116.

## Outcome and clean laps

```
y = 100·log(t / median clean lap of the race)
```

This is on the same % scale as the race-time models. [`prepare_laps`](@ref)
keeps clean green-flag laps and excludes:

- lap 1 (the standing start);
- race-wide slow laps (safety car, VSC, red flag), detected when the field's
  median lap is over 1.25× the race median, plus the restart lap after each;
- a car's laps over 2× its own median lap (a red-flag stoppage, or a very long stop),
  unless `stoppage = false` (`lap_fit.jl … keepslow`, #108). With recorded stops
  no lap is then dropped for its own size, so a car's largest incidents stay in
  for the big-loss component;
  green-flag wet laps are kept too with `keep_laps` (`lap_fit.jl … keepwet`, #116):
  laps with at least half the field on wet tyres where neither the lap nor the one
  before is neutralised (safety car, VSC or red flag, from the conditions table's
  `neutral` column; [`Diomedes.wet_green_laps`](@ref)). Over 2021–2026 they add 263 laps;
- pit in- and out-laps, either **recorded** (Ergast's pit stops, 2011 on) or
  **inferred** from the lap times ([`infer_pit_stops`](@ref)).

Inferred stops give the same effects as recorded ones over 2011–2019
(correlation 0.998 for drivers, 1.000 for car-seasons). That makes 1996–2010 usable.

## Model

```
y = γ[race] + δ[race]·(frac − mean frac) + driver + car-season + β₁[era]·s + β₂[era]·s²
    + η·lead + tyre_driver·s_d + tyre_car·s_m − (race mean) + noise
```

- `frac` is the lap's fraction of the race distance. The per-race slope δ
  (partially pooled) absorbs fuel burn and track evolution.
- `s` is stint age in tens of laps. With `tyres`, its curve (β₁, β₂) is shared
  within an era: the refuelling era (1994–2009) and since 2010.
- `lead = true` adds η, the effect of leading the race on the lap.
- `tyre_mgmt = true` adds driver and car-season slopes on stint age, centred per
  driver (`s_d`) and per car-season (`s_m`), as deviations from the era's curve.
- `starts = true` adds lap 1 as its own term ([`lap_effects`](@ref)): its time
  against the race's median lap 1, with the drivers' and cars' pace effects, a
  quadratic in the grid slot, and driver and car-season start effects. Pit-lane
  starts are left out. The term has its own pace + loss noise.
- `slopes` and `dev` add career slopes and in-season development, as in the race model.
- `age = true` adds the population age curve of the race model ([`AgeCurveBasis`](@ref)): a
  random walk over one-year bins of season age, with its constant and linear parts
  projected out (they can't be separated from the career slopes and car-seasons).
  τ_age ~ half-N(0, 0.2).
- `circuits = true` adds car-season slopes on standardised race covariates from
  [`circuit_table`](@ref): average lap speed (per layout, so a wet race doesn't read as a
  slow circuit), street score and race-window air temperature. The slopes are
  centred within each season, with a spread τ_cx ~ half-N(0, 0.5) per dimension.
- Noise is pace + loss, as in the race model. Traffic, mistakes and damage are
  incident losses.
- Terms from the race model (#108): `track = true` puts walls, mixed and wet races
  on logit π and log λ (as in #81); `dinc = true` adds per-driver incident
  proneness, centred; `big = true` makes a share ρ of incidents big losses of their
  own mean (as in #16), with the proneness then on that share; `rwalk = true`
  replaces the car-season effect and the development trend with a walk of each
  works' chassis form from race to race ([`RaceWalk`](@ref)), pooled across
  off-seasons with steps for changes of ownership and major rule changes;
  `sdrv = false` drops the per-driver start effects (barely identified).
  With `rwrel = true` (`lap_fit.jl … rwrel`, #116) the sampler draws the walk
  states relative to each car's drivers, x̃ = x + W·a, with W each driver's share
  of a state's laps: a shear with Jacobian 1, so the model is unchanged. Over
  2021–2026 it lifts the minimum ESS of drivers from 81 to 219 and of the walk
  from 85 to 334; `rwrels` also subtracts the career slopes, but moves the slow
  direction onto the age curve.
- Maximum pace and consistency (#113), through a [`LapContext`](@ref) built by
  `LapContext(d, laps; traffic, fpace, short)`:
  - `traffic`: shared effects of binned gaps from [`LapTraffic`](@ref): to the car
    ahead on track (dirty air), a lapped car ahead, blue flags, a car close behind
    on the same lap (defending), and leading by over 5 s (cruising);
  - `fpace`: the field's pace on each lap ([`field_pace`](@ref): the field's median lap,
    smoothed over neighbouring laps, beyond the race's mean and trend), with a slope per car-season around a
    shared β_fp ~ N(1, 0.5). With `short` it also moves the shortfall's logit π and
    log λ (κ_fp), but not the edge;
  - `short`: driver and car-season offsets on the small-loss part's logit π and
    log λ, which becomes the shortfall from each driver's peak. The driver and car
    effects are then on the mean pace, and the clean-lap edge sits below it by the
    expected shortfall from the race, driver and car offsets. The field pace
    shapes the losses but doesn't move the edge, which would make a very stiff
    direction (#113: the fit then diverged on nearly every iteration).
- Per-lap conditions (#116, 2021 on), through a [`LapConditions`](@ref) built from
  [`read_lap_conditions`](@ref) (the table of `scripts/conditions/fetch_conditions.py`:
  FastF1 for 2021–22, OpenF1 from 2023) and [`jolpica_driver_numbers`](@ref):
  - `cond`: offsets for each compound against the SOFT ([`COMPOUNDS`](@ref)), a stint
    slope per compound on top of the shared curve, track temperature against the
    race's mean (per 10 °C) and rain falling, on the mean pace. With `short` also
    wet tyres and rain on the shortfall's logit π and log λ, left out of the edge
    as the field pace is;
  - `wetdrv`: each driver's centred deviation on intermediate and wet-tyre laps.
    The car-season deviations (`wetdev`, or `wetwalk` pooled through a wet walk
    per works) are kept for reference: in both probes they weren't identified, with
    2023 cars of 30–50 wet laps from two mixed races coming out several % off;
  - `wetshr`: drivers' deviations as with `wetdrv`, plus car-season deviations for
    the car-seasons with at least 100 wet-tyre laps only (31 of 61 over 2021–2026),
    centred within season, spread ~ half-N(0, 0.5). A likelihood profile showed why
    the few-lap ones failed: towards a faster wet limit the losses absorb the
    difference, so with ~30 wet laps the likelihood is nearly flat there (#116).
  - `wetall`: car-season deviations for every car-season with a wet lap, centred
    within season; `wetallg` centres them over all car-seasons. Alone, either lets
    the few-lap car-seasons take up a race's wetness (2023's Williams at +2.6 %);
  - `wetrace`: a wet-lap offset per race with wet laps, spread ~ half-N(0, 2),
    centred over those races. It takes up how wet each race was, so with
    `wetallg wetrace keepwet` every car-season's deviation stays within ±1 % and
    its uncertainty grows as its wet laps fall (2021–2026 full fit, #116). Races
    run entirely on wet tyres stay at the prior, their offset not told apart from
    the race's mean lap.
  - `racedeg`: each race's own stint slope around the shared tyre curve,
    τ_rd ~ half-N(0, 1), on the stint age centred over all laps (#124). Track
    surface, temperature and Pirelli's nomination change how fast every compound
    degrades at a race, and a per-compound slope keyed to the C-number didn't
    separate them (the labels degrade alike once the race's level is out).
    `racedegc` puts each race's slope on the age centred within that race instead,
    so the slope doesn't trade off against the race's mean lap; it is the variant
    to use (2021–2026 full fits otherwise alike: drivers' R-hat 1.057 → 1.040,
    age curve 1.112 → 1.077).
  - `tyreage`: the stint age starts at the tyre's true age from the timing data
    (OpenF1/FastF1 `tyre_age`, less the 1 lap a new set shows), so a used set
    starts older. 1,574 of 6,542 stints over 2021–2026 started on used sets
    (median 2 laps). Extras above 10 laps are treated as unknown: they came from
    stops the timing data missed.
  - `pitfix`: reconciles Jolpica's pit stops with the timing data's tyre stints
    ([`reconcile_pit_stops`](@ref), #151). A tyre change in the timing data adds a
    stop; a stop the timing data doesn't confirm is dropped under a safety car or
    red flag (277 over 2021–2026, mostly double records) and kept on green, with
    the timing age after it treated as unknown (53).
  - `agemin=k`: pools neighbouring driver-age bins, from the oldest down, until each
    has at least k distinct drivers, labelled by the lap-weighted mean age. With
    k = 3 the 27 ages 18–45 become 19 bins (18–19, 34–35, 36–37, 38–40, 41–45
    merged). Ages 40–45 were Alonso's alone, and the curve there didn't converge
    (R-hat 1.077); pooled, it reaches 1.032.
  - `agerw2`: the age curve as a second-order walk, τ_age ~ half-N(0, 0.05),
    kept for reference. On 2021–2026 the sampler went stiff (step size
    0.001–0.01 at tree depth 10): the data ask for year-to-year jumps that a
    smooth curve can't make, even with the tight prior.
- Driver and car-season effects are centred as in the race model. Every term is
  centred within its race, so γ is the race's mean lap.

## Long fits and warm starts

About 175k laps (2011–19) to 530k (1996–2025): a fit takes hours to a day on a
Raspberry Pi 5. The 2011–2026 fit with every #108 and #113 term (316k laps) took
about 9 h per chain at ~230 leapfrog steps per draw. Fits are checkpointed (see [Running long fits](@ref)). A
larger fit can warm-start from a smaller one: [`lap_warm_start`](@ref) maps the
adapted step size and diagonal metric across by parameter label, and
[`lap_warm_sampler`](@ref) runs AdvancedHMC's NUTS from there with a shorter
adaptation.

```julia
d = prepare_laps(fetch_laps(src, 2011:2019), fetch_results(src, 2011:2019); pits = :inferred)
chain = fit_laps(d; checkpoint = "output/laps_2011-2019_chain1", seed = 1000)
```

## Known issues

- **Refuelling (1994–2009):** the car lightens as fuel from its last stop burns
  off. A single stint curve shared across eras collapsed. The curves are now
  separate per era, and before 2010 the driver and car stint slopes mix tyre
  management with fuel strategy.
- **Two modes in the 1996–2025 fit without the age curve and circuit slopes:**
  they differed in the spread of the per-race trends (τ_δ) and in the lap-noise
  mixture, with the same effects. The current fit with both terms has one mode.
- **Adapted step sizes vary between chains:** in the 1996–2025 fits, some chains
  adapt step sizes that put them at tree depth 9 instead of 7, about four times
  slower. Resuming them at a fixed step size (`lap_fit.jl … step=<x>`) fixes this.
- **Career slopes are loose:** τ_slope ≈ 2, against about 1 in the race model.
  Short careers at the edges of the data trade slope against level.
- **Conditions go to the shortfall:** in the 2011–2026 fit the field pace moves the
  edge with a slope of only 0.17, and wet and mixed laps are read mostly as
  shortfall rather than a slower limit. The conditions terms (#116) cover 2021 on
  only. With them, over 2021–2026, the slope is 0.13, so they don't explain it; a
  2021–2026 fit without them, to compare like for like, hasn't been run.
- **Driver and car levels mix slowly with the conditions terms:** the 2021–2026 fit
  with `cond wetdrv` reached R-hat 1.08–1.09 on drivers, car-seasons and the race
  walk after 4 × 1,000 draws, along the ridge between a dominant driver and their
  car. Its conditions and wet-skill terms converged (R-hat ≤ 1.015).
  `rwrel` removes most of this: with `cond wetallg wetrace keepwet rwrel` the
  maximum R-hat is 1.019, with the age curve now the slowest (minimum ESS 188).
  Adding `racedegc tyreage pitfix agemin=3` (#124) brings it to 1.032 (drivers,
  peak and age curve), with minimum ESS 141 on the age curve and about 257
  leapfrog steps per draw.
- **Step tuning after warm-up can overshoot:** with `racedeg`, the step size tuned
  after warm-up gave deeper trees than the warm-up's own. `notune` keeps the
  warm-up's step.
