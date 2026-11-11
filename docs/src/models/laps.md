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
