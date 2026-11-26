# The lap-time model

The race-time models get one number per driver per race. Lap times
(Jolpica, 1996 on) give about 50 per driver per race, including the
laps a car completed before retiring. The lap-time model ([#12](https://github.com/aleccrowell/diomedes/issues/12)) fits

```math
y = 100\log\frac{t}{\text{median clean lap}} = \gamma_r + \delta_r\,(f - \bar f_r) + a_d + b_m + \beta_1 s + \beta_2 s^2 - (\text{race mean}) + \varepsilon + L ,
```

with ``f`` the lap's fraction of the race distance (fuel burn, track
evolution), ``s`` the tyres' stint age, and the same pace + loss noise as the
race model. The current fit (2021–2026, pit stops checked against the timing data;
[#108](https://github.com/aleccrowell/diomedes/issues/108),
[#113](https://github.com/aleccrowell/diomedes/issues/113),
[#116](https://github.com/aleccrowell/diomedes/issues/116),
[#124](https://github.com/aleccrowell/diomedes/issues/124)) adds:
- **peak pace and shortfall** ([#113](https://github.com/aleccrowell/diomedes/issues/113)): the
  small-loss part of the noise becomes each driver's shortfall from their peak.
  Its probability and size get driver and car-season offsets, so a driver has a
  peak (the clean-lap edge), a frequency and a size of falling short, and a mean
  pace (peak plus expected shortfall). The driver and car effects are sampled on
  the mean pace, which the data pins tightly, and the peak is derived from it;
- **traffic**: shared effects of the gap to the car ahead on track (dirty air),
  of lapped cars ahead and blue flags, of a car close behind on the same lap
  (defending), and of leading by over 5 s (cruising), from cumulative lap times;
- **track conditions** ([#116](https://github.com/aleccrowell/diomedes/issues/116),
  [#124](https://github.com/aleccrowell/diomedes/issues/124)), from the F1 timing data,
  which is why the fit starts in 2021: the tyre compound, track temperature and rain on
  each lap; drivers', car-seasons' and races' offsets on wet tyres; a stint slope for
  each race; and each stint's true tyre age, with pit stops reconciled against the
  timing data's tyre changes ([#151](https://github.com/aleccrowell/diomedes/issues/151));
- **field pace**: the field's median lap, smoothed over neighbouring laps, beyond
  the race's mean and trend, with a slope per car-season, for conditions that change within a race (a drying track, rain
  arriving). It also moves the shortfall's probability and size, but not the edge;
- from the race model ([#108](https://github.com/aleccrowell/diomedes/issues/108)):
  walls, mixed and wet races on the incident probability and size; per-driver
  incident proneness; a share of incidents that are big losses; and a walk of
  each team's chassis form from race to race, which replaces the fixed car-season
  effect and the in-season development trend;
- a career slope per driver (on season-level age), as in the race model;
- a stint-age curve, with driver and car-season deviations from it (tyre
  management), on stint age centred per driver and per car-season;
- an effect of leading the race on the lap (clean air against managing a lead;
  the two can't be told apart);
- lap 1 as its own term: its time against the race's median lap 1, with the
  drivers' and cars' pace effects, a curve in the grid slot, and car-season start
  effects, with its own noise. Per-driver start effects are off, because they
  were barely identified;
- the race model's population age curve, shared by all drivers
  ([#12](https://github.com/aleccrowell/diomedes/issues/12)). Over 2021–2026 it isn't
  identified (each driver covers at most six ages, and the career slopes take any
  steady change), so it isn't published;
- car-season slopes on three circuit dimensions, centred within each season:
  average lap speed, street score and race-time air temperature
  ([#14](https://github.com/aleccrowell/diomedes/issues/14)). The covariates come
  from hand-checked circuit tables, with sources, in `reference/circuits/`.

It is fitted to green-flag laps.
Excluded:
- lap 1 from the pace term (it enters the start term);
- safety-car, VSC and red-flag laps, detected as race-wide slow laps;
- pit in- and out-laps (Jolpica's recorded stops, reconciled with the timing data).

No lap is dropped for its own size: a per-car cut on slow laps, like inferred
pit stops, would also drop a car's largest incidents (about 5 per 1,000 laps,
more for incident-prone drivers; [#108](https://github.com/aleccrowell/diomedes/issues/108)).
Traffic, blue flags, mistakes and damage stay in: the traffic terms, the
shortfall and the incident losses, big ones included, take them.

## What it measures

Green-flag pace under race conditions, plus the start, not race results. Since
[#113](https://github.com/aleccrowell/diomedes/issues/113) it separates a
driver's peak from how often and how far they fall short of it. A driver well ahead lifts and coasts and saves tyres, so
the lap model underrates comfortable winners. Pit-stop timing, strategy and
safety-car periods are left out. So its driver effects are less spread out than
the race model's, compressed most at the front. Neither model is the reference
for the other. The race model measures results, from one observation per driver
per race. The lap model measures clean-lap pace, from about 50 times more
observations, which constrains the split between car and driver and in-season
development more tightly within its seasons. Where the two agree, the result
holds across both models' blind spots. Where they disagree, either could be
wrong: independent checks, such as held-out races or simulated data with known
effects, are needed to tell which. Adding starts, lead laps and tyre management did not reduce the compression: the
slope of lap-model driver effects on race-model ones stayed about 0.41 over
1996–2025, while the correlation rose slightly, from 0.67 to 0.69
([#66](https://github.com/aleccrowell/diomedes/issues/66)).

## Status

This model is in development. The published fit (2021–2026) has the [#108](https://github.com/aleccrowell/diomedes/issues/108), [#113](https://github.com/aleccrowell/diomedes/issues/113), [#116](https://github.com/aleccrowell/diomedes/issues/116) and [#124](https://github.com/aleccrowell/diomedes/issues/124) terms. Earlier fits, below, covered longer spans without the track conditions.

- **Pit laps can be inferred from lap times.** Over 2011–2019, excluding
  inferred pit laps instead of Ergast's recorded ones gives the same effects:
  driver correlation 0.998, car-season correlation 1.000, no difference larger
  than its posterior uncertainty. That opens up 1996–2010, before pit data.
- **It agrees with the race model on cars, less on drivers.** Over 2011–19, the
  correlation with the race model is 0.97 for car-seasons and 0.70 for drivers.
  The race model's driver effects span whole careers, with age and career-slope
  terms; the lap model's cover only those seasons.
- **The tyre term doesn't move the effects.** Removing the stint-age terms
  changes the driver and car effects by less than their posterior uncertainty.
  Tyre wear is real variance within a race, but almost unrelated to who is
  driving.
- **1996–2025, with career slopes and development, no tyre terms.** Cars agree
  with the race model at 0.96 over 320 car-seasons, drivers at 0.68–0.70. The
  drivers of 1996–97, at the edge of the data, are the least reliable.
- **1996–2025 with the #66 terms.** Its drivers correlate 0.977 with
  the fit without them. Laps get faster through a stint in the refuelling era
  (−0.31 % per 10 laps, as fuel burns off) and slower since 2010 (+0.60 %). The
  chains settle in two modes of equal log density. The modes differ in the spread
  of the per-race trends and in the lap-noise mixture. The driver, car,
  tyre-management and start effects agree across them (correlation ≥ 0.998).
- **1996–2025 with the age curve and circuit slopes.** One mode;
  max R-hat 1.014, no divergences. Car-seasons correlate 0.953 with the race
  model (0.954 before), drivers 0.761 (0.718 before). Its effects correlate 0.989
  (cars) and 0.954 (drivers) with the previous fit. The circuit slopes vary by
  about 0.16 % per sd of speed and of street score, and 0.11 % for temperature.
- **2011–2026 with the #108 and #113 terms (the previous published fit).** One mode, max
  R-hat 1.018, no divergences, 4 × 1,000 draws. Car-seasons correlate 0.97 with the
  race model, drivers' mean pace 0.66 (drivers with at least 500 laps). Peak pace
  carries most of the driver spread: across drivers with at least 2,000 laps the
  peaks vary about six times as much as the expected shortfalls, and the two are
  unrelated. The field pace moves the edge only weakly (slope 0.17, where 1 would
  mean conditions slow the limit fully), and wet and mixed laps go mostly to the
  shortfall ([#116](https://github.com/aleccrowell/diomedes/issues/116)).
- **2021–2026 with per-lap conditions, first fit** ([#116](https://github.com/aleccrowell/diomedes/issues/116)).
  Tyre compound, track temperature and rain, and drivers' deviations on wet tyres.
  Drivers' wet deviations spread with an sd of 0.73 %, but carried their car's wet
  performance and how wet their races were. The driver and car levels didn't
  converge (R-hat 1.08–1.09).
- **2021–2026 with car and race wet terms, and the race walk sampled relative to
  each car's drivers** ([#116](https://github.com/aleccrowell/diomedes/issues/116)).
  Every group converged (max R-hat 1.019). Drivers' wet spread fell to 0.29 %.
- **2021–2026 with per-race tyre wear, true tyre age, reconciled pit stops and
  pooled age bins (the current fit;** [#124](https://github.com/aleccrowell/diomedes/issues/124)**).**
  One mode, max R-hat 1.032, no divergences, 4 × 1,000 draws. Car-seasons correlate
  0.94 with the race model, drivers' mean pace 0.64 (drivers with at least 500 laps).
  The compounds' own stint slopes go to about zero once each race has its own; the
  field-pace slope is 0.12 ([results](@ref "Track conditions")).
