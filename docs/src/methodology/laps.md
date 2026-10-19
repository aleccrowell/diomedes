# The lap-time model

The race-time models get one number per driver per race. Lap times
(Jolpica, 1996 on) give about 50 per driver per race, including the
laps a car completed before retiring. The lap-time model ([#12](https://github.com/aleccrowell/diomedes/issues/12)) fits

```math
y = 100\log\frac{t}{\text{median clean lap}} = \gamma_r + \delta_r\,(f - \bar f_r) + a_d + b_m + \beta_1 s + \beta_2 s^2 - (\text{race mean}) + \varepsilon + L ,
```

with ``f`` the lap's fraction of the race distance (fuel burn, track
evolution), ``s`` the tyres' stint age, and the same pace + loss noise as the
race model. The current fit (1996–2025, [#66](https://github.com/aleccrowell/diomedes/issues/66)) adds:
- a career slope per driver (on season-level age) and an in-season development
  trend per car-season, as in the race model;
- separate stint-age curves for the refuelling era (1994–2009) and since 2010.
  Before 2010 the car also lightens as the fuel from its last stop burns off, so
  the curve mixes tyre wear with fuel load;
- driver and car-season deviations from those curves (tyre management), on stint
  age centred per driver and per car-season;
- an effect of leading the race on the lap (clean air against managing a lead;
  the two can't be told apart);
- lap 1 as its own term: its time against the race's median lap 1, with the
  drivers' and cars' pace effects, a curve in the grid slot, and driver and
  car-season start effects, with its own noise;
- the race model's population age curve, shared by all drivers
  ([#12](https://github.com/aleccrowell/diomedes/issues/12));
- car-season slopes on three circuit dimensions, centred within each season:
  average lap speed, street score and race-time air temperature
  ([#14](https://github.com/aleccrowell/diomedes/issues/14)). The covariates come
  from hand-checked circuit tables, with sources, in `reference/circuits/`.

It is fitted to green-flag laps.
Excluded:
- lap 1 from the pace term (it enters the start term);
- safety-car, VSC and red-flag laps, detected as race-wide slow laps;
- pit in- and out-laps.

Traffic, blue flags and mistakes stay in; the loss component absorbs part of
them.

## What it measures

Typical green-flag pace under race conditions, plus the start, not peak pace
and not race results. A driver well ahead lifts and coasts and saves tyres, so
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

This model is in development; see [#12](https://github.com/aleccrowell/diomedes/issues/12) for the current state.

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
- **1996–2025 with the age curve and circuit slopes (the current fit).** One mode;
  max R-hat 1.014, no divergences. Car-seasons correlate 0.953 with the race
  model (0.954 before), drivers 0.761 (0.718 before). Its effects correlate 0.989
  (cars) and 0.954 (drivers) with the previous fit. The circuit slopes vary by
  about 0.16 % per sd of speed and of street score, and 0.11 % for temperature.
