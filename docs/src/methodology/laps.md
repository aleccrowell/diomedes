# The lap-time model

The race-time models get one number per driver per race. Lap times
(Ergast to 2019, Jolpica from 2020) give about 50 per driver per race, including the
laps a car completed before retiring. The lap-time model ([#12](https://github.com/aleccrowell/diomedes/issues/12)) fits

```math
y = 100\log\frac{t}{\text{median clean lap}} = \gamma_r + \delta_r\,(f - \bar f_r) + a_d + b_m + \beta_1 s + \beta_2 s^2 - (\text{race mean}) + \varepsilon + L ,
```

with ``f`` the lap's fraction of the race distance (fuel burn, track
evolution), ``s`` the tyres' stint age, and the same pace + loss noise as the
race model. The current fit (1996–2025) drops the stint-age terms and adds a
career slope per driver (on season-level age) and an in-season development trend
per car-season, as in the race model. It is fitted to green-flag laps.
Excluded:
- lap 1;
- safety-car, VSC and red-flag laps, detected as race-wide slow laps;
- pit in- and out-laps.

Traffic, blue flags and mistakes stay in; the loss component absorbs part of
them.

## What it measures

Typical green-flag pace under race conditions, not peak pace and not race
results. A driver well ahead lifts and coasts and saves tyres, so the lap
model underrates comfortable winners. Starts, pit-stop timing, strategy and
safety-car periods are left out entirely. So its driver effects are less
spread out than the race model's, compressed most at the front. The race model
stays the primary result. The lap model is supporting evidence for the
split between car and driver, from about 50 times more observations, and for
in-season development. Including starts, and driver and car effects on tyre
management, would bring it closer to the race outcome
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
