# The lap-time model

The race-time models get one number per driver per race. Lap times
(Ergast, 1996–2019) give about 50 per driver per race, including the laps a car completed
before retiring. The lap-time model ([#12](https://github.com/aleccrowell/diomedes/issues/12)) fits

```math
y = 100\log\frac{t}{\text{median clean lap}} = \gamma_r + \delta_r\,(f - \bar f_r) + a_d + b_m + \beta_1 s + \beta_2 s^2 - (\text{race mean}) + \varepsilon + L ,
```

with ``f`` the lap's fraction of the race distance (fuel burn, track
evolution), ``s`` the tyres' stint age, and the same pace + loss noise as the
race model. It is fitted to clean green-flag laps only. Excluded:
- lap 1;
- safety-car, VSC and red-flag laps, detected as race-wide slow laps;
- pit in- and out-laps.

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
- **Known issues before 1996–2019 results are usable:** refuelling (1994–2009)
  makes a shared stint term mis-specified across eras, and driver pace needs
  career terms over a 24-season range.
