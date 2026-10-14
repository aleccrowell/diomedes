# Retirements and selection bias

## A competing-risks model of retirements

Retirements make up a large share of F1 starts: 6,308 mechanical failures,
1,781 solo incidents, 893 collisions and 240 retirements of unknown cause in
25,074 starts from 1950 to 2025 (Jolpica data; [#42](https://github.com/aleccrowell/diomedes/issues/42), [#55](https://github.com/aleccrowell/diomedes/issues/55)). The retirement model treats each start as a
survival record in laps. Each cause ``k`` has a hazard that is constant over the
race, plus a jump on the first lap:

```math
H_k(n) = e^{\eta_k}\,\bigl(B_k\,[n \ge 1] + n/L\bigr),
```

with linear predictors

```math
\begin{aligned}
\eta_\text{mech} &= \alpha_\text{mech}(s) + \text{car reliability}_m + \text{race}_r \\
\eta_\text{incident} &= \alpha_\text{inc}(s) + \text{driver proneness}_d + \text{race}_r \\
\eta_\text{collision} &= \alpha_\text{col}(s) + \beta_\text{col}\cdot\text{driver proneness}_d + \text{race}_r .
\end{aligned}
```

A retirement recorded only as "Retired" (most of Jolpica's from 2023) counts
under the total hazard, with no split between causes. Retirements with other
causes (fuel, puncture, illness) are treated as censored. Disqualified starts are dropped, since how they would
have ended is unknown ([#42](https://github.com/aleccrowell/diomedes/issues/42)).

**Results** (1950–2025; the 1950–2019 fit in [#42](https://github.com/aleccrowell/diomedes/issues/42) gave the same within uncertainty):

- **Collisions are only partly a driver trait.** The driver's
  incident-proneness carries over to collisions with loading
  ``\beta_\text{col}`` = 0.37 ± 0.10.
- **Car and driver spread.** Car-season reliability varies with
  ``\sigma_\text{car}`` = 0.50 ± 0.02 on the log-hazard scale, and driver
  proneness with ``\sigma_\text{drv}`` = 0.45 ± 0.04.
- **Slow and unreliable go together.** Slower drivers are more incident-prone
  (correlation 0.43 over 149 drivers with at least 50 starts), and slower
  car-seasons less reliable (0.47 over 675 with at least 20 starts).

## Do the dropped retirements bias the pace effects?

If a car's chance of retiring depends on how it was running *that day*, the
finishers are a selected sample. The test ([#43](https://github.com/aleccrowell/diomedes/issues/43)) uses lap times (1996–2019):

1. **Partial gaps.** For each retirement that covered at least a quarter of the
   distance (1,647 starts), take its gap to the eventual winner at its last
   completed lap.
2. **Calibration.** Gaps change over a race, so each partial gap is shifted by
   how the finishers' gaps changed from that lap to the flag. On finishers,
   this predicts the final gap without bias (0.000 ± 0.009).
3. **Two fits.** Fit the best pace model with and without these partial rows,
   on the same data object so the parameters line up.

**Findings:**
- **Selection is real at the level of results.** Cars that went on to retire
  were running 0.57 ± 0.03 points slower than their effects predict. About 0.2
  points of that is a failing car limping through its final laps. Measured 3
  laps before the end, the gap is 0.35.
- **It doesn't matter for the effects.** Adding the retirees moves drivers by
  0.44 posterior sds on average (correlation 0.96) and car-seasons by 0.46
  (correlation 0.99). The moves are unrelated to how retirement-prone a driver
  or car is (correlations 0.03 and −0.20).

So the models fitted to the Ergast data (1950–2019) kept dropping retirements.
The Jolpica refit of 1950–2025 includes them, measured 3 laps before the car
stopped: they still leave the effects in place, and they make the pace model's
loss mixture less well determined ([Two modes in the loss mixture](@ref)).
