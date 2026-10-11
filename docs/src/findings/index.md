# Findings

Results of the best race-time model so far, fitted to every classified F1
finish from 1950 to 2019 (lapped cars as intervals).

```@raw html
<div id="manifest-table"></div>
<script>
fetch(documenterBaseURL + "/assets/data/race/manifest.json").then(r => r.json()).then(m => {
  document.getElementById("manifest-table").innerHTML =
    `<table><tbody>
       <tr><th>Model</th><td><code>${m.model}</code></td></tr>
       <tr><th>Data</th><td>${m.data}: ${m.rows.timed} timed and ${m.rows.lapped} lapped results</td></tr>
       <tr><th>Posterior</th><td>${m.chains} chains × ${m.draws_per_chain} draws; max R-hat ${m.convergence.max_rhat},
           min bulk ESS ${m.convergence.min_ess} (identified quantities), ${m.convergence.divergences} divergence(s)</td></tr>
       <tr><th>Exported</th><td>${m.date} by <code>${m.exported_by}</code> at commit <code>${m.export_commit}</code></td></tr>
     </tbody></table>`;
});
</script>
```

The model is the pace + loss model with race-duration and random-walk era terms
for incidents, a pace scale by season, a population age curve, per-driver
career slopes, big incident losses and in-season car development (see
[Which pace model to use](@ref)). It was fitted with the fixed race-intercept
prior (`_fgam`); the learned common level, now the default, gave the same
out-of-sample fit (#36).

## Reading the numbers

- **Units.** Effects are in % of race time, on a log scale. Negative is
  faster: a driver effect of −2 means about 2% quicker than the average driver
  in the data over a race distance, all else equal.
- **Drivers** are in average-season units: the pace scale κ stretches or
  shrinks every gap in a season, and driver effects are expressed at κ = 1.
- **Car-season effects** are relative to the other cars in the same season
  only. Comparing cars across seasons needs the drivers that link them, and
  isn't what these numbers measure.
- **Intervals** are 90% posterior intervals.

## Pages

- [Drivers](drivers.md): rankings with uncertainty, and every driver in one
  explorable chart.
- [Cars](cars.md): each season's field, dominant cars, and in-season
  development.
- [Careers](careers.md): the shape of a career, and drivers whose
  trajectories differed most.
- [Eras](eras.md): how spread out results were, and time lost to
  incidents, season by season.
- [Retirements](retirements.md): retirement rates by cause and season,
  the most and least reliable cars, and incident-proneness.
- [Lap times](laps.md): the lap-time model against the race model,
  2011–2019.
- [Rallying (WRC)](wrc.md): drivers and cars from WRC stage
  times, 2018–2025.
