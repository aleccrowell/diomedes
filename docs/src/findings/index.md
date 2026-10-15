# Findings

Results of the best race-time model so far, fitted to every classified F1
finish from 1950 to 2025, from the Jolpica API. Lapped cars enter as intervals,
or at their exact gap where their own time is recorded (2023 on), and
retirements from 1996 as partial gaps.

```@raw html
<div id="manifest-table"></div>
<script>
fetch(documenterBaseURL + "/assets/data/race/manifest.json").then(r => r.json()).then(m => {
  const rows = m.rows;
  const extra = [rows.lapped_exact ? `${rows.lapped_exact} lapped at exact gaps` : "",
                 rows.partial ? `${rows.partial} retirements as partial gaps` : ""].filter(Boolean).join(", ");
  const groups = m.convergence.groups || {};
  const grow = Object.entries(groups).map(([k, v]) =>
    `<tr><td>${k}</td><td>${v.max_rhat}</td><td>${v.min_ess}</td><td><code>${v.worst}</code></td></tr>`).join("");
  const multi = m.modes && m.modes.length > 1;
  const modes = multi ? m.modes.map(x => `${x.label} (chains ${x.chains.join(", ")})`).join(", ") : "";
  document.getElementById("manifest-table").innerHTML =
    `<table><tbody>
       <tr><th>Model</th><td><code>${m.model}</code></td></tr>
       <tr><th>Data</th><td>${m.data}: ${rows.timed} timed and ${rows.lapped} lapped results${extra ? ", " + extra : ""}</td></tr>
       <tr><th>Posterior</th><td>${m.chains} chains × ${m.draws_per_chain} draws${m.chains_fitted && m.chains_fitted > m.chains ? ` (of ${m.chains_fitted} fitted)` : ""}, ${m.convergence.divergences} divergence(s)</td></tr>
       <tr><th>Convergence</th><td>identified quantities (R-hat, min bulk ESS, worst level):
         <table><thead><tr><th>group</th><th>R-hat</th><th>ESS</th><th>worst</th></tr></thead><tbody>${grow}</tbody></table></td></tr>
       ${multi ? `<tr><th>Modes</th><td>${modes}</td></tr>` : ""}
       <tr><th>Exported</th><td>${m.date} by <code>${m.exported_by}</code> at commit <code>${m.export_commit}</code></td></tr>
     </tbody></table>` +
    (multi ? `<div class="admonition is-info"><header class="admonition-header">The chains found ${m.modes.length} modes</header>
      <div class="admonition-body"><p>The chains agree on the driver and car effects, but settled in
      ${m.modes.length} different splits between routine and big losses. Mode
      ${m.modes.filter(x => x.kept).map(x => x.label).join(", ")} has much higher posterior density, so these pages use
      chains ${m.chains_used.join(", ")} of ${m.chains_fitted}; the convergence above is over those. The other mode is
      reported for comparison (<a href="eras/">Eras</a>,
      <a href="../methodology/noise/#Two-modes-in-the-loss-mixture">Two modes in the loss mixture</a>).</p></div></div>` : "");
});
</script>
```

The model is the pace + loss model with race-duration and random-walk era terms
for incidents, a pace scale by season, a population age curve, per-driver
career slopes, big incident losses and in-season car development (see
[Which pace model to use](@ref)). The common race level is learned (the default
since [#36](https://github.com/aleccrowell/diomedes/issues/36)). Earlier exports of the 1950–2019 data used the fixed race-intercept
prior (`_fgam`), which gave the same out-of-sample fit.

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
- [Driver vs car](explorer.md): every driver-season split into the driver's
  part and the car's part.
- [Cars](cars.md): each season's field, dominant cars, and in-season
  development.
- [Careers](careers.md): the shape of a career, and drivers whose
  trajectories differed most.
- [Eras](eras.md): how spread out results were, and time lost to
  incidents, season by season.
- [Retirements](retirements.md): retirement rates by cause and season,
  the most and least reliable cars, and incident-proneness.
- [Lap times](laps.md): supporting evidence from lap-by-lap pace,
  1996–2025. It checks the race model's car effects and in-season development,
  and isn't a second driver ranking.
- [Rallying (WRC)](wrc.md): drivers and cars from WRC stage
  times, 2018–2025.
- [Data downloads](data.md): every table behind the charts, with its columns
  and units.
