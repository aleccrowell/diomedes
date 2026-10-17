# Lap times

Results of the lap-time model (see [The lap-time model](@ref)), fitted to
green-flag laps from 1996 to 2025 (Jolpica lap times, pit laps inferred from the
lap times). Besides driver and car effects it has:
- a career slope per driver (season-level age);
- an in-season development trend per car-season;
- separate stint-age curves for the refuelling era (1996–2009) and since 2010;
- driver and car-season deviations from those curves (tyre management);
- an effect of leading the race;
- lap 1 as its own term, with the grid slot and driver and car-season start
  effects ([#66](https://github.com/aleccrowell/diomedes/issues/66)).

It is **in development** ([#12](https://github.com/aleccrowell/diomedes/issues/12)).
Its chains settle in two modes, which differ in a few scalars but not in the
driver and car results ([below](#Two-modes)).

!!! note "What these numbers measure"
    The headline results are the race model's: it measures the outcome that
    counts, the time to finish a race. The lap model measures something narrower:
    typical green-flag pace under race conditions, plus the start (lap 1). That
    includes traffic, blue flags and the tyre, fuel and engine management drivers
    do when they don't need to push, but not pit-stop timing, strategy or
    safety-car periods. So it **underrates drivers who win comfortably** (they
    manage their pace) and flatters drivers who lose most of their time in the
    parts of a race it leaves out. Read it as supporting evidence. Its car-season effects check the race
    model's split between car and driver, from largely independent data. Its
    in-season development trends are measured race by race. Don't read it as a
    second driver ranking.

```@raw html
<div id="manifest-laps"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps/manifest.json").then(r => r.json()).then(m => {
  document.getElementById("manifest-laps").innerHTML =
    `<p><small>Fit <code>${m.fit}</code>. Data: ${m.data}. Convergence: max R-hat ${m.convergence.max_rhat},
     min bulk ESS ${m.convergence.min_ess}, ${m.convergence.divergences} divergence(s). Compared with the race model
     <code>${m.race_model}</code>. Exported ${m.date}.</small></p>`;
});
</script>
```

## Lap model against the race model: cars

```@raw html
<p>Each point is a car-season, placed by its effect in the two models. They agree
closely (correlation
<span class="diomedes-stat" data-path="assets/data/laps/vs_race_cars.csv" data-stat="corr" data-x="race_mean" data-y="lap_mean">…</span>
over <span class="diomedes-stat" data-path="assets/data/laps/vs_race_cars.csv" data-stat="count" data-format="int">…</span>
car-seasons). The lap model's spread is larger, because clean laps leave out the
safety-car bunching and strategy noise that compress race time gaps.</p>
```

```@raw html
<div id="fig-laps-cars" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-cars", {
  data: { url: "assets/data/laps/vs_race_cars.csv",
          format: { type: "csv", parse: { laps: "number", lap_mean: "number", lap_sd: "number", season: "number",
                                          race_mean: "number", race_sd: "number" } } },
  // "constructor" can't be used as a field name in Vega (Object.prototype); use a copy
  transform: [{ calculate: "datum.constructor", as: "team" }],
  params: [{ name: "zoom", select: "interval", bind: "scales" }],
  width: "container", height: 420,
  mark: { type: "circle", size: 70, opacity: 0.8 },
  encoding: {
    x: { field: "race_mean", type: "quantitative", title: "Race model: car effect (%)" },
    y: { field: "lap_mean", type: "quantitative", title: "Lap model: car effect (%)" },
    // a gradient legend: 30 seasons as separate entries would run past the plot's height
    color: { field: "season", type: "quantitative", title: "Season", scale: { scheme: "viridis" },
             legend: { format: "d", gradientLength: 300 } },
    tooltip: [{ field: "team", title: "Constructor" }, { field: "season", title: "Season" },
              { field: "race_mean", title: "Race model (%)", format: ".2f" },
              { field: "lap_mean", title: "Lap model (%)", format: ".2f" }, { field: "laps", title: "Laps" }]
  }
});
</script>
```

## Lap model against the race model: drivers

```@raw html
<p>Drivers agree less (correlation
<span class="diomedes-stat" data-path="assets/data/laps/vs_race_drivers.csv" data-stat="corr" data-x="race_mean" data-y="lap_mean" data-filter="laps>=500">…</span>
over the <span class="diomedes-stat" data-path="assets/data/laps/vs_race_drivers.csv" data-stat="count" data-filter="laps>=500" data-format="int">…</span>
drivers with at least 500 clean laps), and the lap model's driver effects are
less spread out. Race gaps include what the lap model leaves out: pit stops,
strategy, safety-car periods and the laps lost to incidents. So back-markers look
faster on laps than in races, and the front-runners less dominant. Adding starts,
lead laps and tyre management did not change that
(<a href="https://github.com/aleccrowell/diomedes/issues/66">#66</a>). The race model's effects also cover whole
careers from 1950, while the lap model starts in 1996: a career that began
earlier enters here only at its end.</p>
<p>Treat the drivers of 1996–97 with care. Ukyo Katayama, for example, is among
the fastest on clean laps from 1,125 laps in 1996–97, but slow in the race
model. At the edge of the data, a short career, its slope and the few team-mates
linking its cars are hard to separate
(<a href="https://github.com/aleccrowell/diomedes/issues/12">#12</a>).</p>
```

```@raw html
<div id="fig-laps-drivers" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-drivers", {
  data: { url: "assets/data/laps/vs_race_drivers.csv",
          format: { type: "csv", parse: { laps: "number", lap_mean: "number", lap_sd: "number",
                                          race_mean: "number", race_sd: "number" } } },
  transform: [{ filter: "datum.laps >= 500" }],
  params: [{ name: "zoom", select: "interval", bind: "scales" }],
  width: "container", height: 420,
  mark: { type: "circle", opacity: 0.8 },
  encoding: {
    x: { field: "race_mean", type: "quantitative", title: "Race model: driver effect, whole career (%)" },
    y: { field: "lap_mean", type: "quantitative", title: "Lap model: driver effect, 1996–2025 (%)" },
    size: { field: "laps", type: "quantitative", title: "Laps", scale: { range: [20, 300] } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "laps", title: "Laps" },
              { field: "race_mean", title: "Race model (%)", format: ".2f" },
              { field: "lap_mean", title: "Lap model (%)", format: ".2f" }]
  }
});
</script>
```

Drivers with at least 500 clean laps.

## Two modes

The four chains settle in two modes during warm-up and stay there. They differ
in how the variation in lap times is shared out. In one mode the per-race trend
over the race distance varies more from race to race. In the other, more of the
variation goes into the lap-noise mixture. Both modes fit the data equally well
(the same log density), so the published results pool all four chains. The
driver, car, tyre-management and start effects agree across the two modes, so
which mode is right doesn't change them. As with the race model's
modes ([Checking for multiple modes](@ref)), the scalars that split the modes are
not converged when the chains are pooled. Within each mode they are (R-hat ≤ 1.012),
as are the cars, tyre management and starts. The driver effects and career slopes
are short of the targets: R-hat 1.04 pooled over the four chains (bulk ESS 119),
up to 1.07 within a mode, against 1.01 for the previous fit without the #66 terms.

```@raw html
<div id="laps-modes"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps/manifest.json").then(r => r.json()).then(m => {
  if (!m.modes) return;
  const f = (x, d) => (x === null || x === undefined) ? "–" : Number(x).toFixed(d);
  let h = "<table><thead><tr><th>Mode</th><th>Chains</th><th>τ<sub>δ</sub></th><th>Lap noise σ (%)</th>" +
          "<th>Lost-time lap probability</th><th>Lost-time lap mean (%)</th><th>Mean log density</th>" +
          "<th>Max R-hat within mode</th></tr></thead><tbody>";
  for (const md of m.modes) {
    h += `<tr><td>${md.label}</td><td>${md.chains.join(", ")}</td><td>${f(md.scalars.tau_delta, 2)}</td>` +
         `<td>${f(md.scalars.sigma, 3)}</td><td>${f(md.scalars.pi_base, 3)}</td><td>${f(md.scalars.lambda_base, 2)}</td>` +
         `<td>${f(md.mean_logjoint, 0)}</td><td>${md.convergence ? f(md.convergence.max_rhat, 3) : "–"}</td></tr>`;
  }
  h += "</tbody></table>";
  const names = { drivers: "Drivers", cars: "Car-seasons", tyremgmt_drivers: "Driver tyre management",
                  tyremgmt_cars: "Car tyre management", starts_drivers: "Driver starts", starts_cars: "Car starts" };
  for (const e of (m.effects_by_mode || [])) {
    h += `<p>Effects, mode ${e.modes}:</p><table><thead><tr><th></th><th>n</th><th>Correlation</th>` +
         "<th>Largest difference</th><th>Median difference / sd</th><th>Largest difference / sd</th></tr></thead><tbody>";
    for (const k of Object.keys(names)) {
      const a = e[k]; if (!a) continue;
      h += `<tr><td>${names[k]}</td><td>${a.n}</td><td>${f(a.corr, 4)}</td><td>${f(a.max_diff, 3)}</td>` +
           `<td>${f(a.median_diff_sd, 2)}</td><td>${f(a.max_diff_sd, 2)}</td></tr>`;
    }
    h += "</tbody></table>";
  }
  document.getElementById("laps-modes").innerHTML = h;
});
</script>
```

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/laps/vs_race_cars.csv"><code>vs_race_cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/vs_race_drivers.csv"><code>vs_race_drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/drivers.csv"><code>drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/cars.csv"><code>cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/modes.csv"><code>modes.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
