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
  effects ([#66](https://github.com/aleccrowell/diomedes/issues/66));
- a population age curve shared by all drivers, as in the race model
  ([#12](https://github.com/aleccrowell/diomedes/issues/12));
- car-season slopes on three circuit dimensions: average lap speed, street score
  and race-time air temperature ([#14](https://github.com/aleccrowell/diomedes/issues/14)).

It is **in development** ([#12](https://github.com/aleccrowell/diomedes/issues/12)).
Its four chains converge to a single mode ([below](#Convergence)).

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

## The shape of a career on laps

The population age curve on clean laps: how pace changes with age for the
average driver, in % of lap time (negative = faster), with a 90% band. As in the
race model ([Careers](careers.md)), only its **shape** is identified: it has no
constant or linear part, because those can't be told apart from the drivers'
career slopes and the car-seasons. Read the bends, not the slope. The curve
flattens through the mid-twenties, bottoms out around 27–29 and bends back up
in the early thirties. Ages under 20 and over 38 have few laps.

```@raw html
<div id="fig-laps-age" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-age", {
  data: { url: "assets/data/laps/age_curve.csv",
          format: { type: "csv", parse: { age: "number", laps: "number", mean: "number", q05: "number", q95: "number" } } },
  width: "container", height: 300,
  encoding: { x: { field: "age", type: "quantitative", title: "Age", scale: { zero: false } } },
  layer: [
    { mark: { type: "area", opacity: 0.25 },
      encoding: { y: { field: "q05", type: "quantitative", title: "Pace relative to the curve's average (%)", scale: { reverse: true } },
                  y2: { field: "q95" } } },
    { mark: { type: "line", point: true },
      encoding: { y: { field: "mean", type: "quantitative" },
                  tooltip: [{ field: "age", title: "Age" }, { field: "laps", title: "Laps at this age" },
                            { field: "mean", title: "Curve (%)", format: ".2f" },
                            { field: "q05", title: "5%", format: ".2f" }, { field: "q95", title: "95%", format: ".2f" }] } }
  ]
});
</script>
```

The vertical axis is reversed, so faster is higher.

## Circuit slopes

Each car-season has a slope on three circuit dimensions, centred within the
season: how much faster or slower it is, relative to the field, at a circuit one
standard deviation above average. The dimensions are average lap speed (sd
20 km/h), street score (sd 0.34 on the 0–1 scale; see the
[circuit tables](https://github.com/aleccrowell/diomedes/tree/master/reference/circuits))
and air temperature at race time (sd 4.8 °C). Positive means relatively
**slower** there. The spread of the slopes across car-seasons is about 0.16 %
per sd for speed and street score, and 0.11 % for temperature.

```@raw html
<div id="fig-laps-circuits" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-circuits", {
  data: { url: "assets/data/laps/cars.csv",
          format: { type: "csv", parse: { laps: "number", speed_mean: "number", street_mean: "number", temp_mean: "number" } } },
  transform: [
    { filter: "datum.laps >= 500" },
    { fold: ["speed_mean", "street_mean", "temp_mean"], as: ["dimension", "slope"] },
    { calculate: "{speed_mean: 'Speed', street_mean: 'Street score', temp_mean: 'Temperature'}[datum.dimension]", as: "dim" }
  ],
  width: "container", height: 220,
  mark: { type: "tick", opacity: 0.6, thickness: 2 },
  encoding: {
    x: { field: "slope", type: "quantitative", title: "Slope, % of lap time per sd (positive = relatively slower)" },
    y: { field: "dim", type: "nominal", title: null },
    tooltip: [{ field: "id", title: "Car-season" }, { field: "dim", title: "Dimension" },
              { field: "slope", title: "Slope (%)", format: ".2f" }, { field: "laps", title: "Laps" }]
  }
});
</script>
```

Car-seasons with at least 500 laps.

## Convergence

All four chains converge to one mode. The fit before this one, without the age
curve and circuit slopes, split into two modes that differed in the spread of the
per-race trends (τ<sub>δ</sub> 4.60 against 4.12) and in the lap-noise mixture,
without changing the effects. Here τ<sub>δ</sub> = 4.58 in every chain. Three
chains adapted step sizes that left them taking about four times as many
leapfrog steps per draw as the fourth (tree depth 9 against 7). They were resumed from their first block at a fixed step size of
0.0272, the step size the fourth chain had tuned to
([#12](https://github.com/aleccrowell/diomedes/issues/12)). The table gives the
chains' scalars.

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
<p><a class="diomedes-data" data-path="assets/data/laps/vs_race_cars.csv"><code>vs_race_cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/vs_race_drivers.csv"><code>vs_race_drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/drivers.csv"><code>drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/cars.csv"><code>cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/age_curve.csv"><code>age_curve.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/modes.csv"><code>modes.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
