# Retirements

Results of the competing-risks retirement model (see
[Retirements and selection bias](@ref)), fitted to every F1 start from 1950 to
2019. Effects are on the **log-hazard scale**: +0.5 means a retirement rate
1.65× the season's average, −0.5 about 0.6×.

```@raw html
<div id="manifest-retire"></div>
<script>
fetch(documenterBaseURL + "/assets/data/retire/manifest.json").then(r => r.json()).then(m => {
  document.getElementById("manifest-retire").innerHTML =
    `<p><small>Data: ${m.data}. Convergence: max R-hat ${m.convergence.max_rhat}, min bulk ESS ${m.convergence.min_ess},
     ${m.convergence.divergences} divergence(s). Exported ${m.date} (commit <code>${m.export_commit}</code>).</small></p>`;
});
</script>
```

## Retirements by season and cause

The model's expected retirements per start over a race distance (lines), and
the observed rates (points), by cause. Collisions were rarely recorded as a
cause before about 1990, and the season-level random walks absorb that.

```@raw html
<div id="fig-retire-seasons" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-retire-seasons", {
  data: { url: "assets/data/retire/seasons.csv",
          format: { type: "csv", parse: { season: "number", mech: "number", mech_obs: "number", incident: "number",
                                          incident_obs: "number", collision: "number", collision_obs: "number" } } },
  transform: [
    { fold: ["mech", "incident", "collision", "mech_obs", "incident_obs", "collision_obs"] },
    { calculate: "indexof(datum.key, '_obs') >= 0 ? 'observed' : 'model'", as: "kind" },
    { calculate: "{mech: 'mechanical', incident: 'incident', collision: 'collision'}[replace(datum.key, '_obs', '')]", as: "cause" }
  ],
  width: "container", height: 300,
  encoding: {
    x: { field: "season", type: "quantitative", title: "Season", axis: { format: "d" }, scale: { zero: false } },
    y: { field: "value", type: "quantitative", title: "Retirements per start" },
    color: { field: "cause", type: "nominal", title: "Cause", sort: ["mechanical", "incident", "collision"] },
    tooltip: [{ field: "season", title: "Season" }, { field: "cause", title: "Cause" }, { field: "kind", title: "" },
              { field: "value", title: "Per start", format: ".3f" }]
  },
  layer: [
    { transform: [{ filter: "datum.kind == 'observed'" }], mark: { type: "point", opacity: 0.45, size: 18 } },
    { transform: [{ filter: "datum.kind == 'model'" }], mark: { type: "line", strokeWidth: 2 } }
  ]
});
</script>
```

## Reliability

Car-season mechanical reliability relative to its season (negative = more
reliable), for car-seasons with at least 20 starts: the 15 most and 15 least
reliable. Intervals are ±1.645 sd.

```@raw html
<div id="fig-retire-cars" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-retire-cars", {
  data: { url: "assets/data/retire/cars.csv",
          format: { type: "csv", parse: { season: "number", effect: "number", sd: "number", starts: "number", mech_dnf: "number" } } },
  transform: [
    { filter: "datum.starts >= 20" },
    { calculate: "datum.id", as: "car" },
    { calculate: "datum.effect - 1.645 * datum.sd", as: "lo" },
    { calculate: "datum.effect + 1.645 * datum.sd", as: "hi" },
    { window: [{ op: "row_number", as: "best" }], sort: [{ field: "effect", order: "ascending" }] },
    { window: [{ op: "row_number", as: "worst" }], sort: [{ field: "effect", order: "descending" }] },
    { filter: "datum.best <= 15 || datum.worst <= 15" },
    { calculate: "datum.effect < 0 ? 'more reliable' : 'less reliable'", as: "group" }
  ],
  width: "container", height: { step: 14 },
  encoding: {
    y: { field: "car", type: "nominal", sort: { field: "effect", order: "ascending" }, title: null },
    color: { field: "group", type: "nominal", title: null,
             scale: { domain: ["more reliable", "less reliable"], range: ["#4c78a8", "#e45756"] } },
    tooltip: [{ field: "car", title: "Car" }, { field: "starts", title: "Starts" },
              { field: "mech_dnf", title: "Mechanical retirements" },
              { field: "effect", title: "Log-hazard effect", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative", title: "Mechanical failure log-hazard, relative to season" }, x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 45 }, encoding: { x: { field: "effect", type: "quantitative" } } }
  ]
});
</script>
```

## Incident-proneness

Driver incident-proneness (solo incidents, and collisions with loading
β_col ≈ 0.36), for drivers with at least 50 starts: the 15 least and 15 most
incident-prone.

```@raw html
<div id="fig-retire-drivers" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-retire-drivers", {
  data: { url: "assets/data/retire/drivers.csv",
          format: { type: "csv", parse: { effect: "number", sd: "number", starts: "number", incidents: "number", collisions: "number" } } },
  transform: [
    { filter: "datum.starts >= 50" },
    { calculate: "datum.effect - 1.645 * datum.sd", as: "lo" },
    { calculate: "datum.effect + 1.645 * datum.sd", as: "hi" },
    { window: [{ op: "row_number", as: "best" }], sort: [{ field: "effect", order: "ascending" }] },
    { window: [{ op: "row_number", as: "worst" }], sort: [{ field: "effect", order: "descending" }] },
    { filter: "datum.best <= 15 || datum.worst <= 15" },
    { calculate: "datum.effect < 0 ? 'fewer incidents' : 'more incidents'", as: "group" }
  ],
  width: "container", height: { step: 14 },
  encoding: {
    y: { field: "name", type: "nominal", sort: { field: "effect", order: "ascending" }, title: null },
    color: { field: "group", type: "nominal", title: null,
             scale: { domain: ["fewer incidents", "more incidents"], range: ["#4c78a8", "#e45756"] } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "starts", title: "Starts" },
              { field: "incidents", title: "Solo incidents" }, { field: "collisions", title: "Collisions" },
              { field: "effect", title: "Log-hazard effect", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative", title: "Incident log-hazard, relative to the average driver" }, x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 45 }, encoding: { x: { field: "effect", type: "quantitative" } } }
  ]
});
</script>
```

Slower drivers tend to be more incident-prone (correlation 0.42 with the race
model's driver effects), and slower car-seasons less reliable (0.46)
([#42](https://github.com/aleccrowell/diomedes/pull/42)).

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/retire/seasons.csv"><code>seasons.csv</code></a>, <a class="diomedes-data" data-path="assets/data/retire/cars.csv"><code>cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/retire/drivers.csv"><code>drivers.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
