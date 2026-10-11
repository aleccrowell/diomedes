# Lap times

Results of the lap-time model (see [The lap-time model](@ref)), fitted to
clean green-flag laps from 2011 to 2019. It is **in development**
([#12](https://github.com/aleccrowell/diomedes/issues/12)); these results cover
only the seasons with recorded pit stops.

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

Each point is a car-season, placed by its effect in the two models. They agree
closely (correlation 0.97). The lap model's spread is larger, because clean
laps leave out the safety-car bunching and strategy noise that compress race
time gaps.

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
  width: 480, height: 420,
  mark: { type: "circle", size: 70, opacity: 0.8 },
  encoding: {
    x: { field: "race_mean", type: "quantitative", title: "Race model: car effect (%)" },
    y: { field: "lap_mean", type: "quantitative", title: "Lap model: car effect (%)" },
    color: { field: "season", type: "ordinal", title: "Season", scale: { scheme: "viridis" } },
    tooltip: [{ field: "team", title: "Constructor" }, { field: "season", title: "Season" },
              { field: "race_mean", title: "Race model (%)", format: ".2f" },
              { field: "lap_mean", title: "Lap model (%)", format: ".2f" }, { field: "laps", title: "Laps" }]
  }
});
</script>
```

## Lap model against the race model: drivers

Drivers agree less (correlation 0.70). The race model's driver effects cover a
whole career, with an age curve and career slopes, while the lap model sees
only 2011–2019. A driver whose best years were before 2011 is fast in the race
model and ordinary here: Michael Schumacher's 2010–12 comeback, for example.

```@raw html
<div id="fig-laps-drivers" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-drivers", {
  data: { url: "assets/data/laps/vs_race_drivers.csv",
          format: { type: "csv", parse: { laps: "number", lap_mean: "number", lap_sd: "number",
                                          race_mean: "number", race_sd: "number" } } },
  transform: [{ filter: "datum.laps >= 500" }],
  params: [{ name: "zoom", select: "interval", bind: "scales" }],
  width: 480, height: 420,
  mark: { type: "circle", opacity: 0.8 },
  encoding: {
    x: { field: "race_mean", type: "quantitative", title: "Race model: driver effect, whole career (%)" },
    y: { field: "lap_mean", type: "quantitative", title: "Lap model: driver effect, 2011–2019 (%)" },
    size: { field: "laps", type: "quantitative", title: "Laps", scale: { range: [20, 300] } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "laps", title: "Laps" },
              { field: "race_mean", title: "Race model (%)", format: ".2f" },
              { field: "lap_mean", title: "Lap model (%)", format: ".2f" }]
  }
});
</script>
```

Drivers with at least 500 clean laps.
