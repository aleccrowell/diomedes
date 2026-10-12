# Cars

Car-season effects from the best race-time model. Each is relative to the
other cars **in the same season**: the model can't compare cars across seasons
(see [The question and what is identifiable](@ref)). Negative is faster.

## One season's field

Choose a season to see its cars with 90% intervals.

```@raw html
<div id="fig-car-season" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-car-season", {
  data: { url: "assets/data/race/cars.csv",
          format: { type: "csv", parse: { season: "number", rows: "number", mean: "number", sd: "number",
                                          q05: "number", q95: "number", dev_mean: "number", dev_sd: "number" } } },
  params: [{ name: "chosen", value: 1988,
             bind: { input: "range", min: 1950, max: 2019, step: 1, name: "Season " } }],
  // "constructor" can't be a sort field (it collides with Object.prototype.constructor
  // in Vega's parser), so the chart uses a copy named "team"
  transform: [{ filter: "datum.season == chosen" }, { calculate: "datum.constructor", as: "team" }],
  width: "container",
  height: { step: 18 },
  encoding: {
    y: { field: "team", type: "nominal", sort: { field: "mean", order: "ascending" }, title: null },
    tooltip: [
      { field: "team", title: "Constructor" }, { field: "season", title: "Season" },
      { field: "rows", title: "Results" },
      { field: "mean", title: "Effect (%)", format: ".2f" },
      { field: "q05", title: "5%", format: ".2f" }, { field: "q95", title: "95%", format: ".2f" },
      { field: "dev_mean", title: "Development trend (% per season)", format: ".2f" }
    ]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "q05", type: "quantitative", title: "Car effect within season, % of race time (negative = faster)" },
                  x2: { field: "q95" } } },
    { mark: { type: "point", filled: true, size: 50 },
      encoding: { x: { field: "mean", type: "quantitative" } } }
  ]
});
</script>
```

## Dominance: the fastest car's margin

For each season, the gap between the fastest and second-fastest car (posterior
means). A large margin is a dominant car. The margins are in each season's own
units, so compare their shape more than their exact size across eras.

```@raw html
<div id="fig-car-margin" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-car-margin", {
  data: { url: "assets/data/race/cars.csv",
          format: { type: "csv", parse: { season: "number", rows: "number", mean: "number" } } },
  transform: [
    { filter: "datum.rows >= 10" },
    { window: [{ op: "row_number", as: "rank" }], groupby: ["season"], sort: [{ field: "mean", order: "ascending" }] },
    { filter: "datum.rank <= 2" },
    { pivot: "rank", value: "mean", groupby: ["season"] },
    { calculate: "datum['2'] - datum['1']", as: "margin" }
  ],
  width: "container", height: 260,
  layer: [
    { mark: { type: "bar", opacity: 0.85 },
      encoding: { x: { field: "season", type: "ordinal", title: "Season",
                       axis: { values: [1950, 1960, 1970, 1980, 1990, 2000, 2010] } },
                  y: { field: "margin", type: "quantitative", title: "Fastest car's margin over the next (%)" },
                  tooltip: [{ field: "season", title: "Season" }, { field: "margin", title: "Margin (%)", format: ".2f" }] } }
  ]
});
</script>
```

Cars with fewer than 10 results in a season are left out of this chart.

## In-season development

How each car's form changed **during** its season, relative to the field: the
change in its effect from the first round to the last. Negative means it got
faster than the field did. The 20 largest trends in each direction (cars with at
least 20 results):

```@raw html
<div id="fig-car-dev" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-car-dev", {
  data: { url: "assets/data/race/cars.csv",
          format: { type: "csv", parse: { season: "number", rows: "number", dev_mean: "number", dev_sd: "number" } } },
  transform: [
    { filter: "datum.rows >= 20" },
    { calculate: "datum.constructor + ' ' + datum.season", as: "car" },
    { calculate: "datum.dev_mean - 1.645 * datum.dev_sd", as: "lo" },
    { calculate: "datum.dev_mean + 1.645 * datum.dev_sd", as: "hi" },
    { window: [{ op: "row_number", as: "up" }], sort: [{ field: "dev_mean", order: "ascending" }] },
    { window: [{ op: "row_number", as: "down" }], sort: [{ field: "dev_mean", order: "descending" }] },
    { filter: "datum.up <= 20 || datum.down <= 20" },
    { calculate: "datum.dev_mean < 0 ? 'improved' : 'faded'", as: "direction" }
  ],
  width: "container", height: { step: 14 },
  encoding: {
    y: { field: "car", type: "nominal", sort: { field: "dev_mean", order: "ascending" }, title: null },
    color: { field: "direction", type: "nominal", title: null,
             scale: { domain: ["improved", "faded"], range: ["#4c78a8", "#e45756"] } },
    tooltip: [{ field: "car", title: "Car" }, { field: "rows", title: "Results" },
              { field: "dev_mean", title: "Trend (% per season)", format: ".2f" },
              { field: "dev_sd", title: "sd", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative", title: "Development trend, % per season (negative = improved)" },
                  x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 45 }, encoding: { x: { field: "dev_mean", type: "quantitative" } } }
  ]
});
</script>
```

Intervals are ±1.645 sd (about 90%). The spread of trends is about 0.5% per
season ([#27](https://github.com/aleccrowell/diomedes/issues/27)). Several of the
largest match known histories, such as Force India's 2015 B-spec car and
Renault's strong start and fade in 2011.

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/race/cars.csv"><code>cars.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
