# Drivers

Driver effects from the best race-time model (see [Findings](@ref) for what
was fitted and how to read the units). Negative is faster.

!!! warning "Comparing eras"
    Drivers from different decades are compared through chains of team-mates
    and shared fields, and the pace scale κ removes era-wide changes in how
    spread out results are. These links are real but long. Treat differences
    between eras as less certain than the intervals suggest, because they rest on
    the model's assumptions about how pace and incidents changed over time.

## Fastest drivers

Posterior means (points) and 90% intervals (lines). Use the sliders to change
how many drivers are shown and the minimum number of results a driver needs.
Drivers with few results get wide intervals and are shrunk towards the average.

```@raw html
<div id="fig-driver-ranking" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-driver-ranking", {
  data: { url: "assets/data/race/drivers.csv",
          format: { type: "csv", parse: { rows: "number", first_season: "number", last_season: "number",
                                          mean: "number", sd: "number", q05: "number", q95: "number" } } },
  params: [
    { name: "top", value: 30, bind: { input: "range", min: 10, max: 80, step: 5, name: "Drivers shown " } },
    { name: "minrows", value: 20, bind: { input: "range", min: 1, max: 100, step: 1, name: "Minimum results " } }
  ],
  transform: [
    { filter: "datum.rows >= minrows" },
    { window: [{ op: "row_number", as: "rank" }], sort: [{ field: "mean", order: "ascending" }] },
    { filter: "datum.rank <= top" },
    { calculate: "datum.name + ' (' + datum.first_season + '–' + datum.last_season + ')'", as: "label" }
  ],
  height: { step: 16 },
  width: "container",
  encoding: {
    y: { field: "label", type: "nominal", sort: { field: "mean", order: "ascending" }, title: null,
         axis: { labelLimit: 260 } },
    tooltip: [
      { field: "name", title: "Driver" },
      { field: "first_season", title: "First season" },
      { field: "last_season", title: "Last season" },
      { field: "rows", title: "Results" },
      { field: "mean", title: "Effect (%)", format: ".2f" },
      { field: "q05", title: "5%", format: ".2f" },
      { field: "q95", title: "95%", format: ".2f" }
    ]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "q05", type: "quantitative", title: "Driver effect, % of race time (negative = faster)" },
                  x2: { field: "q95" } } },
    { mark: { type: "point", filled: true, size: 45 },
      encoding: { x: { field: "mean", type: "quantitative" } } }
  ]
});
</script>
```

## Every driver

All 550 drivers with a classified finish, by the middle of their career. Point
size is the number of results. Drag to pan, scroll to zoom, and type a name to
highlight it.

```@raw html
<div id="fig-driver-scatter" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-driver-scatter", {
  data: { url: "assets/data/race/drivers.csv",
          format: { type: "csv", parse: { rows: "number", first_season: "number", last_season: "number",
                                          mean: "number", sd: "number", q05: "number", q95: "number" } } },
  params: [
    { name: "search", value: "", bind: { input: "text", placeholder: "e.g. Prost", name: "Highlight driver " } },
    { name: "zoom", select: "interval", bind: "scales" }
  ],
  transform: [
    { calculate: "(datum.first_season + datum.last_season) / 2", as: "mid" },
    { calculate: "search != '' && indexof(lower(datum.name), lower(search)) >= 0", as: "match" }
  ],
  width: "container",
  height: 420,
  mark: { type: "circle" },
  encoding: {
    x: { field: "mid", type: "quantitative", title: "Middle of career (season)", scale: { zero: false },
         axis: { format: "d" } },
    y: { field: "mean", type: "quantitative", title: "Driver effect, % of race time (negative = faster)",
         scale: { reverse: true } },
    size: { field: "rows", type: "quantitative", title: "Results", scale: { range: [10, 300] } },
    color: { condition: { test: "datum.match", value: "#e45756" }, value: "#4c78a8" },
    opacity: { condition: { test: "search == '' || datum.match", value: 0.75 }, value: 0.15 },
    tooltip: [
      { field: "name", title: "Driver" },
      { field: "first_season", title: "First season" },
      { field: "last_season", title: "Last season" },
      { field: "rows", title: "Results" },
      { field: "mean", title: "Effect (%)", format: ".2f" },
      { field: "sd", title: "sd", format: ".2f" }
    ]
  }
});
</script>
```

The vertical axis is reversed, so faster drivers are higher.

## Download the data

```@raw html
<p>The table behind these charts is
<a class="diomedes-data" data-path="assets/data/race/drivers.csv"><code>drivers.csv</code></a>
(columns described on the <a href="../data/">data page</a>).</p>
```

It has one row per
driver with the posterior mean, sd and 90% interval of the effect, the number
of results, the first and last season, and the career slope (% per decade of
age, relative to the average driver's trajectory).
