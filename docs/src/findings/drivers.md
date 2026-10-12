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

## How sure is the ranking?

Intervals on effects don't directly answer "who is the fastest?". Ranking the
drivers with at least 20 results in each of the 4,000 posterior draws does: the
bars show how often each driver ranks first, in the top 3 and in the top 10.
Even the most likely fastest driver is first in only about a fifth of the
draws.

```@raw html
<div id="fig-driver-ranks" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-driver-ranks", {
  data: { url: "assets/data/race/ranks.csv",
          format: { type: "csv", parse: { rows: "number", p_top1: "number", p_top3: "number", p_top10: "number",
                                          rank_median: "number", rank_q05: "number", rank_q95: "number" } } },
  transform: [
    { window: [{ op: "row_number", as: "order" }], sort: [{ field: "p_top10", order: "descending" }] },
    { filter: "datum.order <= 25" },
    { fold: ["p_top10", "p_top3", "p_top1"], as: ["level", "p"] },
    { calculate: "{p_top1: 'fastest', p_top3: 'top 3', p_top10: 'top 10'}[datum.level]", as: "which" }
  ],
  width: "container", height: { step: 16 },
  mark: { type: "bar", height: { band: 0.8 } },
  encoding: {
    y: { field: "name", type: "nominal", sort: { field: "order", op: "min", order: "ascending" }, title: null,
         axis: { labelLimit: 200 } },
    x: { field: "p", type: "quantitative", title: "Posterior probability", axis: { format: ".0%" },
         scale: { domain: [0, 1] }, stack: null },
    color: { field: "which", type: "nominal", title: "Rank",
             scale: { domain: ["top 10", "top 3", "fastest"], range: ["#c6dbef", "#6baed6", "#08519c"] } },
    order: { field: "p", sort: "descending" },
    tooltip: [{ field: "name", title: "Driver" }, { field: "rows", title: "Results" },
              { field: "which", title: "Rank" }, { field: "p", title: "Probability", format: ".1%" },
              { field: "rank_median", title: "Median rank" }, { field: "rank_q05", title: "Rank 5%" },
              { field: "rank_q95", title: "Rank 95%" }]
  }
});
</script>
```

```@raw html
<p>The 25 drivers most often in the top 10 are shown. All drivers with at least 20
results are in <a class="diomedes-data" data-path="assets/data/race/ranks.csv"><code>ranks.csv</code></a>.</p>
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
<p>The tables behind these charts are
<a class="diomedes-data" data-path="assets/data/race/drivers.csv"><code>drivers.csv</code></a> and
<a class="diomedes-data" data-path="assets/data/race/ranks.csv"><code>ranks.csv</code></a>
(columns described on the <a href="../data/">data page</a>).</p>
```

It has one row per
driver with the posterior mean, sd and 90% interval of the effect, the number
of results, the first and last season, and the career slope (% per decade of
age, relative to the average driver's trajectory).
