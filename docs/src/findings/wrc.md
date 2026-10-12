# Rallying (WRC)

The pace + loss model fitted to World Rally Championship stage times,
2018–2025, top two tiers (Rally1, including WRC cars, and Rally2, including R5).
Each special stage is a "race". Cars are grouped by manufacturer × tier ×
season. See [Data handling](@ref) for how rally data differs from F1.

```@raw html
<div id="manifest-wrc"></div>
<script>
fetch(documenterBaseURL + "/assets/data/wrc/manifest.json").then(r => r.json()).then(m => {
  document.getElementById("manifest-wrc").innerHTML =
    `<p><small>Model <code>${m.model}</code>. Data: ${m.data}. Convergence: ${m.convergence.replace(/(https:\S+)/, '<a href="$1">PR #33</a>')}.
     Exported ${m.date}.</small></p>`;
});
</script>
```

```@raw html
<div class="admonition is-info"><header class="admonition-header">Relative to whom?</header>
<div class="admonition-body"><p>Driver effects are relative to the average of the
<span class="diomedes-stat" data-path="assets/data/wrc/drivers.csv" data-stat="count" data-format="int">…</span>
drivers in the data, most of them amateurs in Rally2 cars. That's why the fastest
driver sits
<span class="diomedes-stat" data-path="assets/data/wrc/drivers.csv" data-stat="min" data-x="mean" data-format="1">…</span>%
from the average. Stage results are much noisier than F1 races, and the model has
no road-order or surface terms
(<a href="https://github.com/aleccrowell/diomedes/pull/33">#33</a>).</p></div></div>
```

## Drivers

```@raw html
<div id="fig-wrc-drivers" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-wrc-drivers", {
  data: { url: "assets/data/wrc/drivers.csv",
          format: { type: "csv", parse: { mean: "number", sd: "number", q05: "number", q95: "number", stages: "number" } } },
  params: [
    { name: "top", value: 25, bind: { input: "range", min: 10, max: 60, step: 5, name: "Drivers shown " } },
    { name: "minstages", value: 300, bind: { input: "range", min: 20, max: 1000, step: 20, name: "Minimum stages " } }
  ],
  transform: [
    { filter: "datum.stages >= minstages" },
    { window: [{ op: "row_number", as: "rank" }], sort: [{ field: "mean", order: "ascending" }] },
    { filter: "datum.rank <= top" }
  ],
  width: "container", height: { step: 16 },
  encoding: {
    y: { field: "name", type: "nominal", sort: { field: "mean", order: "ascending" }, title: null },
    tooltip: [{ field: "name", title: "Driver" }, { field: "stages", title: "Stage times" },
              { field: "mean", title: "Effect (%)", format: ".2f" },
              { field: "q05", title: "5%", format: ".2f" }, { field: "q95", title: "95%", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "q05", type: "quantitative", title: "Driver effect, % of stage time (negative = faster)", scale: { zero: false } },
                  x2: { field: "q95" } } },
    { mark: { type: "point", filled: true, size: 45 }, encoding: { x: { field: "mean", type: "quantitative" } } }
  ]
});
</script>
```

## Cars by season

Manufacturer × tier effects within each season (negative = faster). Rally1
cars are about 4–5% faster than Rally2 cars, and the model separates the tiers'
car advantage from their drivers'.

```@raw html
<div id="fig-wrc-cars" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-wrc-cars", {
  data: { url: "assets/data/wrc/cars.csv",
          format: { type: "csv", parse: { season: "number", mean: "number", sd: "number", q05: "number", q95: "number",
                                          stage_times: "number" } } },
  transform: [{ filter: "datum.stage_times >= 50" }],
  width: "container", height: 360,
  encoding: {
    x: { field: "season", type: "ordinal", title: "Season" },
    y: { field: "mean", type: "quantitative", title: "Car effect within season, % (negative = faster)", scale: { reverse: true } },
    color: { field: "manufacturer", type: "nominal", title: "Manufacturer" },
    shape: { field: "tier", type: "nominal", title: "Tier" },
    tooltip: [{ field: "manufacturer", title: "Manufacturer" }, { field: "tier", title: "Tier" },
              { field: "season", title: "Season" }, { field: "stage_times", title: "Stage times" },
              { field: "mean", title: "Effect (%)", format: ".2f" },
              { field: "q05", title: "5%", format: ".2f" }, { field: "q95", title: "95%", format: ".2f" }]
  },
  mark: { type: "point", filled: true, size: 80 }
});
</script>
```

Levels with at least 50 stage times. The vertical axis is reversed, so faster is
higher.

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/wrc/drivers.csv"><code>drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/wrc/cars.csv"><code>cars.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
