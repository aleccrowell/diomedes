# Cars

Car-season effects from the best race-time model. Each is relative to the
other cars **in the same season**: the model can't compare cars across seasons
(see [The question and what is identifiable](@ref)). Negative is faster.

## One season's field

Choose a season to see its cars with 90% intervals. Each effect is the sum of
a chassis part, an engine part and a residual (see
[Chassis and engines](@ref)); the tooltip splits it.

```@raw html
<div id="fig-car-season" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-car-season", {
  data: { url: "assets/data/race/cars.csv",
          format: { type: "csv", parse: { season: "number", rows: "number", mean: "number", sd: "number",
                                          q05: "number", q95: "number", dev_mean: "number", dev_sd: "number",
                                          speed_mean: "number", rough_mean: "number", temp_mean: "number",
                                          chassis_mean: "number", engine_mean: "number", residual_mean: "number" } } },
  params: [{ name: "chosen", value: 1988,
             bind: { input: "range", min: 1950, max: 2025, step: 1, name: "Season " } }],
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
      { field: "chassis_mean", title: "Chassis part (%)", format: ".2f" },
      { field: "engine_mean", title: "Engine part (%)", format: ".2f" },
      { field: "residual_mean", title: "Residual (%)", format: ".2f" },
      { field: "dev_mean", title: "Development trend (% per season)", format: ".2f" },
      { field: "speed_mean", title: "Speed slope (% per sd)", format: ".2f" },
      { field: "rough_mean", title: "Roughness slope (% per sd)", format: ".2f" },
      { field: "temp_mean", title: "Temperature slope (% per sd)", format: ".2f" }
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

## Circuit slopes

Which conditions suited a car, relative to its field: each car-season has a
slope on three properties of the races it ran
([#81](https://github.com/aleccrowell/diomedes/issues/81)):

- **average speed** (layout length × laps ÷ the winner's time, averaged over the
  circuit's dry races; one sd is 26 km/h);
- **roughness**, the road-surface score from the
  [circuit tables](https://github.com/aleccrowell/diomedes/tree/master/reference/circuits)
  (0 a racing surface, 2 public-road bumps, crowns and drains; one sd is 0.76);
- **air temperature** at race time (one sd is 5 °C).

A slope is the change in the car's effect, in % of race time, at a race one
standard deviation above the average. Positive means relatively **slower** there.
Slopes measure only how a car's form changed between circuits, not its overall
level: they average to zero over each car's own races, and over each season's
cars. The 15 steepest in each direction, per dimension (cars with at least 20
results):

```@raw html
<div id="fig-car-circuits" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-car-circuits", {
  data: { url: "assets/data/race/cars.csv",
          format: { type: "csv", parse: { season: "number", rows: "number", speed_mean: "number", speed_sd: "number",
                                          rough_mean: "number", rough_sd: "number", temp_mean: "number", temp_sd: "number" } } },
  params: [{ name: "dim", value: "temp",
             bind: { input: "select", options: ["speed", "rough", "temp"],
                     labels: ["Speed", "Roughness", "Temperature"], name: "Dimension " } }],
  transform: [
    { filter: "datum.rows >= 20" },
    { calculate: "datum.constructor + ' ' + datum.season", as: "car" },
    { calculate: "datum[dim + '_mean']", as: "slope" },
    { calculate: "datum[dim + '_sd']", as: "slope_sd" },
    { calculate: "datum.slope - 1.645 * datum.slope_sd", as: "lo" },
    { calculate: "datum.slope + 1.645 * datum.slope_sd", as: "hi" },
    { window: [{ op: "row_number", as: "up" }], sort: [{ field: "slope", order: "ascending" }] },
    { window: [{ op: "row_number", as: "down" }], sort: [{ field: "slope", order: "descending" }] },
    { filter: "datum.up <= 15 || datum.down <= 15" },
    { calculate: "datum.slope < 0 ? 'relatively faster' : 'relatively slower'", as: "direction" }
  ],
  width: "container", height: { step: 14 },
  encoding: {
    y: { field: "car", type: "nominal", sort: { field: "slope", order: "ascending" }, title: null },
    color: { field: "direction", type: "nominal", title: "As the dimension rises",
             scale: { domain: ["relatively faster", "relatively slower"], range: ["#4c78a8", "#e45756"] } },
    tooltip: [{ field: "car", title: "Car" }, { field: "rows", title: "Results" },
              { field: "slope", title: "Slope (% per sd)", format: ".2f" }, { field: "slope_sd", title: "sd", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative", title: "Slope, % of race time per sd (negative = relatively faster as it rises)" },
                  x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 45 }, encoding: { x: { field: "slope", type: "quantitative" } } }
  ]
});
</script>
```

Intervals are ±1.645 sd (about 90%). Most slopes are small next to their
uncertainty. The clearest are from 1996 on, when cars ran more races per
season; earlier slopes shrink towards zero. The slopes typically spread by
0.15% per sd for speed, 0.11% for roughness and 0.09% for temperature.

## Chassis and engines

Each car-season's effect combines three parts
([#95](https://github.com/aleccrowell/diomedes/issues/95)):

- **chassis:** the works that built the car, followed across its seasons as a
  random walk. A rename keeps the works: Toleman → Benetton → Renault → Lotus →
  Alpine is one walk (Enstone), as is Jordan → … → Aston Martin (Silverstone).
- **engine:** the works that built the engine, also a random walk. Rebadged
  engines go to their builder: TAG Heuer is Renault, Petronas is Ferrari.
- **residual:** what's left for that car-season, such as a works team against a
  customer with the same chassis and engine.

A walk moves a little each season: about 0.35% of race time per year for a
chassis and 0.17% for an engine, more after a change of ownership (×1.9) or a
major rule change. The walks carry most of the variation between cars (the
residual's spread is 0.43%). Choose a walk to see its path, relative to each
season's field (negative = faster), with 90% intervals:

```@raw html
<div id="fig-car-walks" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-car-walks", {
  data: { url: "assets/data/race/walks.csv",
          format: { type: "csv", parse: { season: "number", mean: "number", q05: "number", q95: "number" } } },
  params: [{ name: "pick", value: "chassis: brackley",
             bind: { input: "select", options: ["chassis: brackley", "chassis: milton-keynes", "chassis: enstone", "chassis: silverstone", "chassis: hinwil", "chassis: faenza", "chassis: ferrari", "chassis: mclaren", "chassis: williams", "chassis: lotus", "chassis: brabham", "chassis: tyrrell", "chassis: ligier", "chassis: arrows", "chassis: march", "chassis: cooper", "chassis: brm", "chassis: maserati", "chassis: lola", "chassis: ensign", "chassis: osella", "chassis: haas", "engine: ferrari", "engine: mercedes", "engine: renault", "engine: honda", "engine: cosworth", "engine: bmw", "engine: porsche", "engine: climax", "engine: brm", "engine: maserati", "engine: alfa-romeo", "engine: hart"], name: "Walk " } }],
  transform: [{ calculate: "datum.family + ': ' + datum.walk", as: "key" }, { filter: "datum.key == pick" },
              { calculate: "datum.own_change == 'true' ? 'change of ownership' : datum.reg_change == 'true' ? 'rule change' : 'none'", as: "change" }],
  width: "container", height: 280,
  encoding: { x: { field: "season", type: "quantitative", title: "Season", axis: { format: "d" } } },
  layer: [
    { mark: { type: "area", opacity: 0.25 },
      encoding: { y: { field: "q05", type: "quantitative", title: "Walk state, % of race time (negative = faster)" }, y2: { field: "q95" } } },
    { mark: { type: "line", point: true },
      encoding: { y: { field: "mean", type: "quantitative" },
                  tooltip: [{ field: "walk", title: "Walk" }, { field: "season", title: "Season" },
                            { field: "mean", title: "State (%)", format: ".2f" },
                            { field: "q05", title: "5%", format: ".2f" }, { field: "q95", title: "95%", format: ".2f" },
                            { field: "change", title: "Larger step into this season" }] } },
    { mark: { type: "point", shape: "diamond", size: 90, filled: true, color: "#e45756" },
      transform: [{ filter: "datum.change != 'none'" }],
      encoding: { y: { field: "mean", type: "quantitative" } } }
  ]
});
</script>
```

Red diamonds mark seasons entered across a change of ownership or a major rule
change, where the walk may take a larger step. Gaps in a line are seasons the
works or builder didn't race. Engine levels partly reflect their customers:
Ferrari's engine sits slightly slow throughout, likely because its customers were
slower teams. Read engine levels as the engine and its customers together.

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/race/cars.csv"><code>cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/race/walks.csv"><code>walks.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
