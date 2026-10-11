# Careers

## The shape of a career

The population age curve: how pace changes with age for the average driver, in
% of race time (negative = faster), with a 90% band. Only its **shape** is
identified, not its overall tilt or the age of its minimum (see
[The question and what is identifiable](@ref)). Read the bends, not the slope.

```@raw html
<div id="fig-age-curve" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-age-curve", {
  data: { url: "assets/data/race/age_curve.csv",
          format: { type: "csv", parse: { age: "number", rows: "number", mean: "number", q05: "number", q95: "number" } } },
  width: 600, height: 300,
  encoding: { x: { field: "age", type: "quantitative", title: "Age", scale: { zero: false } } },
  layer: [
    { mark: { type: "area", opacity: 0.25 },
      encoding: { y: { field: "q05", type: "quantitative", title: "Pace relative to the curve's average (%)", scale: { reverse: true } },
                  y2: { field: "q95" } } },
    { mark: { type: "line", point: true },
      encoding: { y: { field: "mean", type: "quantitative" },
                  tooltip: [{ field: "age", title: "Age" }, { field: "rows", title: "Results at this age" },
                            { field: "mean", title: "Curve (%)", format: ".2f" },
                            { field: "q05", title: "5%", format: ".2f" }, { field: "q95", title: "95%", format: ".2f" }] } }
  ]
});
</script>
```

The vertical axis is reversed, so faster is higher. Ages above about 42 have
few results, which is why the band widens there.

## Individual trajectories

On top of the shared curve, each driver has a linear career slope: how much
faster or slower they got per decade of age **relative to the average
driver's trajectory**. Negative means they improved more than the average
driver did. The 15 largest in each direction, among drivers with at least 40
results:

```@raw html
<div id="fig-career-slopes" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-career-slopes", {
  data: { url: "assets/data/race/drivers.csv",
          format: { type: "csv", parse: { rows: "number", first_season: "number", last_season: "number",
                                          slope_mean: "number", slope_sd: "number" } } },
  transform: [
    { filter: "datum.rows >= 40" },
    { calculate: "datum.name + ' (' + datum.first_season + '–' + datum.last_season + ')'", as: "label" },
    { calculate: "datum.slope_mean - 1.645 * datum.slope_sd", as: "lo" },
    { calculate: "datum.slope_mean + 1.645 * datum.slope_sd", as: "hi" },
    { window: [{ op: "row_number", as: "up" }], sort: [{ field: "slope_mean", order: "ascending" }] },
    { window: [{ op: "row_number", as: "down" }], sort: [{ field: "slope_mean", order: "descending" }] },
    { filter: "datum.up <= 15 || datum.down <= 15" },
    { calculate: "datum.slope_mean < 0 ? 'improved' : 'declined'", as: "direction" }
  ],
  width: 520, height: { step: 15 },
  encoding: {
    y: { field: "label", type: "nominal", sort: { field: "slope_mean", order: "ascending" }, title: null,
         axis: { labelLimit: 260 } },
    color: { field: "direction", type: "nominal", title: null,
             scale: { domain: ["improved", "declined"], range: ["#4c78a8", "#e45756"] } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "rows", title: "Results" },
              { field: "slope_mean", title: "Slope (% per decade)", format: ".2f" },
              { field: "slope_sd", title: "sd", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative", title: "Career slope relative to the average driver, % per decade" },
                  x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 45 }, encoding: { x: { field: "slope_mean", type: "quantitative" } } }
  ]
});
</script>
```

Intervals are ±1.645 sd (about 90%). Kubica's decline is his comeback after a
serious injury.
