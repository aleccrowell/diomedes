# Eras

How the sport itself changed, season by season, from the model's season-level
random walks. Bands are 90% intervals.

## How spread out results were

The pace scale κ multiplies every driver difference in a season. Values above 1
mean results were more spread out than average; below 1, closer. It doesn't
measure how fast cars were in absolute terms.

```@raw html
<div id="fig-kappa" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-kappa", {
  data: { url: "assets/data/race/seasons.csv",
          format: { type: "csv", parse: { season: "number", kappa_mean: "number", kappa_q05: "number", kappa_q95: "number" } } },
  width: 640, height: 260,
  layer: [
    { mark: { type: "area", opacity: 0.25 },
      encoding: { x: { field: "season", type: "quantitative", title: "Season", axis: { format: "d" }, scale: { zero: false } },
                  y: { field: "kappa_q05", type: "quantitative", title: "Pace scale κ" }, y2: { field: "kappa_q95" } } },
    { mark: "line",
      encoding: { x: { field: "season", type: "quantitative" },
                  y: { field: "kappa_mean", type: "quantitative" },
                  tooltip: [{ field: "season", title: "Season" }, { field: "kappa_mean", title: "κ", format: ".2f" }] } },
    { data: { values: [{}] },
      mark: { type: "rule", strokeDash: [4, 4], opacity: 0.6 }, encoding: { y: { datum: 1 } } }
  ]
});
</script>
```

## Time lost to incidents

The probability π that a result includes a routine loss (traffic, a slow stop,
a mistake), and the mean size λ of that loss in % of race time, both at average
race length. Rare big losses (repairs, many laps down) are a separate component
and aren't shown here (see [Noise: pace and incidents](@ref)).

```@raw html
<div id="fig-incidents" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-incidents", {
  data: { url: "assets/data/race/seasons.csv",
          format: { type: "csv", parse: { season: "number", pi_mean: "number", pi_q05: "number", pi_q95: "number",
                                          lambda_mean: "number", lambda_q05: "number", lambda_q95: "number" } } },
  vconcat: [
    { width: 640, height: 180,
      encoding: { x: { field: "season", type: "quantitative", title: null, axis: { format: "d" }, scale: { zero: false } } },
      layer: [
        { mark: { type: "area", opacity: 0.25 },
          encoding: { y: { field: "pi_q05", type: "quantitative", title: "Loss probability π", axis: { format: ".0%" } },
                      y2: { field: "pi_q95" } } },
        { mark: "line", encoding: { y: { field: "pi_mean", type: "quantitative" },
                                    tooltip: [{ field: "season", title: "Season" }, { field: "pi_mean", title: "π", format: ".1%" }] } }
      ] },
    { width: 640, height: 180,
      encoding: { x: { field: "season", type: "quantitative", title: "Season", axis: { format: "d" }, scale: { zero: false } } },
      layer: [
        { mark: { type: "area", opacity: 0.25 },
          encoding: { y: { field: "lambda_q05", type: "quantitative", title: "Mean loss λ (%)" }, y2: { field: "lambda_q95" } } },
        { mark: "line", encoding: { y: { field: "lambda_mean", type: "quantitative" },
                                    tooltip: [{ field: "season", title: "Season" }, { field: "lambda_mean", title: "λ (%)", format: ".2f" }] } }
      ] }
  ]
});
</script>
```
