# Eras

How the sport itself changed, season by season, from the model's season-level
random walks. Bands are 90% intervals.

```@raw html
<div id="eras-modes"></div>
<script>
fetch(documenterBaseURL + "/assets/data/race/manifest.json").then(r => r.json()).then(m => {
  if (!m.modes || m.modes.length < 2) return;
  const kept = m.modes.filter(x => x.kept), dropped = m.modes.filter(x => !x.kept);
  const name = x => `<b>${x.label}</b> (chains ${x.chains.join(", ")})`;
  document.getElementById("eras-modes").innerHTML =
    `<div class="admonition is-info"><header class="admonition-header">Two solutions for the loss mixture</header>
     <div class="admonition-body"><p>This fit's chains settled in ${m.modes.length} modes of the loss mixture.
     Mode ${kept.map(name).join(", ")} has much higher posterior density, and the pooled tables use it.
     ${dropped.map(name).join(", ")}, a local mode with a negligible share of the posterior, is drawn dashed
     for comparison. Where the lines overlap, the conclusion doesn't depend on the mode (see
     <a href="../../methodology/noise/#Two-modes-in-the-loss-mixture">Two modes in the loss mixture</a>).</p></div></div>`;
});
</script>
```

## How spread out results were

The pace scale κ multiplies every driver difference in a season. Values above 1
mean results were more spread out than average; below 1, closer. It doesn't
measure how fast cars were in absolute terms.

```@raw html
<div id="fig-kappa" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-kappa", {
  data: { url: "assets/data/race/seasons_by_mode.csv",
          format: { type: "csv", parse: { season: "number", kappa_mean: "number", kappa_q05: "number", kappa_q95: "number" } } },
  width: "container", height: 260,
  encoding: { x: { field: "season", type: "quantitative", title: "Season", axis: { format: "d" }, scale: { zero: false } } },
  layer: [
    { mark: { type: "area", opacity: 0.2 }, transform: [{ filter: "datum.kept === 'true'" }],
      encoding: { color: { field: "mode", type: "nominal", title: "Mode" }, y: { field: "kappa_q05", type: "quantitative", title: "Pace scale κ" }, y2: { field: "kappa_q95" } } },
    { mark: "line",
      encoding: { color: { field: "mode", type: "nominal", title: "Mode" },
                  strokeDash: { field: "kept", type: "nominal", title: "Used", sort: "descending",
                                scale: { domain: ["true", "false"], range: [[1, 0], [5, 3]] } }, y: { field: "kappa_mean", type: "quantitative" },
                  tooltip: [{ field: "season", title: "Season" }, { field: "mode", title: "Mode" },
                            { field: "kappa_mean", title: "κ", format: ".2f" }] } },
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
<div id="fig-incidents-pi" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-incidents-pi", {
  data: { url: "assets/data/race/seasons_by_mode.csv",
          format: { type: "csv", parse: { season: "number", pi_mean: "number", pi_q05: "number", pi_q95: "number" } } },
  width: "container", height: 180,
  encoding: { x: { field: "season", type: "quantitative", title: "Season", axis: { format: "d" }, scale: { zero: false } } },
  layer: [
    { mark: { type: "area", opacity: 0.2 }, transform: [{ filter: "datum.kept === 'true'" }],
      encoding: { color: { field: "mode", type: "nominal", title: "Mode" }, y: { field: "pi_q05", type: "quantitative", title: "Loss probability π", axis: { format: ".0%" } }, y2: { field: "pi_q95" } } },
    { mark: "line",
      encoding: { color: { field: "mode", type: "nominal", title: "Mode" },
                  strokeDash: { field: "kept", type: "nominal", title: "Used", sort: "descending",
                                scale: { domain: ["true", "false"], range: [[1, 0], [5, 3]] } }, y: { field: "pi_mean", type: "quantitative" },
                  tooltip: [{ field: "season", title: "Season" }, { field: "mode", title: "Mode" },
                            { field: "pi_mean", title: "π", format: ".1%" }] } }
  ]
});
</script>
<div id="fig-incidents-lambda" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-incidents-lambda", {
  data: { url: "assets/data/race/seasons_by_mode.csv",
          format: { type: "csv", parse: { season: "number", lambda_mean: "number", lambda_q05: "number", lambda_q95: "number" } } },
  width: "container", height: 180,
  encoding: { x: { field: "season", type: "quantitative", title: "Season", axis: { format: "d" }, scale: { zero: false } } },
  layer: [
    { mark: { type: "area", opacity: 0.2 }, transform: [{ filter: "datum.kept === 'true'" }],
      encoding: { color: { field: "mode", type: "nominal", title: "Mode" }, y: { field: "lambda_q05", type: "quantitative", title: "Mean loss λ (%)" }, y2: { field: "lambda_q95" } } },
    { mark: "line",
      encoding: { color: { field: "mode", type: "nominal", title: "Mode" },
                  strokeDash: { field: "kept", type: "nominal", title: "Used", sort: "descending",
                                scale: { domain: ["true", "false"], range: [[1, 0], [5, 3]] } }, y: { field: "lambda_mean", type: "quantitative" },
                  tooltip: [{ field: "season", title: "Season" }, { field: "mode", title: "Mode" },
                            { field: "lambda_mean", title: "λ (%)", format: ".2f" }] } }
  ]
});
</script>
```

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/race/seasons_by_mode.csv"><code>seasons_by_mode.csv</code></a> (per mode) and
<a class="diomedes-data" data-path="assets/data/race/seasons.csv"><code>seasons.csv</code></a> (pooled over all chains). Columns are described in <a href="../data/">Data downloads</a>.</p>
```
