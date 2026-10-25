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
     <a href="../../methodology/noise/#Checking-for-multiple-modes">Checking for multiple modes</a>).</p></div></div>`;
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
race length, **for a dry race at a circuit of average tightness**. Rain and walls
are separate terms (see [Rain and walls](#Rain-and-walls) below), so these paths
show how the sport changed apart from the weather and the circuits it visited.
Rare big losses (repairs, many laps down) are a separate component and aren't
shown here (see [Noise: pace and incidents](@ref)).

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

## Rain and walls

How rain and walled circuits change routine losses
([#81](https://github.com/aleccrowell/diomedes/issues/81)). Each race is flagged
dry, mixed or wet from race reports, and each circuit has a walls score
(0: mostly run-off; 2: barriers line most of the lap), both from the
[circuit tables](https://github.com/aleccrowell/diomedes/tree/master/reference/circuits).
Left: how many times higher the **odds** of a routine loss are; right: how many
times larger the mean loss is. Both are against a dry race at an open circuit,
with 90% intervals; 1 means no change.

```@raw html
<div id="fig-track" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-track", {
  data: { url: "assets/data/race/track.csv",
          format: { type: "csv", parse: { pi_odds_mean: "number", pi_odds_q05: "number", pi_odds_q95: "number",
                                          lambda_mult_mean: "number", lambda_mult_q05: "number", lambda_mult_q95: "number" } } },
  hconcat: [
    { width: 260, height: { step: 30 },
      layer: [
        { mark: { type: "rule", opacity: 0.6 },
          encoding: { y: { field: "label", type: "nominal", title: null, sort: null, axis: { labelLimit: 280 } }, x: { field: "pi_odds_q05", type: "quantitative", title: "Odds of a routine loss (×)",
                           scale: { type: "log" } }, x2: { field: "pi_odds_q95" } } },
        { mark: { type: "point", filled: true, size: 60 },
          encoding: { y: { field: "label", type: "nominal", title: null, sort: null, axis: { labelLimit: 280 } }, x: { field: "pi_odds_mean", type: "quantitative" },
                      tooltip: [{ field: "label", title: "Term" }, { field: "pi_odds_mean", title: "Odds ×", format: ".2f" },
                                { field: "pi_odds_q05", title: "5%", format: ".2f" }, { field: "pi_odds_q95", title: "95%", format: ".2f" }] } },
        { data: { values: [{}] }, mark: { type: "rule", strokeDash: [4, 4], opacity: 0.6 }, encoding: { x: { datum: 1 } } }
      ] },
    { width: 260, height: { step: 30 },
      layer: [
        { mark: { type: "rule", opacity: 0.6 },
          encoding: { y: { field: "label", type: "nominal", title: null, sort: null, axis: null }, x: { field: "lambda_mult_q05", type: "quantitative", title: "Mean loss size (×)",
                           scale: { zero: false } }, x2: { field: "lambda_mult_q95" } } },
        { mark: { type: "point", filled: true, size: 60 },
          encoding: { y: { field: "label", type: "nominal", title: null, sort: null, axis: null }, x: { field: "lambda_mult_mean", type: "quantitative" },
                      tooltip: [{ field: "label", title: "Term" }, { field: "lambda_mult_mean", title: "Loss ×", format: ".2f" },
                                { field: "lambda_mult_q05", title: "5%", format: ".2f" }, { field: "lambda_mult_q95", title: "95%", format: ".2f" }] } },
        { data: { values: [{}] }, mark: { type: "rule", strokeDash: [4, 4], opacity: 0.6 }, encoding: { x: { datum: 1 } } }
      ] }
  ]
});
</script>
```

Rain is the larger effect: a wet race has about 6.5 times the odds of a routine
loss (90% interval 3.3–11), and a mixed one about 2.2 times (1.5–3.1); each loss
is also 22–26% larger. These odds are smaller than before per-driver incident
proneness was added ([#101](https://github.com/aleccrowell/diomedes/issues/101)): some of what looked like rain was who
was driving. Walls barely change how often losses happen, but make each one
about a quarter larger, as a mistake that would be a trip across the
run-off costs more against a barrier. In seasons with many wet races, the dry
paths above are lower than the incident rates seen in those seasons' results.

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/race/seasons_by_mode.csv"><code>seasons_by_mode.csv</code></a> (per mode) and
<a class="diomedes-data" data-path="assets/data/race/seasons.csv"><code>seasons.csv</code></a> (pooled over all chains);
<a class="diomedes-data" data-path="assets/data/race/track.csv"><code>track.csv</code></a> (rain and walls). Columns are described in <a href="../data/">Data downloads</a>.</p>
```
