# Model progression

Each term was added in its own issue, fitted to every classified F1 finish
(1950–2019, 4 chains × 1,000 draws) and kept only if it improved out-of-sample
fit. Fit is measured as expected log predictive density (elpd) by PSIS-LOO;
gains are over the previous row unless stated. The first two rows were fitted
before the red-flag correction ([#19](https://github.com/aleccrowell/diomedes/issues/19)).

| Step | Model | Gain (elpd) | Fit time | Source |
|---|---|---|---|---|
| pace + loss noise, era random walks | `pl_rw` | +1,243 ± 60 over no era terms | | [#19](https://github.com/aleccrowell/diomedes/issues/19) |
| race duration in the loss | `pl_dur_rw` | (adopted without a direct comparison) | | [#19](https://github.com/aleccrowell/diomedes/issues/19) |
| pace scale by season | `…_kappa` | +98 ± 12 | ~3 h | [#20](https://github.com/aleccrowell/diomedes/issues/20) |
| population age curve | `…_age` | +30 ± 9.2 | ~3 h | [#22](https://github.com/aleccrowell/diomedes/issues/22) |
| big incident losses | `…_big` | +155 ± 23 | ~3.6 h | [#23](https://github.com/aleccrowell/diomedes/issues/23) |
| per-driver career slopes | `…_slope` | +89.7 ± 13.7 | ~4.6 h | [#24](https://github.com/aleccrowell/diomedes/issues/24) |
| in-season development | `…_dev` | +155.7 ± 16.4 | ~4.8 h | [#27](https://github.com/aleccrowell/diomedes/issues/27) |
| learned common race level | (default) | +2.9 ± 6.0 (equal) | | [#34](https://github.com/aleccrowell/diomedes/issues/34), [#37](https://github.com/aleccrowell/diomedes/issues/37) |
| circuit slopes; walls and rain on incidents | `…_circ_track` | +346 ± 28 | ~9.2 h | [#81](https://github.com/aleccrowell/diomedes/issues/81) |

The last row was fitted to F1 1950–2025 with retirements from 1996 as partial
gaps, the data of the published fit, and compared with the same model without
the two terms. The two terms were added together and their gain isn't split.

The gains, with ±2 standard errors (the first step, era terms over none, is
+1,243 ± 60 and is left off the scale):

```@raw html
<div id="fig-elpd-gains" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-elpd-gains", {
  data: { values: [
    { step: 1, term: "pace scale by season (κ)", gain: 98, se: 12 },
    { step: 2, term: "population age curve", gain: 30, se: 9.2 },
    { step: 3, term: "big incident losses", gain: 155, se: 23 },
    { step: 4, term: "per-driver career slopes", gain: 89.7, se: 13.7 },
    { step: 5, term: "in-season development", gain: 155.7, se: 16.4 },
    { step: 6, term: "learned common race level", gain: 2.9, se: 6.0 },
    { step: 7, term: "circuit slopes; walls and rain", gain: 346, se: 28 }
  ] },
  transform: [
    { calculate: "datum.gain - 2 * datum.se", as: "lo" },
    { calculate: "datum.gain + 2 * datum.se", as: "hi" }
  ],
  width: "container", height: { step: 30 },
  encoding: {
    y: { field: "term", type: "nominal", sort: { field: "step" }, title: null, axis: { labelLimit: 220 } },
    tooltip: [{ field: "term", title: "Term" }, { field: "gain", title: "elpd gain" }, { field: "se", title: "SE" }]
  },
  layer: [
    { mark: { type: "bar", height: 16, color: "#4c78a8", opacity: 0.8 },
      encoding: { x: { field: "gain", type: "quantitative", title: "Gain in expected log predictive density over the previous model" } } },
    { mark: { type: "rule", color: "#333333" }, encoding: { x: { field: "lo", type: "quantitative" }, x2: { field: "hi" } } }
  ]
});
</script>
```

## What each term found

**Race duration** ([#19](https://github.com/aleccrowell/diomedes/issues/19)). Longer races don't have more incidents (slope on
logit π −0.014 ± 0.073), but each incident costs a smaller share of a long
race (slope on log λ −0.071 ± 0.025).

**Pace scale κ** ([#20](https://github.com/aleccrowell/diomedes/issues/20)). A season's spread of results is not constant: κ is
about 2.5 in the early 1950s, about 1.4 in the mid-1960s and the mid-1980s turbo
era, and 0.55–0.6 in 2013–19. The largest steps fall in 1989 and 2010. Without
κ, drivers of the 1950s, when gaps were large, absorb the spread: the best
driver's effect shrank from −2.58% to −2.08% when κ was added.

**Age curve** ([#22](https://github.com/aleccrowell/diomedes/issues/22)). Concave: drivers are slowest when young (+0.54 at 18),
reach a plateau around 25–33 (about −0.1), and slow again from about 36
(+0.34 at 40). Only the curvature is identified (see
[The question and what is identifiable](@ref)).

**Career slopes** ([#24](https://github.com/aleccrowell/diomedes/issues/24)). Beyond the shared curve, drivers' trajectories
differ by about 1.08% per decade of age (τ). Some examples: Kubica declined (+1.4,
his comeback after injury) and Verstappen improved fast (−1.4).

**In-season development** ([#27](https://github.com/aleccrowell/diomedes/issues/27)). Cars' form relative to the field changes
during a season by about 0.5% per season (τ_dev = 0.50 ± 0.05). The largest
trends match known histories: Force India's 2015 B-spec car (−0.62 ± 0.27),
Renault 2011's strong start and fade (+0.97 ± 0.30). Trends are relative to the
field, so a dominant car can "fade" while still winning (McLaren 1988, +0.59).

**Circuit slopes, walls and rain** ([#81](https://github.com/aleccrowell/diomedes/issues/81)).
Each car-season has a slope on its races' average speed, roughness (the road
surface score) and air temperature. Across car-seasons the slopes spread by
about 0.15% of race time per standard deviation of speed (26 km/h), 0.11% for
roughness and 0.09% for temperature (5 °C). McLaren 2024 has the field's
steepest temperature slope (faster in hot races). Rain dominates incidents: the
odds of a routine loss are about 3.4× higher in a mixed race and 10× in a wet
one, and each loss is 16–27% larger. Walls hardly change how often losses happen but
make them about 23% larger. Car and driver effects barely move (correlation
0.996 with the previous fit).

## How models were compared

- **PSIS-LOO** (Pareto-smoothed importance-sampling leave-one-out) on every row:
  timed and lapped results alike, so all models are scored on the same data
  ([`pointwise_loglik`](@ref)).
- **Its weak spot.** Rows with Pareto k > 0.7 make the estimate unreliable.
  There are 350 such rows (2.7%) for the era models, rising to 814 for the
  development model and 869 with the circuit terms ([#18](https://github.com/aleccrowell/diomedes/issues/18), [#27](https://github.com/aleccrowell/diomedes/issues/27), [#81](https://github.com/aleccrowell/diomedes/issues/81)).
- **Grouped K-fold** checks the close calls. Each race's results are spread
  over 5 folds, so every race keeps an intercept. K-fold confirmed the era
  ranking: random walk over decade by 43.5 ± 15.2, and over regime by
  54.6 ± 16.9 ([#35](https://github.com/aleccrowell/diomedes/issues/35)). It hasn't been run on the larger models, where one fold
  takes 3.6–4.6 hours on the Raspberry Pi 5 used for these fits ([#18](https://github.com/aleccrowell/diomedes/issues/18)).
