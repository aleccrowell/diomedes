# Noise: pace and incidents

## Two sources of variation

Most of a race's spread is small, symmetric variation in pace. Some of it is
something else: a spin, a slow stop, damage, a penalty. That can only *lose*
time, and sometimes a great deal of it. The pace + loss model writes the
residual of each result as

```math
y = \mu + \varepsilon + L, \qquad
\varepsilon \sim \mathcal{N}(0, \sigma), \qquad
L = \begin{cases} 0 & \text{with probability } 1-\pi \\
                  \text{Exponential(mean } \lambda) & \text{with probability } \pi \end{cases}
```

so the density is a mixture of a Gaussian and an exponentially modified
Gaussian (EMG). The figure draws it from that formula; move the sliders to see
how the two components shape the slow tail.

```@raw html
<div id="fig-noise-density" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-noise-density", {
  data: { sequence: { start: -2, stop: 8, step: 0.02, as: "x" } },
  params: [
    { name: "sigma", value: 0.4, bind: { input: "range", min: 0.1, max: 1.5, step: 0.05, name: "pace noise σ (%) " } },
    { name: "piinc", value: 0.5, bind: { input: "range", min: 0, max: 0.95, step: 0.05, name: "loss probability π " } },
    { name: "lam", value: 1.5, bind: { input: "range", min: 0.2, max: 5, step: 0.1, name: "mean loss λ (%) " } }
  ],
  transform: [
    { calculate: "(1 - piinc) * densityNormal(datum.x, 0, sigma)", as: "pace" },
    { calculate: "piinc / lam * exp(sigma * sigma / (2 * lam * lam) - datum.x / lam) * cumulativeNormal(datum.x / sigma - sigma / lam, 0, 1)", as: "loss" },
    { calculate: "datum.pace + datum.loss", as: "total" },
    { fold: ["total", "pace", "loss"], as: ["component", "density"] },
    { calculate: "{total: 'result (mixture)', pace: 'clean race: (1−π)·Normal(0, σ)', loss: 'incident: π·EMG(σ, λ)'}[datum.component]", as: "label" }
  ],
  width: "container", height: 260,
  mark: { type: "line", strokeWidth: 2 },
  encoding: {
    x: { field: "x", type: "quantitative", title: "Residual: result minus expected (% of race time; positive = slower)" },
    y: { field: "density", type: "quantitative", title: "Density" },
    color: { field: "label", type: "nominal", title: null,
             scale: { domain: ["result (mixture)", "clean race: (1−π)·Normal(0, σ)", "incident: π·EMG(σ, λ)"],
                      range: ["#333333", "#4c78a8", "#e45756"] },
             legend: { orient: "top", columns: 1, labelLimit: 400 } },
    strokeDash: { condition: { test: "datum.component == 'total'", value: [1, 0] }, value: [5, 3] }
  }
});
</script>
``` Both the density and the interval probabilities of lapped cars
have closed forms. These need care in the tails, where naive formulas cancel
catastrophically ([`paceloss_effects`](@ref)).

The alternative, symmetric heavy-tailed noise, is much worse. With Gaussian
noise, adding lapped cars inflated σ from 0.48% to 3.7%, and Student-t(4)
brought it to about 1.0%. Even so, Student-t(4) noise is about 1,660 elpd behind the
plain pace + loss model, and 2,905 ± 110 behind pace + loss with era terms
(`pl_rw`) ([#19](https://github.com/aleccrowell/diomedes/issues/19); fits from before the red-flag correction).

## Incidents vary by race and era

The incident probability ``\pi`` and the mean loss ``\lambda`` vary by race:

```math
\operatorname{logit}\pi_r = a_\pi + \beta^\pi_\text{dur}\, d_r + \text{era}^\pi_{s(r)}, \qquad
\log\lambda_r = a_\lambda + \beta^\lambda_\text{dur}\, d_r + \text{era}^\lambda_{s(r)} ,
```

with ``d_r`` the race's standardised log duration. The era terms are random
walks over seasons with Student-t(3) steps: mostly smooth, with occasional
jumps. That beat effects for each decade (by 53 ± 15 elpd) and for a priori
regulation eras (68 ± 17) ([#19](https://github.com/aleccrowell/diomedes/issues/19)). Grouped K-fold cross-validation confirmed
the ranking ([#35](https://github.com/aleccrowell/diomedes/issues/35)).

## A second, bigger loss

A single result can dominate a season's λ. Malaysia 2011's car 42 laps down
pushed that season's mean loss to 10.7% ([#16](https://github.com/aleccrowell/diomedes/issues/16)). So losses get two components:

```math
L \sim (1-\rho)\,\text{Exponential}(\lambda_r) + \rho\,\text{Exponential}(\lambda_2),
\qquad \lambda_2 = \lambda_\text{base}\, e^{\delta}, \ \delta > 0 .
```

With it, about 2% of incidents are big losses (ρ, with ``\lambda_2`` about 6.2×
the baseline, roughly 10%). The routine component fires about 73% of the time
with a mean of 1.7%, and σ falls from 0.51 to 0.40 ([#23](https://github.com/aleccrowell/diomedes/issues/23)). The fit improved by
155 ± 23 elpd ([#23](https://github.com/aleccrowell/diomedes/issues/23)).

!!! note "Reading π and λ"
    With the big-loss component, the "incident" component mostly describes
    routine time lost in a race (traffic, a slow stop), not crashes. π and λ are
    better read as the shape of the slow tail than as literal incident rates
    ([#23](https://github.com/aleccrowell/diomedes/issues/23)).

## The race level and the pace/loss split

The split between pace noise σ and routine losses (π, λ) is only weakly
identified. Shifting every race intercept trades against it at almost no
likelihood cost. With a fixed prior on the intercepts, the summed prior over
about 1,000 races decided the split, and changing that prior moved σ from 0.36
to 0.40 ([#32](https://github.com/aleccrowell/diomedes/issues/32)). Learning the common race level ``\mu_\gamma`` removes that
pull at equal out-of-sample fit (2.9 ± 6.0 elpd), and is the default ([#36](https://github.com/aleccrowell/diomedes/issues/36)).
The remaining uncertainty is honest: the incident intercept ``a_\pi`` has
posterior sd 0.19 ([#32](https://github.com/aleccrowell/diomedes/issues/32)). The driver and car effects don't depend on the split.

## Two modes in the loss mixture

The Jolpica refit of 1950–2025 ([#55](https://github.com/aleccrowell/diomedes/issues/55)) adds the 2020–2025 seasons,
lapped cars at exact gaps (from 2023, when their own times are recorded), and
retirements as partial gaps (from 1996). Its chains didn't all settle in the same
place:

```@raw html
<div id="noise-modes"><p>(The exported fit has a single mode.)</p></div>
<script>
fetch(documenterBaseURL + "/assets/data/race/manifest.json").then(r => r.json()).then(m => {
  if (!m.modes || m.modes.length < 2) return;
  const f = (x, d) => x == null ? "–" : Number(x).toFixed(d);
  const rows = m.modes.map(x =>
    `<tr><td><b>${x.label}</b></td><td>${x.chains.join(", ")}</td>
     <td>${x.convergence ? f(x.convergence.max_rhat, 3) : "–"}</td><td>${f(x.mean_logjoint, 1)}</td>
     <td>${x.kept ? "yes" : "no"}</td></tr>`).join("");
  const agree = (m.effects_by_mode || []).map(e =>
    `<li><b>${e.modes}</b>: driver effects correlate ${f(e.drivers.corr, 4)}, the largest difference
     ${f(e.drivers.max_diff, 2)}% (${f(e.drivers.max_diff_sd, 2)} posterior sds); car-seasons ${f(e.cars.corr, 4)},
     ${f(e.cars.max_diff, 2)}% (${f(e.cars.max_diff_sd, 2)} sds).</li>`).join("");
  document.getElementById("noise-modes").innerHTML =
    `<table><thead><tr><th>Mode</th><th>Chains</th><th>Max R-hat within the mode</th><th>Mean log density</th><th>Used</th></tr></thead>
     <tbody>${rows}</tbody></table>

     <p>How far the effects differ between modes:</p><ul>${agree}</ul>`;
});
</script>
```

Both modes describe the same slow tail of results, split differently between
the two loss components. Each row below is one chain (posterior mean and 90%
interval):

```@raw html
<div id="fig-modes"></div>
<script>
[["sigma", "Pace noise σ (%)", ".2f"], ["pi_base", "Routine-loss probability", ".0%"],
 ["lambda_base", "Routine-loss mean (%)", ".2f"], ["rho", "Big-loss share of incidents", ".1%"],
 ["lambda_big", "Big-loss mean (%)", ".1f"]].forEach(([q, title, fmt]) => {
  const id = "fig-modes-" + q;
  const div = document.createElement("div");
  div.id = id; div.className = "diomedes-figure";
  document.getElementById("fig-modes").appendChild(div);
  diomedesFigure(id, {
    data: { url: "assets/data/race/modes.csv",
            format: { type: "csv", parse: { chain: "number", mean: "number", q05: "number", q95: "number" } } },
    transform: [{ filter: `datum.quantity === "${q}"` }],
    width: "container", height: 70,
    encoding: { y: { field: "chain", type: "ordinal", title: "Chain" },
                color: { field: "mode", type: "nominal", title: "Mode" } },
    layer: [
      { mark: "rule", encoding: { x: { field: "q05", type: "quantitative", title, scale: { zero: false }, axis: { format: fmt } },
                                  x2: { field: "q95" } } },
      { mark: { type: "point", filled: true, size: 50 },
        encoding: { x: { field: "mean", type: "quantitative" },
                    tooltip: [{ field: "chain", title: "Chain" }, { field: "mode", title: "Mode" },
                              { field: "mean", title, format: fmt }] } }
    ]
  });
});
</script>
```

```@raw html
<p>In mode A, <span class="diomedes-stat" data-path="assets/data/race/modes.csv" data-stat="mean" data-x="mean" data-filter="quantity==rho &amp; mode==A" data-format="pct"></span>
of incidents are big losses and the routine loss averages
<span class="diomedes-stat" data-path="assets/data/race/modes.csv" data-stat="mean" data-x="mean" data-filter="quantity==lambda_base &amp; mode==A"></span>%.
In mode B, <span class="diomedes-stat" data-path="assets/data/race/modes.csv" data-stat="mean" data-x="mean" data-filter="quantity==rho &amp; mode==B" data-format="pct"></span>
are big and the routine loss averages
<span class="diomedes-stat" data-path="assets/data/race/modes.csv" data-stat="mean" data-x="mean" data-filter="quantity==lambda_base &amp; mode==B"></span>%:
fewer big losses and larger routine ones, with slightly less pace noise.</p>
```

**Mode B is the posterior; A is a local mode.** B's mean log density is about
150 higher, its log-likelihood about 145 higher, and it predicts better out of
sample (PSIS-LOO, 130 ± 25 elpd; [#55](https://github.com/aleccrowell/diomedes/issues/55)). A difference that size means A holds a
negligible share of the posterior: chains that start near it stay there, but the
correct posterior summary is B alone. So the tables on these pages use B's
chains, and the export drops any mode whose mean log density is more than 10
below the best one's ([#64](https://github.com/aleccrowell/diomedes/issues/64)). Mode A is still shown, here and dashed in [Eras](../findings/eras.md).

**What depends on the mode.** The driver and car effects barely do (the list
above), so this choice doesn't move the rankings. The loss scalars, the season
paths of π and λ, and the big-loss share and size do.

**Why it can happen.** A retirement row is a car that was running slower than
its effects predict before it stopped ([Do the dropped retirements bias the pace effects?](@ref)).
These rows add a block of moderately slow results, and either loss component
can absorb it: a few more big losses, or a wider routine one. A refit without
those rows will show whether they cause the second mode ([#55](https://github.com/aleccrowell/diomedes/issues/55)). Half the chains
landing in a local mode also means a fit like this needs more than one chain per
mode to be trusted: R-hat over all chains is what flagged it.
