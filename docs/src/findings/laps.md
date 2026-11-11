# Lap times

Results of the lap-time model (see [The lap-time model](@ref)), fitted to
green-flag laps from 2011 to the 2026 races run when it was fitted (Jolpica lap times,
recorded pit stops). Besides driver and car effects it has:
- each driver's **peak pace** and how often and how far their laps fall short of
  it ([#113](https://github.com/aleccrowell/diomedes/issues/113), [below](#Peak-pace-and-consistency));
- traffic around each lap (dirty air, blue flags, lapped cars, a car close
  behind or a big lead), and the field's pace on each lap of a race, with a
  slope per car-season ([#113](https://github.com/aleccrowell/diomedes/issues/113));
- a career slope per driver (season-level age);
- a stint-age curve, with driver and car-season deviations from it (tyre management);
- an effect of leading the race;
- lap 1 as its own term, with the grid slot and car-season start effects
  ([#66](https://github.com/aleccrowell/diomedes/issues/66));
- a population age curve shared by all drivers, as in the race model
  ([#12](https://github.com/aleccrowell/diomedes/issues/12));
- car-season slopes on three circuit dimensions: average lap speed, street score
  and race-time air temperature ([#14](https://github.com/aleccrowell/diomedes/issues/14));
- from the race model ([#108](https://github.com/aleccrowell/diomedes/issues/108)):
  walls, mixed and wet races on the incident terms, per-driver incident proneness,
  big incident losses, and each team's form as a walk from race to race in place
  of a fixed car-season effect.

It has no tyre compound or weather terms: those are known only from 2021, and a
separate 2021–2026 fit with them is [below](#Track-conditions-(2021–2026)).

It is **in development**: per-lap conditions are in progress
([#116](https://github.com/aleccrowell/diomedes/issues/116), [results so far](#Track-conditions-(2021–2026))).
Its four chains converge to a single mode ([below](#Convergence)). The previous
fit, 1996–2025 without the #108 and #113 terms, is described in
[#66](https://github.com/aleccrowell/diomedes/issues/66) and
[#12](https://github.com/aleccrowell/diomedes/issues/12). Its driver effects
correlate 0.88 with this fit's mean pace (39 drivers with at least 2,000 laps in both).

!!! note "What these numbers measure"
    The race model and the lap model are two views of the same question from
    different data, and neither is the reference for the other. The race model
    measures the time to finish a race, from one result per driver per race
    since 1950. That includes pit stops, strategy, safety-car periods and
    incidents, and it relies on its own split between pace and incident losses.
    The lap model measures green-flag pace under race conditions, plus the start
    (lap 1), from about 50 laps per driver per race since 2011. Its driver
    effects in the comparisons below are **mean pace**: the peak less the laps that
    fall short of it. Traffic is now a term of its own, but the tyre, fuel and
    engine management drivers do when they don't need to push stays in, apart
    from what the lead and big-lead terms take. Pit-stop timing, strategy and
    safety-car periods are left out. So it **underrates drivers who win
    comfortably** (they manage their pace) and flatters drivers who lose most of
    their time in the parts of a race it leaves out. Where the two agree, the
    result doesn't depend on either model's blind spots. Where they disagree,
    either could be wrong. Its driver ranking answers a different question from
    the race model's: pace on clean laps, not results.

```@raw html
<div id="manifest-laps"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps/manifest.json").then(r => r.json()).then(m => {
  document.getElementById("manifest-laps").innerHTML =
    `<p><small>Fit <code>${m.fit}</code>. Data: ${m.data}. Convergence: max R-hat ${m.convergence.max_rhat},
     min bulk ESS ${m.convergence.min_ess}, ${m.convergence.divergences} divergence(s). Compared with the race model
     <code>${m.race_model}</code>. Exported ${m.date}.</small></p>`;
});
</script>
```

## Lap model against the race model: cars

```@raw html
<p>Each point is a car-season, placed by its effect in the two models. They agree
closely (correlation
<span class="diomedes-stat" data-path="assets/data/laps/vs_race_cars.csv" data-stat="corr" data-x="race_mean" data-y="lap_mean">…</span>
over <span class="diomedes-stat" data-path="assets/data/laps/vs_race_cars.csv" data-stat="count" data-format="int">…</span>
car-seasons). The lap model's spread is larger, because clean laps leave out the
safety-car bunching and strategy noise that compress race time gaps.</p>
```

```@raw html
<div id="fig-laps-cars" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-cars", {
  data: { url: "assets/data/laps/vs_race_cars.csv",
          format: { type: "csv", parse: { laps: "number", lap_mean: "number", lap_sd: "number", season: "number",
                                          race_mean: "number", race_sd: "number" } } },
  // "constructor" can't be used as a field name in Vega (Object.prototype); use a copy
  transform: [{ calculate: "datum.constructor", as: "team" }],
  params: [{ name: "zoom", select: "interval", bind: "scales" }],
  width: "container", height: 420,
  mark: { type: "circle", size: 70, opacity: 0.8 },
  encoding: {
    x: { field: "race_mean", type: "quantitative", title: "Race model: car effect (%)" },
    y: { field: "lap_mean", type: "quantitative", title: "Lap model: car effect (%)" },
    // a gradient legend: 16 seasons as separate entries would run past the plot's height
    color: { field: "season", type: "quantitative", title: "Season", scale: { scheme: "viridis" },
             legend: { format: "d", gradientLength: 300 } },
    tooltip: [{ field: "team", title: "Constructor" }, { field: "season", title: "Season" },
              { field: "race_mean", title: "Race model (%)", format: ".2f" },
              { field: "lap_mean", title: "Lap model (%)", format: ".2f" }, { field: "laps", title: "Laps" }]
  }
});
</script>
```

## Lap model against the race model: drivers

```@raw html
<p>Drivers agree less (correlation
<span class="diomedes-stat" data-path="assets/data/laps/vs_race_drivers.csv" data-stat="corr" data-x="race_mean" data-y="lap_mean" data-filter="laps>=500">…</span>
over the <span class="diomedes-stat" data-path="assets/data/laps/vs_race_drivers.csv" data-stat="count" data-filter="laps>=500" data-format="int">…</span>
drivers with at least 500 clean laps), and the lap model's driver effects are
less spread out. Race gaps include what the lap model leaves out: pit stops,
strategy, safety-car periods and the laps lost to incidents. So back-markers look
faster on laps than in races, and the front-runners less dominant. Adding starts,
lead laps and tyre management did not change that
(<a href="https://github.com/aleccrowell/diomedes/issues/66">#66</a>). The pattern holds in this fit, with
traffic as its own term: lap-model mean pace rises about 0.45 % per 1 % of
race-model effect. The race model's effects also cover whole
careers from 1950, while this fit starts in 2011: a career that began earlier
(Alonso, Räikkönen, Massa, Button, Webber) enters here only from 2011.</p>
```

```@raw html
<div id="fig-laps-drivers" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-drivers", {
  data: { url: "assets/data/laps/vs_race_drivers.csv",
          format: { type: "csv", parse: { laps: "number", lap_mean: "number", lap_sd: "number",
                                          race_mean: "number", race_sd: "number" } } },
  transform: [{ filter: "datum.laps >= 500" }],
  params: [{ name: "zoom", select: "interval", bind: "scales" }],
  width: "container", height: 420,
  mark: { type: "circle", opacity: 0.8 },
  encoding: {
    x: { field: "race_mean", type: "quantitative", title: "Race model: driver effect, whole career (%)" },
    y: { field: "lap_mean", type: "quantitative", title: "Lap model: mean pace, 2011–2026 (%)" },
    size: { field: "laps", type: "quantitative", title: "Laps", scale: { range: [20, 300] } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "laps", title: "Laps" },
              { field: "race_mean", title: "Race model (%)", format: ".2f" },
              { field: "lap_mean", title: "Lap model (%)", format: ".2f" }]
  }
});
</script>
```

Drivers with at least 500 clean laps.

## Peak pace and consistency

The model splits each driver's laps in two
([#113](https://github.com/aleccrowell/diomedes/issues/113)). The **peak** is
their clean-lap pace on the car's limit. On other laps they **fall short** of it.
How often, and by how much, varies by driver and by car-season, because a hard
car to drive shouldn't read as inconsistent drivers. Mean pace is the peak plus
the expected shortfall (how often × how far). All three are in % of lap time
against the drivers' average, lower = better. The shortfall is net of the
traffic, lead and big-lead terms, of the field's pace on the lap, and of
incidents, which have their own losses.

```@raw html
<p>Most of the difference between drivers is peak pace. Over the
<span class="diomedes-stat" data-path="assets/data/laps/peak.csv" data-stat="count" data-filter="laps>=2000" data-format="int">…</span>
drivers with at least 2,000 laps, the peaks spread about six times as much as the
expected shortfalls (sd 0.22 % against 0.04 %), and they are unrelated (correlation
<span class="diomedes-stat" data-path="assets/data/laps/peak.csv" data-stat="corr" data-x="peak" data-y="short_mean" data-filter="laps>=2000">…</span>).
Peak and mean pace correlate
<span class="diomedes-stat" data-path="assets/data/laps/peak.csv" data-stat="corr" data-x="peak" data-y="mean_pace" data-filter="laps>=2000">…</span>.
Max Verstappen is the outlier on both: the fastest peak (−0.66 %) and the
smallest expected shortfall (0.075 %), because he falls short on 8 % of laps, where
the other drivers do on 12–37 %. Drivers differ mostly in how often they fall short,
not how far: the mean size of a shortfall stays between 0.7 % and 1.1 %. Car-seasons
vary more than drivers in both.</p>
<p>A caveat: a shortfall the driver chose and one they couldn't avoid look the same.
Beyond the traffic and lead terms, a car managing its pace (holding a position,
saving tyres or fuel) reads as falling short. And this fit has no tyre or weather terms, so
wet and mixed laps go mostly to the shortfall rather than to a slower limit, and
drivers with more wet laps may look less consistent than they are. The 2021–2026
fit with those terms is <a href="#Track-conditions-(2021–2026)">below</a>
(<a href="https://github.com/aleccrowell/diomedes/issues/116">#116</a>).</p>
```

```@raw html
<div id="fig-laps-peak" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-peak", {
  data: { url: "assets/data/laps/peak.csv",
          format: { type: "csv", parse: { laps: "number", peak: "number", peak_sd: "number", mean_pace: "number",
                                          short_mean: "number", short_mean_sd: "number", short_sd: "number",
                                          short_freq: "number", short_size: "number" } } },
  transform: [{ filter: "datum.laps >= 2000" }],
  params: [{ name: "zoom", select: "interval", bind: "scales" }],
  width: "container", height: 420,
  mark: { type: "circle", opacity: 0.8 },
  encoding: {
    x: { field: "peak", type: "quantitative", title: "Peak pace (%, lower = faster)", scale: { zero: false } },
    y: { field: "short_mean", type: "quantitative", title: "Expected shortfall (%, lower = more consistent)",
         scale: { zero: false } },
    size: { field: "laps", type: "quantitative", title: "Laps", scale: { range: [20, 300] } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "laps", title: "Laps" },
              { field: "peak", title: "Peak (%)", format: ".2f" },
              { field: "mean_pace", title: "Mean pace (%)", format: ".2f" },
              { field: "short_mean", title: "Expected shortfall (%)", format: ".3f" },
              { field: "short_freq", title: "Share of laps short", format: ".0%" },
              { field: "short_size", title: "Mean shortfall when short (%)", format: ".2f" }]
  }
});
</script>
```

Drivers with at least 2,000 clean laps, at the base rates: a dry race, with a car
whose own shortfall offsets are zero.

## Track conditions (2021–2026)

From 2021 the F1 timing data says which tyre compound each lap was run on, the
track temperature and whether rain was falling
([#116](https://github.com/aleccrowell/diomedes/issues/116)). A second fit adds these
to the model above, over 2021–2026 only. It has every term of the main fit, plus:
- an offset for each compound against the SOFT, and a stint slope per compound on
  top of the shared stint curve;
- a slope on track temperature, against the race's mean;
- rain on the peak, and wet tyres and rain on how often and how far laps fall
  short;
- each driver's deviation on wet-tyre laps (intermediates and wets) from their dry
  pace.

Its conditions terms and wet deviations converge (R-hat ≤ 1.015, bulk ESS ≥ 400).
Its driver and car-season levels don't yet (R-hat up to 1.09). They mix slowly
along the ridge between a dominant driver and their car: Verstappen against Red Bull
in 2023, Norris and Piastri against McLaren in 2025. So the rankings on this page
stay on the 2011–2026 fit, and this section shows only the conditions results.

```@raw html
<div id="manifest-laps-conditions"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps_conditions/manifest.json").then(r => r.json()).then(m => {
  const g = m.convergence.groups, w = g["wet drv"], s = g["scalars"];
  document.getElementById("manifest-laps-conditions").innerHTML =
    `<p><small>Fit <code>${m.fit}</code>. Data: ${m.data}. Conditions: ${m.conditions}. Convergence: scalars max
     R-hat ${s.max_rhat} (min bulk ESS ${s.min_ess}), wet deviations ${w.max_rhat} (${w.min_ess}); over every
     group ${m.convergence.max_rhat} (${m.convergence.min_ess}), ${m.convergence.divergences} divergence(s).
     Exported ${m.date}.</small></p>`;
});
</script>
```

**Compounds.** Against the SOFT, a MEDIUM lap is 0.16 % slower and a HARD lap
0.24 % slower, at the same stint age. Both lose less pace through a stint than the
SOFT, by 0.06 % and 0.10 % per 10 laps. An INTERMEDIATE lap is 7.3 % slower than a
dry SOFT lap and a WET lap 14.2 % slower. On those tyres laps get faster through
the stint (−0.9 % and −3.5 % per 10 laps). That is probably mostly the track
drying or the rain easing, not the tyres.

**Temperature and rain.** A track 10 °C hotter than the race's mean makes the lap
0.26 % slower. Rain falling doesn't slow the peak beyond the tyres (−0.20 %); it
goes to the shortfall instead. In rain, laps fall short of the peak about 7.5
times as often (logit +2.0) and by 1.5 times as much. On wet tyres laps fall short
by less (×0.44), as the tyres' own offsets already carry the slower limit, and how
often they fall short is barely identified (logit +1.3 ± 0.7).

**The field's pace.** The slope of a lap on the field's pace that lap stays at
0.13 with these terms in. Conditions, then, don't explain why the field pace moves
a driver's limit only weakly. The main fit's slope (0.17) is over 2011–2026, so the
two aren't like for like.

```@raw html
<div id="laps-conditions-table"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps_conditions/conditions.csv").then(r => r.text()).then(t => {
  const rows = t.trim().split("\n").slice(1).map(l => {
    const c = l.match(/("[^"]*"|[^,]+)/g).map(x => x.replace(/^"|"$/g, ""));
    return { group: c[1], label: c[2], unit: c[3], mean: Number(c[4]), sd: Number(c[5]) };
  });
  let h = "<table><thead><tr><th>Term</th><th>Mean</th><th>sd</th><th>Unit</th></tr></thead><tbody>";
  for (const r of rows) h += `<tr><td>${r.label}</td><td>${r.mean.toFixed(3)}</td><td>${r.sd.toFixed(3)}</td><td>${r.unit}</td></tr>`;
  document.getElementById("laps-conditions-table").innerHTML = h + "</tbody></table>";
});
</script>
```

### Wet-weather skill

```@raw html
<p>The wet deviations are centred across drivers. A negative one means the driver
loses less pace on wet tyres than the others do, relative to their own dry pace.
They spread with an sd of 0.73 % (± 0.17). Over the
<span class="diomedes-stat" data-path="assets/data/laps_conditions/wet_drivers.csv" data-stat="count" data-filter="wet_laps>=100" data-format="int">…</span>
drivers with at least 100 wet laps, they run from about −0.66 % (Magnussen, Norris, Vettel,
Verstappen) to +0.85 % (Albon). 15 of the 21 differ from zero by more than
1.645 sd.</p>
```

Read them with three caveats:
- **They include the car.** Car-seasons' own wet deviations couldn't be identified:
  in both probes, 2023 cars with 30–50 wet laps, all from two mixed races, came out
  several % faster or slower in the wet. So a driver's deviation also carries how
  well their car worked in the wet. Norris and Piastri share a McLaren.
- **Few races.** Wet-tyre laps are 4.7 % of the laps, from 18 races, so
  one race goes a long way.
- **Wet tyres, not rain.** The deviation is on intermediate and wet-tyre laps. A
  damp track on slicks counts as dry.

```@raw html
<div id="fig-laps-wet" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-wet", {
  data: { url: "assets/data/laps_conditions/wet_drivers.csv",
          format: { type: "csv", parse: { wet_laps: "number", mean: "number", sd: "number" } } },
  transform: [{ filter: "datum.wet_laps >= 100" },
              { calculate: "datum.mean - 1.645 * datum.sd", as: "lo" },
              { calculate: "datum.mean + 1.645 * datum.sd", as: "hi" }],
  width: "container", height: { step: 18 },
  encoding: {
    y: { field: "name", type: "nominal", title: null, sort: { field: "mean", order: "ascending" } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "wet_laps", title: "Wet-tyre laps" },
              { field: "mean", title: "Wet deviation (%)", format: ".2f" },
              { field: "sd", title: "sd", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative",
                       title: "Wet-tyre deviation, % of lap time (negative = better in the wet)" },
                  x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 50 },
      encoding: { x: { field: "mean", type: "quantitative" } } }
  ]
});
</script>
```

Drivers with at least 100 wet-tyre laps, 2021–2026: posterior mean and ±1.645 sd
(about a 90 % interval).

## The shape of a career on laps

The population age curve on clean laps: how pace changes with age for the
average driver, in % of lap time (negative = faster), with a 90% band. As in the
race model ([Careers](careers.md)), only its **shape** is identified: it has no
constant or linear part, because those can't be told apart from the drivers'
career slopes and the car-seasons. Read the bends, not the slope. Pace improves
steeply through the late teens, by about 0.40 % from 17 to 21, then stays within
about ±0.06 % from 21 to 33. From 34 on it jumps from one year to the next:
each age there rests on a handful of drivers (Alonso, Räikkönen, Hamilton,
Button), so it mostly follows their seasons, and it shows no clear decline.

```@raw html
<div id="fig-laps-age" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-age", {
  data: { url: "assets/data/laps/age_curve.csv",
          format: { type: "csv", parse: { age: "number", laps: "number", mean: "number", q05: "number", q95: "number" } } },
  width: "container", height: 300,
  encoding: { x: { field: "age", type: "quantitative", title: "Age", scale: { zero: false } } },
  layer: [
    { mark: { type: "area", opacity: 0.25 },
      encoding: { y: { field: "q05", type: "quantitative", title: "Pace relative to the curve's average (%)", scale: { reverse: true } },
                  y2: { field: "q95" } } },
    { mark: { type: "line", point: true },
      encoding: { y: { field: "mean", type: "quantitative" },
                  tooltip: [{ field: "age", title: "Age" }, { field: "laps", title: "Laps at this age" },
                            { field: "mean", title: "Curve (%)", format: ".2f" },
                            { field: "q05", title: "5%", format: ".2f" }, { field: "q95", title: "95%", format: ".2f" }] } }
  ]
});
</script>
```

The vertical axis is reversed, so faster is higher.

## Circuit slopes

Each car-season has a slope on three circuit dimensions, centred within the
season: how much faster or slower it is, relative to the field, at a circuit one
standard deviation above average. The dimensions are average lap speed (sd
20 km/h), street score (sd 0.35 on the 0–1 scale; see the
[circuit tables](https://github.com/aleccrowell/diomedes/tree/master/reference/circuits))
and air temperature at race time (sd 4.8 °C). Positive means relatively
**slower** there. The spread of the slopes across car-seasons is about 0.11 %
per sd for speed, 0.08 % for street score and 0.05 % for temperature. That's
smaller than in the 1996–2025 fit (0.16, 0.16 and 0.11 %). This fit covers fewer
seasons, and its race-to-race walk of each team's form can take up some of what
a fixed car-season effect left to the circuit slopes; which of the two matters
more hasn't been checked.

```@raw html
<div id="fig-laps-circuits" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-circuits", {
  data: { url: "assets/data/laps/cars.csv",
          format: { type: "csv", parse: { laps: "number", speed_mean: "number", street_mean: "number", temp_mean: "number" } } },
  transform: [
    { filter: "datum.laps >= 500" },
    { fold: ["speed_mean", "street_mean", "temp_mean"], as: ["dimension", "slope"] },
    { calculate: "{speed_mean: 'Speed', street_mean: 'Street score', temp_mean: 'Temperature'}[datum.dimension]", as: "dim" }
  ],
  width: "container", height: 220,
  mark: { type: "tick", opacity: 0.6, thickness: 2 },
  encoding: {
    x: { field: "slope", type: "quantitative", title: "Slope, % of lap time per sd (positive = relatively slower)" },
    y: { field: "dim", type: "nominal", title: null },
    tooltip: [{ field: "id", title: "Car-season" }, { field: "dim", title: "Dimension" },
              { field: "slope", title: "Slope (%)", format: ".2f" }, { field: "laps", title: "Laps" }]
  }
});
</script>
```

Car-seasons with at least 500 laps.

## Convergence

All four chains converge to one mode: τ<sub>δ</sub> = 4.03 and lap noise
σ = 0.521 % in every chain, with no divergences. They sampled at the same tree
depth (about 7.6, 220–233 leapfrog steps per draw) and the same step size
(0.038–0.039). The driver, car-season, peak and race-walk levels mix most slowly
(bulk ESS 310–380 from 4,000 draws). The shortfall, tyre, start, circuit and
field-pace terms all have an ESS above 1,000. The first full fit with these terms
diverged on nearly every iteration, because the field pace moved the clean-lap
edge through the shortfall. With the edge set by the base shortfall only, it
didn't diverge at all ([#113](https://github.com/aleccrowell/diomedes/issues/113)).
The table gives the chains' scalars.

```@raw html
<div id="laps-modes"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps/manifest.json").then(r => r.json()).then(m => {
  if (!m.modes) return;
  const f = (x, d) => (x === null || x === undefined) ? "–" : Number(x).toFixed(d);
  let h = "<table><thead><tr><th>Mode</th><th>Chains</th><th>τ<sub>δ</sub></th><th>Lap noise σ (%)</th>" +
          "<th>Lost-time lap probability</th><th>Lost-time lap mean (%)</th><th>Mean log density</th>" +
          "<th>Max R-hat within mode</th></tr></thead><tbody>";
  for (const md of m.modes) {
    h += `<tr><td>${md.label}</td><td>${md.chains.join(", ")}</td><td>${f(md.scalars.tau_delta, 2)}</td>` +
         `<td>${f(md.scalars.sigma, 3)}</td><td>${f(md.scalars.pi_base, 3)}</td><td>${f(md.scalars.lambda_base, 2)}</td>` +
         `<td>${f(md.mean_logjoint, 0)}</td><td>${md.convergence ? f(md.convergence.max_rhat, 3) : "–"}</td></tr>`;
  }
  h += "</tbody></table>";
  const names = { drivers: "Drivers", cars: "Car-seasons", tyremgmt_drivers: "Driver tyre management",
                  tyremgmt_cars: "Car tyre management", starts_drivers: "Driver starts", starts_cars: "Car starts" };
  for (const e of (m.effects_by_mode || [])) {
    h += `<p>Effects, mode ${e.modes}:</p><table><thead><tr><th></th><th>n</th><th>Correlation</th>` +
         "<th>Largest difference</th><th>Median difference / sd</th><th>Largest difference / sd</th></tr></thead><tbody>";
    for (const k of Object.keys(names)) {
      const a = e[k]; if (!a) continue;
      h += `<tr><td>${names[k]}</td><td>${a.n}</td><td>${f(a.corr, 4)}</td><td>${f(a.max_diff, 3)}</td>` +
           `<td>${f(a.median_diff_sd, 2)}</td><td>${f(a.max_diff_sd, 2)}</td></tr>`;
    }
    h += "</tbody></table>";
  }
  document.getElementById("laps-modes").innerHTML = h;
});
</script>
```

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/laps/vs_race_cars.csv"><code>vs_race_cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/vs_race_drivers.csv"><code>vs_race_drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/drivers.csv"><code>drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/cars.csv"><code>cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/age_curve.csv"><code>age_curve.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/peak.csv"><code>peak.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/incidents.csv"><code>incidents.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/modes.csv"><code>modes.csv</code></a>; from the conditions fit, <a class="diomedes-data" data-path="assets/data/laps_conditions/conditions.csv"><code>conditions.csv</code></a> and <a class="diomedes-data" data-path="assets/data/laps_conditions/wet_drivers.csv"><code>wet_drivers.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
