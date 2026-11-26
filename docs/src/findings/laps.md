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

It is **in development**. The per-lap conditions are done for 2021 on
([#116](https://github.com/aleccrowell/diomedes/issues/116), [#124](https://github.com/aleccrowell/diomedes/issues/124)), in that separate fit.
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
- on wet-tyre laps (intermediates and wets): each driver's deviation from their dry
  pace, each car-season's, and an offset for each race run partly or fully on wet tyres;
- green-flag laps in the wet are kept even when the field is slow. The main fit drops
  every lap where the field's median is over 1.25× the race's, which also dropped many
  wet laps that weren't run behind a safety car. This keeps 263 more laps;
- a stint slope for each race, around the shared tyre curve
  ([#124](https://github.com/aleccrowell/diomedes/issues/124), [below](#How-fast-tyres-wore-at-each-race));
- each stint's true tyre age: a set used earlier in the weekend starts older, from
  the timing data's tyre ages;
- pit stops checked against the timing data's tyre changes
  ([#151](https://github.com/aleccrowell/diomedes/issues/151)): stops it shows are
  added, and stops it doesn't confirm under a safety car or red flag (mostly double
  records) are dropped;
- driver-age bins pooled at the ends so each has at least 3 drivers. Ages 40–45 were
  Alonso's alone, and the age curve there didn't converge.

Every group of terms converges: R-hat ≤ 1.032 and bulk ESS ≥ 141, with no divergences.
That includes the driver and car-season levels, which mixed slowly in earlier fits
along the ridge between a dominant driver and their car. The sampler draws each
car's form relative to its drivers, which leaves the model unchanged. The rankings on
this page stay on the 2011–2026 fit, which covers more seasons, and this section shows
only the conditions results.

```@raw html
<div id="manifest-laps-conditions"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps_conditions/manifest.json").then(r => r.json()).then(m => {
  const g = m.convergence.groups, w = g["wet drv"], c = g["wet car"], s = g["scalars"];
  document.getElementById("manifest-laps-conditions").innerHTML =
    `<p><small>Fit <code>${m.fit}</code>. Data: ${m.data}. Conditions: ${m.conditions}. Convergence: scalars max
     R-hat ${s.max_rhat} (min bulk ESS ${s.min_ess}), drivers' wet deviations ${w.max_rhat} (${w.min_ess})${c ? `,
     car-seasons' ${c.max_rhat} (${c.min_ess})` : ""}; over every group ${m.convergence.max_rhat}
     (${m.convergence.min_ess}), ${m.convergence.divergences} divergence(s). Exported ${m.date}.</small></p>`;
});
</script>
```

**Compounds.** Against the SOFT, a MEDIUM lap is 0.17 % slower and a HARD lap
0.25 % slower, at the same tyre age. Through a stint they lose pace at much the same
rate as the SOFT (MEDIUM +0.01 %, HARD −0.04 % per 10 laps). In the earlier fit
without per-race slopes and true tyre ages, the harder compounds seemed to wear more
slowly, by 0.06 % and 0.10 %; with them in, most of that difference is gone. In the average wet race, an INTERMEDIATE lap
is 7.9 % slower than a dry SOFT lap and a WET lap 12.3 % slower. On those tyres laps
get faster through the stint (−0.7 % and −1.0 % per 10 laps). That is probably mostly
the track drying or the rain easing, not the tyres.

**Temperature and rain.** A track 10 °C hotter than the race's mean makes the lap
0.28 % slower. Rain falling doesn't slow the peak beyond the tyres (−0.21 %); it
goes to the shortfall instead. In rain, laps fall short of the peak about 7 times
as often (logit +1.9) and by 1.7 times as much. On wet tyres laps fall short by
less than half as much (×0.43), as the tyres' own offsets already carry the slower limit.
How often they fall short is barely identified (logit +1.5 ± 0.5).

**The field's pace.** The slope of a lap on the field's pace that lap is
0.12 with these terms in. Conditions, then, don't explain why the field pace moves
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

### How fast tyres wore at each race

Track surface, temperature and the compounds Pirelli brings change how quickly every
tyre loses pace at a race. So each race has its own stint slope, around the shared
tyre curve and centred over the races, measured against each stint's age relative
to the race's average. The slopes spread with an sd of 0.40 % per 10 laps (± 0.02),
and 116 of the 134 differ from zero by more than 1.645 sd. Japan 2022 wore fastest,
at +2.1 % per 10 laps against the average race, then Barcelona 2026 (+1.1 %) and
Bahrain 2022 (+1.0 %). Canada 2024 wore slowest (−1.4 %), then Australia 2025 (−0.7 %).
Japan 2022 was run entirely on wet tyres and Canada 2024 and Australia 2025 partly,
so their slopes probably include the track drying or getting wetter as well.

A per-compound slope keyed to Pirelli's C-numbers (C1, the hardest, to C6) didn't do
this job ([#124](https://github.com/aleccrowell/diomedes/issues/124)). Once each
race's own level is taken out, the compounds wear alike, and what varies is the race.

```@raw html
<div id="fig-laps-race-deg" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-race-deg", {
  data: { url: "assets/data/laps_conditions/race_degradation.csv",
          format: { type: "csv", parse: { laps: "number", mean: "number", sd: "number" } } },
  transform: [{ calculate: "substring(datum.id, 0, 4) + ' ' + replace(datum.name, ' Grand Prix', '')", as: "race" },
              { calculate: "datum.mean - 1.645 * datum.sd", as: "lo" },
              { calculate: "datum.mean + 1.645 * datum.sd", as: "hi" }],
  width: "container", height: { step: 9 },
  encoding: {
    y: { field: "race", type: "nominal", title: null, sort: { field: "mean", order: "ascending" },
         axis: { labelFontSize: 8 } },
    tooltip: [{ field: "race", title: "Race" }, { field: "laps", title: "Laps" },
              { field: "mean", title: "Stint slope (% per 10 laps)", format: ".2f" },
              { field: "sd", title: "sd", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative",
                       title: "Race's stint slope against the average race, % per 10 laps (positive = tyres wore faster)" },
                  x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 25 },
      encoding: { x: { field: "mean", type: "quantitative" } } }
  ]
});
</script>
```

Every race, 2021–2026: posterior mean and ±1.645 sd.

### How wet each race was

Two wet races aren't alike: a drying track after an early shower is much quicker than
standing water. So each race with wet-tyre laps gets its own offset on them, centred
over those races. Without it the car-seasons' wet deviations take up how wet their
races were: in an earlier fit, every 2023 car came out 1–3 % slower in the wet, as
2023's wet laps come from two races. The offsets spread with an sd of 2.1 % (± 0.5).
Monaco 2023's wet laps were the slowest against the average wet race, by +4.3 %, and
Singapore 2022's the quickest, at −4.4 %.
For races run entirely on wet tyres (Turkey 2021, Japan 2022, São Paulo 2024), the
offset can't be told apart from the race's mean lap. It stays near zero with a wide sd.

```@raw html
<div id="laps-wet-races-table"></div>
<script>
fetch(documenterBaseURL + "/assets/data/laps_conditions/wet_races.csv").then(r => r.text()).then(t => {
  const rows = t.trim().split("\n").slice(1).map(l => {
    const c = l.match(/("[^"]*"|[^,]+)/g).map(x => x.replace(/^"|"$/g, ""));
    return { id: c[0], name: c[1], laps: Number(c[2]), mean: Number(c[3]), sd: Number(c[4]) };
  });
  let h = "<table><thead><tr><th>Race</th><th>Wet-tyre laps</th><th>Offset (%)</th><th>sd</th></tr></thead><tbody>";
  for (const r of rows) h += `<tr><td>${r.id.slice(0, 4)} ${r.name}</td><td>${r.laps}</td><td>${r.mean.toFixed(2)}</td><td>${r.sd.toFixed(2)}</td></tr>`;
  document.getElementById("laps-wet-races-table").innerHTML = h + "</tbody></table>";
});
</script>
```

### Wet-weather skill: drivers

```@raw html
<p>The drivers' wet deviations are centred across drivers. A negative one means the driver
loses less pace on wet tyres than the others do, relative to their own dry pace, in
the same car and the same race. They spread with an sd of 0.30 % (± 0.05). Over the
<span class="diomedes-stat" data-path="assets/data/laps_conditions/wet_drivers.csv" data-stat="count" data-filter="wet_laps>=100" data-format="int">…</span>
drivers with at least 100 wet laps, they run from −0.52 % (Verstappen) to +0.51 %
(Tsunoda). 9 of the 23 differ from zero by more than 1.645 sd.</p>
```

Taking the car and the race out shrinks them: in a fit with drivers' deviations only,
they spread more than twice as widely (0.73 %), as each driver carried their car's wet
performance and the races they ran in. Read them with two caveats:
- **Few races.** Wet-tyre laps are 5.0 % of the laps, from 17 races, so
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

### Wet-weather performance: cars

```@raw html
<p>Each of the
<span class="diomedes-stat" data-path="assets/data/laps_conditions/wet_cars.csv" data-stat="count" data-format="int">…</span>
car-seasons with a wet-tyre lap has its own deviation, centred over all of them and
shrunk towards zero by a shared spread (sd 0.46 % ± 0.06). Cars with few wet laps stay
close to zero with wide intervals. They run from −0.85 % (Racing Bulls 2026, from 45 wet
laps) to +0.94 % (Haas 2023, from 50); 18 of the 61 differ from zero by more than 1.645 sd.
A driver and their car can differ: Red Bull's 2024 car comes out slower than average
in the wet (+0.51 %), while Verstappen is the best of the drivers.</p>
```

```@raw html
<div id="fig-laps-wet-cars" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-laps-wet-cars", {
  data: { url: "assets/data/laps_conditions/wet_cars.csv",
          format: { type: "csv", parse: { wet_laps: "number", mean: "number", sd: "number" } } },
  transform: [{ calculate: "replace(datum.id, /_(\\d{4})$/, ' $1')", as: "car" },
              { calculate: "datum.mean - 1.645 * datum.sd", as: "lo" },
              { calculate: "datum.mean + 1.645 * datum.sd", as: "hi" }],
  width: "container", height: { step: 14 },
  encoding: {
    y: { field: "car", type: "nominal", title: null, sort: { field: "mean", order: "ascending" } },
    tooltip: [{ field: "car", title: "Car-season" }, { field: "wet_laps", title: "Wet-tyre laps" },
              { field: "mean", title: "Wet deviation (%)", format: ".2f" },
              { field: "sd", title: "sd", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.6 },
      encoding: { x: { field: "lo", type: "quantitative",
                       title: "Car's wet-tyre deviation, % of lap time (negative = better in the wet)" },
                  x2: { field: "hi" } } },
    { mark: { type: "point", filled: true },
      encoding: { x: { field: "mean", type: "quantitative" },
                  size: { field: "wet_laps", type: "quantitative", title: "Wet-tyre laps",
                          scale: { range: [15, 90] } } } }
  ]
});
</script>
```

Every car-season with a wet-tyre lap, 2021–2026: posterior mean and ±1.645 sd; the
point's size shows its wet-tyre laps.

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
<p><a class="diomedes-data" data-path="assets/data/laps/vs_race_cars.csv"><code>vs_race_cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/vs_race_drivers.csv"><code>vs_race_drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/drivers.csv"><code>drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/cars.csv"><code>cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/age_curve.csv"><code>age_curve.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/peak.csv"><code>peak.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/incidents.csv"><code>incidents.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps/modes.csv"><code>modes.csv</code></a>; from the conditions fit, <a class="diomedes-data" data-path="assets/data/laps_conditions/conditions.csv"><code>conditions.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps_conditions/wet_drivers.csv"><code>wet_drivers.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps_conditions/wet_cars.csv"><code>wet_cars.csv</code></a>, <a class="diomedes-data" data-path="assets/data/laps_conditions/wet_races.csv"><code>wet_races.csv</code></a> and <a class="diomedes-data" data-path="assets/data/laps_conditions/race_degradation.csv"><code>race_degradation.csv</code></a>. Columns are described in <a href="../data/">Data downloads</a>.</p>
```
