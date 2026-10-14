# Data downloads

Every chart on these pages is drawn from the CSV files below: posterior
summaries exported from named fits by `scripts/export_site.jl`. Each directory has
a `manifest.json` recording which fit the tables came from, its data, its
convergence, and when and from which commit they were exported. Effects in % are
on the log scale `100·log(T / T_ref)`; negative is faster. Intervals are 90%
posterior intervals (`q05`, `q95`). `sd` is the posterior standard deviation.

## Race-model files

```@raw html
<p>F1 1950–2025 (Jolpica API), best pace + loss model. The manifest also records
the modes the chains settled in, convergence by group, and how far the effects
differ between modes.
<a class="diomedes-data" data-path="assets/data/race/manifest.json">manifest.json</a></p>
```

```@raw html
<table>
<thead><tr><th>File</th><th>One row per</th><th>Columns</th></tr></thead>
<tbody>
<tr><td><a class="diomedes-data" data-path="assets/data/race/drivers.csv">drivers.csv</a></td><td>driver</td>
<td><code>id</code>, <code>name</code>; <code>rows</code>: results used; <code>first_season</code>, <code>last_season</code>;
<code>mean</code>, <code>sd</code>, <code>q05</code>, <code>q95</code>: driver effect, % of race time in average-season units (κ = 1), relative to the average driver;
<code>slope_mean</code>, <code>slope_sd</code>: career slope, % per decade of age, relative to the average driver's trajectory</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/race/cars.csv">cars.csv</a></td><td>car-season</td>
<td><code>id</code> (constructor_season), <code>constructor</code>, <code>season</code>, <code>rows</code>;
<code>mean</code>, <code>sd</code>, <code>q05</code>, <code>q95</code>: car effect, % of race time relative to the other cars in the same season;
<code>dev_mean</code>, <code>dev_sd</code>: in-season development trend, % from the season's first round to its last, relative to the field</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/race/ranks.csv">ranks.csv</a></td><td>driver with ≥ 20 results</td>
<td><code>id</code>, <code>name</code>, <code>rows</code>; <code>p_top1</code>, <code>p_top3</code>, <code>p_top10</code>: share of the
posterior draws in which the driver ranks fastest, in the top 3, in the top 10 among these drivers;
<code>rank_median</code>, <code>rank_q05</code>, <code>rank_q95</code></td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/race/driver_seasons.csv">driver_seasons.csv</a></td><td>driver × car-season raced</td>
<td><code>driver</code>, <code>name</code>, <code>season</code>, <code>car</code>, <code>constructor</code>, <code>rows</code>;
<code>driver_*</code>: driver part, κ_season × (driver effect + career terms averaged over that season's results);
<code>car_*</code>: car part (car-season effect); <code>total_*</code>: their sum. All in that season's % units
(mean, q05, q95), relative to the season's average entrant</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/race/seasons.csv">seasons.csv</a></td><td>season</td>
<td><code>season</code>, <code>races</code>; <code>pi_*</code>: probability that a result includes a routine loss;
<code>lambda_*</code>: mean routine loss, % of race time (both at average race duration); <code>kappa_*</code>: pace scale κ</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/race/seasons_by_mode.csv">seasons_by_mode.csv</a></td><td>mode × season</td>
<td><code>mode</code> (chains grouped by where the loss mixture settled; see <code>modes.csv</code>), then the columns of
<code>seasons.csv</code> over that mode's draws</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/race/modes.csv">modes.csv</a></td><td>chain × quantity</td>
<td><code>chain</code>, <code>mode</code>; <code>quantity</code>, <code>label</code>: <code>sigma</code> (pace noise σ, %),
<code>pi_base</code> (routine-loss probability), <code>lambda_base</code> (routine-loss mean, %), <code>rho</code> (big-loss share of
incidents), <code>lambda_big</code> (big-loss mean, %), from the intercepts (average race duration, before the season paths);
<code>mean</code>, <code>q05</code>, <code>q95</code> over that chain's draws</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/race/age_curve.csv">age_curve.csv</a></td><td>age (years)</td>
<td><code>age</code>, <code>rows</code>; <code>mean</code>, <code>q05</code>, <code>q95</code>: population career curve, % of race time.
Only its shape is identified: it has no constant or linear part</td></tr>
</tbody></table>
```

## Retirement-model files

```@raw html
<p>F1 1950–2025 (Jolpica API), competing-risks hazards. Effects are on the log-hazard scale:
+0.5 is a rate 1.65× the average.
<a class="diomedes-data" data-path="assets/data/retire/manifest.json">manifest.json</a></p>
<table>
<thead><tr><th>File</th><th>One row per</th><th>Columns</th></tr></thead>
<tbody>
<tr><td><a class="diomedes-data" data-path="assets/data/retire/cars.csv">cars.csv</a></td><td>car-season</td>
<td><code>id</code>, <code>constructor</code>, <code>season</code>; <code>effect</code>, <code>sd</code>: mechanical-failure log-hazard relative to the season;
<code>starts</code>, <code>mech_dnf</code>: starts and mechanical retirements</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/retire/drivers.csv">drivers.csv</a></td><td>driver</td>
<td><code>id</code>, <code>name</code>; <code>effect</code>, <code>sd</code>: incident-proneness log-hazard relative to the average driver;
<code>starts</code>, <code>incidents</code>, <code>collisions</code></td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/retire/seasons.csv">seasons.csv</a></td><td>season</td>
<td><code>mech</code>, <code>incident</code>, <code>collision</code>: modelled retirements per start over a race distance, by cause;
<code>*_obs</code>: observed rates</td></tr>
</tbody></table>
```

## Lap-model files

```@raw html
<p>F1 1996–2025, clean laps (Ergast to 2019, Jolpica from 2020), pit laps inferred from lap times. Effects are % of the race's median clean lap.
<a class="diomedes-data" data-path="assets/data/laps/manifest.json">manifest.json</a></p>
<table>
<thead><tr><th>File</th><th>One row per</th><th>Columns</th></tr></thead>
<tbody>
<tr><td><a class="diomedes-data" data-path="assets/data/laps/drivers.csv">drivers.csv</a></td><td>driver</td>
<td><code>id</code>, <code>name</code>, <code>mean</code>, <code>sd</code>, <code>laps</code>; <code>first_season</code>, <code>last_season</code>;
<code>slope_mean</code>, <code>slope_sd</code>: career slope, % per decade of age, relative to the average driver</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/laps/cars.csv">cars.csv</a></td><td>car-season</td>
<td><code>id</code>, <code>mean</code>, <code>sd</code>, <code>laps</code>; <code>dev_mean</code>, <code>dev_sd</code>: in-season development, % from the season's first round to its last, relative to the field</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/laps/vs_race_drivers.csv">vs_race_drivers.csv</a></td><td>driver in both models</td>
<td><code>lap_mean</code>, <code>lap_sd</code>: lap model; <code>race_mean</code>, <code>race_sd</code>: race model (whole career)</td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/laps/vs_race_cars.csv">vs_race_cars.csv</a></td><td>car-season in both models</td>
<td><code>lap_mean</code>, <code>lap_sd</code>, <code>race_mean</code>, <code>race_sd</code>, <code>constructor</code>, <code>season</code>, <code>laps</code></td></tr>
</tbody></table>
```

## WRC files

```@raw html
<p>WRC 2018–2025 stage times, top two tiers. Effects are % of stage time.
<a class="diomedes-data" data-path="assets/data/wrc/manifest.json">manifest.json</a></p>
<table>
<thead><tr><th>File</th><th>One row per</th><th>Columns</th></tr></thead>
<tbody>
<tr><td><a class="diomedes-data" data-path="assets/data/wrc/drivers.csv">drivers.csv</a></td><td>driver</td>
<td><code>id</code> (wrc.com id), <code>name</code>; <code>mean</code>, <code>sd</code>, <code>q05</code>, <code>q95</code>: relative to the average of all drivers in the file; <code>stages</code></td></tr>
<tr><td><a class="diomedes-data" data-path="assets/data/wrc/cars.csv">cars.csv</a></td><td>manufacturer × tier × season</td>
<td><code>id</code>, <code>manufacturer</code>, <code>tier</code>, <code>season</code>; <code>mean</code>, <code>sd</code>, <code>q05</code>, <code>q95</code>: within season; <code>stage_times</code></td></tr>
</tbody></table>
```
