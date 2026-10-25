# Driver vs car

The question behind the project: how much of a result was the driver, and how
much the car? For each season a driver raced, the model splits their expected
pace, relative to the average entrant, into two parts:

- **Driver part:** the driver's pace that season: their effect plus their
  career terms at that age (the population age curve and their career slope),
  scaled by that season's pace scale κ.
- **Car part:** the car-season's effect, relative to the other cars that season.

**Total** is their sum. All three are in % of race time for that season;
negative is faster. Points are posterior means and lines 90% intervals. Type a
name to see every season of a career; a driver who changed teams mid-season has
a row per car.

```@raw html
<div id="fig-explorer" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-explorer", {
  data: { url: "assets/data/race/driver_seasons.csv",
          format: { type: "csv", parse: { season: "number", rows: "number",
                    driver_mean: "number", driver_q05: "number", driver_q95: "number",
                    car_mean: "number", car_q05: "number", car_q95: "number",
                    total_mean: "number", total_q05: "number", total_q95: "number" } } },
  params: [{ name: "who", value: "Ayrton Senna", bind: { input: "text", placeholder: "driver name", name: "Driver " } }],
  transform: [
    { filter: "who != '' && indexof(lower(datum.name), lower(who)) >= 0" },
    // the "constructor" field can't be referenced in Vega expressions (it collides
    // with Object.prototype.constructor), so the team comes from the car id
    { calculate: "replace(datum.car, '_' + datum.season, '')", as: "team" },
    { calculate: "datum.name + ' ' + datum.season + ' (' + datum.team + ')'", as: "row" },
    { fold: ["driver_mean", "car_mean", "total_mean"], as: ["part_key", "mean"] },
    { calculate: "{driver_mean: 'driver', car_mean: 'car', total_mean: 'total'}[datum.part_key]", as: "part" },
    { calculate: "datum.part == 'driver' ? datum.driver_q05 : datum.part == 'car' ? datum.car_q05 : datum.total_q05", as: "lo" },
    { calculate: "datum.part == 'driver' ? datum.driver_q95 : datum.part == 'car' ? datum.car_q95 : datum.total_q95", as: "hi" }
  ],
  width: "container", height: { step: 24 },
  encoding: {
    y: { field: "row", type: "nominal", title: null, sort: { field: "season", op: "min" }, axis: { labelLimit: 280 } },
    yOffset: { field: "part", sort: ["driver", "car", "total"] },
    color: { field: "part", type: "nominal", title: "Part", sort: ["driver", "car", "total"],
             scale: { domain: ["driver", "car", "total"], range: ["#4c78a8", "#f58518", "#54a24b"] } },
    tooltip: [{ field: "name", title: "Driver" }, { field: "season", title: "Season" },
              { field: "car", title: "Car" }, { field: "rows", title: "Results" }, { field: "part", title: "Part" },
              { field: "mean", title: "Mean (%)", format: ".2f" },
              { field: "lo", title: "5%", format: ".2f" }, { field: "hi", title: "95%", format: ".2f" }]
  },
  layer: [
    { mark: { type: "rule", opacity: 0.7 },
      encoding: { x: { field: "lo", type: "quantitative", title: "% of race time vs the season's average entrant (negative = faster)" },
                  x2: { field: "hi" } } },
    { mark: { type: "point", filled: true, size: 45 }, encoding: { x: { field: "mean", type: "quantitative" } } }
  ]
});
</script>
```

## Reading it

- **Driver and car parts are measured on the same scale within a season**, so
  they compare directly: in Senna's Lotus seasons the driver part is about one
  and a half times the car part; at McLaren the car part is the larger.
- **Across seasons**, the total isn't a measure of absolute speed. It's relative
  to that season's field, and cars are only compared within their season (see
  [The question and what is identifiable](@ref)).
- **The driver part changes over a career** through κ and the career terms.
  Only the curvature of the age curve and each driver's slope relative to the
  average driver are identified, so read changes within a career, not the
  absolute level of one season against another driver's in another era.

## Download the data

```@raw html
<p><a class="diomedes-data" data-path="assets/data/race/driver_seasons.csv"><code>driver_seasons.csv</code></a>.
Columns are described in <a href="../data/">Data downloads</a>.</p>
```
