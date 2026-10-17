# Data handling

## Outcome: % gap to the winner

The race-time outcome is the percentage gap to the winner on a log scale,

```math
y = 100 \log\frac{T}{T_\text{winner}} ,
```

so effects add across drivers and cars and read as "% of race time". The
z-scored outcome of the original model, by contrast, depends on how spread out
each field was.

## Lapped cars are interval-censored, not dropped

In F1, only lead-lap finishers have a race time: the original model, using
times alone, dropped the slower half of most fields. Lapped cars still carry
information. A car that completed ``l`` of the winner's ``L`` laps crossed the
line after the winner but less than one of its own laps later. At constant pace
its full-distance gap therefore lies in

```math
y \in \Bigl[\,100\log\tfrac{L}{l},\; 100\log\tfrac{L}{l-1}\Bigr) ,
```

and enters the likelihood as the probability of that interval. The intervals
widen for cars further behind:

```@raw html
<div id="fig-lapped-intervals" class="diomedes-figure"></div>
<script>
diomedesFigure("fig-lapped-intervals", {
  data: { sequence: { start: 1, stop: 6, as: "k" } },
  params: [{ name: "laps", value: 60, bind: { input: "range", min: 20, max: 80, step: 1, name: "race length L (laps) " } }],
  transform: [
    { calculate: "laps - datum.k", as: "l" },
    { calculate: "100 * log(laps / datum.l)", as: "lo" },
    { calculate: "100 * log(laps / (datum.l - 1))", as: "hi" },
    { calculate: "datum.k + (datum.k == 1 ? ' lap down' : ' laps down') + ' (' + datum.l + ' of ' + laps + ' laps)'", as: "label" }
  ],
  width: "container", height: { step: 26 },
  encoding: {
    y: { field: "label", type: "nominal", sort: { field: "k" }, title: null, axis: { labelLimit: 220 } },
    tooltip: [{ field: "label", title: "Car" }, { field: "lo", title: "Gap from (%)", format: ".2f" },
              { field: "hi", title: "Gap to (%)", format: ".2f" }]
  },
  layer: [
    { mark: { type: "bar", height: 14, opacity: 0.6, color: "#4c78a8" },
      encoding: { x: { field: "lo", type: "quantitative", title: "Possible gap to the winner over the full distance (% of race time)", scale: { zero: true } },
                  x2: { field: "hi" } } },
    { mark: { type: "tick", thickness: 2, color: "#4c78a8" }, encoding: { x: { field: "lo", type: "quantitative" } } }
  ]
});
</script>
```

Over 1950–2019 this adds 6,858 lapped results to 6,276 timed ones ([#19](https://github.com/aleccrowell/diomedes/issues/19)). Lapped cars are
identified by their status ("+2 Laps"), not their position: an early version
counted 599 retirements with numeric positions as lapped ([#19](https://github.com/aleccrowell/diomedes/issues/19)).

In the Ergast data, retirements are dropped. Whether that biases the effects is
tested separately (see [Retirements and selection bias](@ref)).

## Jolpica data, 1950–2025

The current fits use the Jolpica API, Ergast's successor, which continues the
same tables past 2019 ([#55](https://github.com/aleccrowell/diomedes/issues/55)). Over 1950–2019 it matches the Ergast dump: the same
drivers, cars and races, and 17 rows that differ (mostly disqualifications
Jolpica codes and the dump doesn't). Its lap times for 1996–2019 match the dump's
on all but 76 of 472,504 laps, single laps whose times differ, over half of them
in five races (Australia 2014; Bahrain, Italy and the United States 2018; Germany
2019). Jolpica also has 16 stray timings, each after a gap in a car's laps and
with no running position (Vettel, Australia 2014: laps 1–3, retired, then a lap
26). Each would make its car's laps unusable, so they are dropped: a car keeps its
laps up to its first gap. Two things change from
2023:

- **Lapped cars have their own time.** Jolpica records "Lapped" with no race
  time but the car's own elapsed time. From that and its laps, the car's pace
  over the full distance gives an exact gap, not an interval.
- **Retirement causes are mostly "Retired".** The retirement model takes these
  as retirements of unknown cause, through the total hazard with no split
  between causes ([A competing-risks model of retirements](@ref)).

The 1950–2025 race refit also adds retirements from 1996 as partial gaps,
measured 3 laps before the car stopped, from lap times (Ergast to 2019, Jolpica
after).

## Red-flag suspensions

Since 2005, a red-flagged race resumes and its official time includes the
stoppage. Canada 2011's winner took 244.7 minutes, of which about 123 were the
suspension ([#17](https://github.com/aleccrowell/diomedes/issues/17)). Suspensions are detected from lap times, per car: every car
carries the stoppage in exactly one lap. This flags exactly the 12 races
red-flagged and resumed in 2005–2019, with no false positives. Each detected
duration matches the excess in the official time to within 1–2 minutes ([#17](https://github.com/aleccrowell/diomedes/issues/17)).
Times are then racing times: official time minus suspension.

## The Indianapolis 500

The Indy 500 counted towards the championship from 1950 to 1960, but its field
mostly raced only there. It is excluded by default ([#19](https://github.com/aleccrowell/diomedes/issues/19)).

## Priors in the data's own units

Weakly informative priors for quantities in % are scaled by the dataset's gap
scale ``s``: the median absolute deviation of results from their race median.
``s`` is 1.15 for F1 and 4.9 for WRC ([#31](https://github.com/aleccrowell/diomedes/issues/31)). Priors written in F1 units left
WRC posteriors deep in the prior tails, with the mean loss 2.9 prior sds out
([#30](https://github.com/aleccrowell/diomedes/issues/30)). On F1 the rescaling changed almost nothing: driver correlation 0.99935
with the previous fit ([#30](https://github.com/aleccrowell/diomedes/issues/30)).

## Rallying

WRC special stages are "races": every competitor on a stage is timed, so there
are no lapped cars. Classes share stages: 1,612 of 1,664 stages have both
Rally1 and Rally2 cars. Cars are grouped by manufacturer × tier × season,
because the 1,546 entrant teams are too many to use as levels ([#10](https://github.com/aleccrowell/diomedes/issues/10)). Stage
results are much noisier than F1 races: pace noise 1.68% against 0.51%, and an
incident on about 15% of stages ([#10](https://github.com/aleccrowell/diomedes/issues/10)).

## Lap times

Lap times (1996–) and pit stops (2011–) feed the lap-time model and the
retirement selection check. They come from Jolpica, like the results (the published lap fit too; earlier
lap fits read 1996–2019 from the Ergast dump). Pit stops before 2011 are inferred from lap times
([`infer_pit_stops`](@ref)). Over 2011–2019, the inferred stops give the same
effects as the recorded ones (driver correlation 0.998) ([#12](https://github.com/aleccrowell/diomedes/issues/12)).
