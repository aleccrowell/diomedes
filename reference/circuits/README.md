# Circuit reference tables

Hand-checked circuit and race metadata for the lap model's car × circuit terms
([#14](https://github.com/aleccrowell/diomedes/issues/14)) and the race model's circuit terms
([#81](https://github.com/aleccrowell/diomedes/issues/81)). F1 Grands Prix 1950–2025 (street scores
and layout lengths; the other tables 1996–2025 for now); circuit ids
are Jolpica/Ergast `circuitId`s. Every row cites its sources; rows marked "uncertain" or
"unconfirmed" say why in `notes`.

## `street_scores.csv` — how street-like a circuit is (one row per circuit era)

Three criteria, each 0–2:

| | 0 | 1 | 2 |
|---|---|---|---|
| **W** walls | mostly gravel or tarmac run-off | walls on some sections | barriers line most of the lap, little run-off |
| **R** roads | permanent racing facility | temporary circuit on non-public ground, or partly public roads | public roads open to traffic outside the event |
| **S** surface | racing surface | road layout, smooth or resurfaced for racing | road surface: bumps, crowns, painted lines, drains |

`street_score = (W + R + 2·S) / 8`, in [0, 1]. Surface is weighted double because the score
enters the model as a car-season slope, and surface (ride, mechanical grip) is what most
distinguishes how cars cope; walls mostly change how drivers manage risk.

The race model (#81) uses the criteria separately instead: **rough** = S as a car-season pace
slope, and **tight** = W on incident probability and loss size. R is kept as a record but is not
used there; it is what made airfields, the Indianapolis oval and the Nordschleife awkward to score
on one scale.

Before 1996, many circuits had trees, ditches, banks, houses or straw bales lining the lap
instead of barriers. These count as walls, and the notes say so. The 1950–95 rows were scored
from Wikipedia circuit and race articles. Each row was then source-checked against at least one
non-Wikipedia source, usually racingcircuits.info or the Motor Sport database. Ten scores changed
in the check; the notes say which and why.

## `layout_lengths.csv` — lap length of the layout raced (one row per layout era)

`length_km` per circuit and year range, matched to every race 1950–2025 (two layouts in one
year are separate rows, named in `race`: Bahrain/Sakhir 2020). Used to compute each race's
average lap speed. Where the circuit article, the race articles and statsf1.com disagree, the
row is marked uncertain. All 1996–2025 differences are under 1 %.

The 1950–95 rows were checked against four sources: the race article infobox, the circuit
article, the F1DB open database (github.com/f1db/f1db) and the Motor Sport database. statsf1.com
was not reachable. Where the race article alone disagreed with three agreeing sources, the three
were used; the notes say "kept". Rows with differences of 0.1 % or more are marked uncertain
(the largest: Detroit 1982, 3.7 %; Galvez 1974–81, 2.6 %).

## `start_times_1996_2004.csv` — documented local start times before 2005

The race data (Jolpica) has start times only from 2005. Sources: Atlas F1 and grandprix.com
timetables (2000–04); 1998–99 derived from formula1.com's pit-stop clock times
(`scripts/circuits/estimate_start_times.py`). 1996–97 are unconfirmed (14:00 local, or the
documented later slot): those seasons' timing stamps look normalised to a 14:00 start.

## `race_weather.csv` — start time and race-window weather (one row per race)

Built by `scripts/circuits/race_weather.py`.
- **Start time:** `local_start` and `utc_start`, and where it came from (`start_source`):
  - documented (1996–2004);
  - Jolpica 2005–06, whose times are local clock times wrongly marked as UTC;
  - Jolpica UTC from 2007.
- **`temp_c`:** mean air temperature over the two hours from the start (Open-Meteo archive, ERA5 reanalysis, at the circuit's coordinates). Air temperature, not track temperature.
- **`rain_mm`:** rainfall over the same window. Not used yet; see [#79](https://github.com/aleccrowell/diomedes/issues/79).
- **`temp_m2h` … `temp_p2h`:** the same with the window moved by ±1 or ±2 hours. Moving the start by an hour changes the temperature by 0.4 °C on average (95 % within 1.0 °C), against a spread of 4.8 °C between races, so uncertain start times matter little.
