# Circuit reference tables

Hand-checked circuit and race metadata for the lap model's car × circuit terms
([#14](https://github.com/aleccrowell/diomedes/issues/14)) and the race model's circuit terms
([#81](https://github.com/aleccrowell/diomedes/issues/81)). F1 Grands Prix 1950–2025; circuit ids
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

## `start_times_1950_1995.csv` — documented local start times 1950–95

Same columns as the 1996–2004 table. Each race's start time is confirmed only where a source
documents it for that race, almost always a Motor Sport magazine race report. That gives 196 of
the 581 races. Every confirmation that rested on a search summary was then checked against the
fetched article, and one wrong time (Zandvoort 1985) was corrected. The other races are
unconfirmed: their time comes from the same circuit's documented years, or failing that the
era's European pattern, and the notes give the reason. Where a race started well after its
schedule (rain, a start-line crash, a restart), the time is the start of the race that counted
(e.g. Monza 1978 at 18:00), since that is the window the weather should cover. The riskiest
assumptions are races where morning against afternoon is in doubt: Jacarepaguá 1978–89, and
Buenos Aires, which swung between 13:00 and 16:30.

## `wet_races.csv` — documented race conditions (one row per race)

`condition` for every race 1950–2025, about the race that counted (from the start to the finish;
practice, qualifying, sprints and annulled starts don't count):
- **wet:** the track was wet for most of the race (most of the field on wet or intermediate
  tyres, or rain throughout);
- **mixed:** wet for part of the race (rain starting or stopping, a wet start drying out, a
  shower causing tyre changes or spins);
- **dry:** no rain or wet track documented during the race.

| | dry | mixed | wet |
|---|---|---|---|
| races | 981 | 105 | 63 |

Every wet or mixed row is backed by a report of that race: Wikipedia, Motor Sport (to 1984),
grandprix.com and German Wikipedia (1985–2005). Rows where sources conflict or the call is
borderline start their notes with "Uncertain:" (48 rows, mostly wet against mixed, or light rain
that may not have wetted the track). 206 dry rows rest only on no rain being mentioned. The rest
have a positive statement, such as an infobox reading dry or sunny, or a report saying so.

This flag, not ERA5, is the race model's wet covariate (#81). ERA5 rain over the race
(`rain_race_mm`) reaches 1 mm in only 49 % of the wet races and 29 % of the mixed ones, but also in
5 % of the dry ones.

## `race_weather.csv` — start time and race-window weather (one row per race)

Built by `scripts/circuits/race_weather.py`.
- **Start time:** `local_start` and `utc_start`, and where it came from (`start_source`):
  - documented (1950–2004);
  - Jolpica 2005–06, whose times are local clock times wrongly marked as UTC;
  - Jolpica UTC from 2007.
- **`temp_c`:** mean air temperature over the two hours from the start (Open-Meteo archive, ERA5 reanalysis, at the circuit's coordinates). Air temperature, not track temperature.
- **`rain_mm`:** rainfall over the same window.
- **`race_hours`, `rain_race_mm`:** the winner's race time, and rainfall from the start to the
  start plus that time (#81). 1950s races averaged 2.9 h (1990s: 1.6 h), so the two-hour window
  would miss the end of the race. Note that `rain_mm` also counts the hour before the start
  (ERA5 hourly rain is the total over the preceding hour); `rain_race_mm` does not.
- **ERA5 misses many wet races** (see `wet_races.csv`). A ~30 km grid cell can't place a downpour at one circuit in
  one hour: Estoril 1985 shows 0.6 mm and Fuji 1976 0.0 mm over the race. It does catch Spa 1966,
  the Nürburgring 1968, Monaco 1984 and Montreal 1981.
- **Before 1979,** ERA5 has no satellite data, so its weather is less reliable at a single
  point and hour, especially rain.
- **`temp_m2h` … `temp_p2h`:** the same with the window moved by ±1 or ±2 hours. Moving the start by an hour changes the temperature by 0.4 °C on average (95 % within 1.0 °C), against a spread of 4.8 °C between races, so uncertain start times matter little.
