# Limitations

- **Comparing eras rests on long chains of links.** Driver effects are comparable
  across decades only through team-mates and shared fields, and through the
  model's assumptions about how pace spread (κ) and incidents changed. Even with
  κ, the top of the all-time driver list is dominated by drivers from before
  1980 ([#20](https://github.com/aleccrowell/diomedes/issues/20)).
- **Car effects are within-season only.** They can't say whether the 2004
  Ferrari was faster than the 1988 McLaren, only how far each was ahead of its
  own field.
- **Separating a driver from their car** relies on team-mates and team changes.
  It is weakest for drivers who spent their careers with one team. In the published
  race fit the driver and car levels mix well (bulk ESS ≥ 696); the slowest direction
  is the split of a car's form between its season residual and the chassis walk
  ([#142](https://github.com/aleccrowell/diomedes/issues/142)).
- **A dominant car can pass some of its advantage to its star driver.** In simulations
  without walks, a car about 2 % ahead gave 0.15–0.21 % of it to the star who drove it
  ([#143](https://github.com/aleccrowell/diomedes/pull/143)). With walks it was 0.08 %
  ([#145](https://github.com/aleccrowell/diomedes/pull/145)). See
  [Checks against known answers](@ref).
- **Career slopes are conservative.** In simulations they are shrunk towards zero (slope
  of estimate on truth 0.2–0.7), so a fast improver is likely understated
  ([#143](https://github.com/aleccrowell/diomedes/pull/143)).
- **The age curve's peak isn't identified,** only its shape; career slopes are
  relative to the average driver ([#22](https://github.com/aleccrowell/diomedes/issues/22), [#24](https://github.com/aleccrowell/diomedes/issues/24)).
- **Pace noise against routine losses** (σ against π, λ) is weakly identified
  ([#32](https://github.com/aleccrowell/diomedes/issues/32)). The effects don't depend on it, but those parameters shouldn't be
  read as precise.
- **Model comparison is getting harder.** As the models grow, more rows have
  Pareto k > 0.7, which makes PSIS-LOO less reliable. Grouped K-fold has
  confirmed the era-term ranking, but hasn't been run on the latest terms
  ([#18](https://github.com/aleccrowell/diomedes/issues/18)).
- **In-season development is linear in the race model,** one trend per
  car-season ([#14](https://github.com/aleccrowell/diomedes/issues/14)). The lap model replaces it with a walk of each works'
  chassis form from race to race ([#108](https://github.com/aleccrowell/diomedes/issues/108)).
- **Car × circuit effects are three linear slopes** (speed, roughness,
  temperature). Other circuit traits, such as altitude, kerbs or the need for
  traction out of slow corners, are not modelled. Before 1979 the race-time
  temperatures come from a reanalysis with no satellite data, and many start
  times before 1996 are assumed from the circuit's or era's usual time
  ([#81](https://github.com/aleccrowell/diomedes/issues/81)).
- **Chassis against engine is weakly identified** where an engine had one
  user. The walks' engine levels also absorb some of their customers' pace, and
  the works and engine-builder tables involve judgement calls (takeovers,
  rebadged engines), marked uncertain in the
  [reference tables](https://github.com/aleccrowell/diomedes/tree/master/reference/cars) ([#95](https://github.com/aleccrowell/diomedes/issues/95)).
- **Incident proneness counts classified results only** (and retirements from
  1996 as partial gaps). It measures how often a driver lost time and still
  finished. A driver whose incidents usually ended in retirement looks less
  prone than they were; the [retirement model](../findings/retirements.md)
  covers those ([#101](https://github.com/aleccrowell/diomedes/issues/101)).
- **2026 is a partial season,** with the 17 races run when it was fitted; its car-season
  effects and walk states will move as the season finishes ([#99](https://github.com/aleccrowell/diomedes/issues/99)).
- **Sparse car-seasons stay imprecise.** A 1950s car with a few races is still
  identified mostly by those races: pooling over a works' seasons cut posterior
  sds by only about 10% ([#91](https://github.com/aleccrowell/diomedes/issues/91), [#95](https://github.com/aleccrowell/diomedes/issues/95)).
- **The race model's rain flag is per race,** dry, mixed or wet. It doesn't say when in the
  race it rained or how hard, and 51 of the calls are marked uncertain in the
  [reference table](https://github.com/aleccrowell/diomedes/tree/master/reference/circuits).
  The lap model's conditions terms have rain per lap, from 2021 only.
- **The lap model's shortfall can't tell choice from limits.** A driver managing
  pace (holding a position, saving tyres or fuel) beyond what the traffic and lead
  terms capture reads as falling short of their peak. Without the conditions
  terms, wet and mixed laps also go mostly to the shortfall
  ([#113](https://github.com/aleccrowell/diomedes/issues/113), [#116](https://github.com/aleccrowell/diomedes/issues/116)).
- **Tyre compounds are known only from 2021.** Before that the lap model can't
  separate compound choice (for example, the 2014–21 rule that the top 10
  qualifiers start on their Q2 tyres) from pace ([#12](https://github.com/aleccrowell/diomedes/issues/12)).
  From 2021 the timing data gives each lap's compound, track temperature and rain,
  and the published lap fit uses them, which is why it covers 2021–2026 only
  ([#116](https://github.com/aleccrowell/diomedes/issues/116), [#124](https://github.com/aleccrowell/diomedes/issues/124)).
- **Rallying lacks covariates.** The WRC fit has no road order or surface
  terms, and its driver effects are relative to a 641-driver field that is
  mostly amateurs ([#33](https://github.com/aleccrowell/diomedes/issues/33), [#28](https://github.com/aleccrowell/diomedes/issues/28)).
