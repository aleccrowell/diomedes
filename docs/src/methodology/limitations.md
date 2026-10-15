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
  It is the main remaining source of slow mixing ([#20](https://github.com/aleccrowell/diomedes/issues/20)), and it is weakest
  for drivers who spent their careers with one team.
- **The age curve's peak isn't identified,** only its shape; career slopes are
  relative to the average driver ([#22](https://github.com/aleccrowell/diomedes/issues/22), [#24](https://github.com/aleccrowell/diomedes/issues/24)).
- **Pace noise against routine losses** (σ against π, λ) is weakly identified
  ([#32](https://github.com/aleccrowell/diomedes/issues/32)). The effects don't depend on it, but those parameters shouldn't be
  read as precise.
- **Model comparison is getting harder.** As the models grow, more rows have
  Pareto k > 0.7, which makes PSIS-LOO less reliable. Grouped K-fold has
  confirmed the era-term ranking, but hasn't been run on the latest terms
  ([#18](https://github.com/aleccrowell/diomedes/issues/18)).
- **In-season development is linear,** one trend per car-season. A random walk
  over rounds, and car × circuit interactions, are not modelled ([#14](https://github.com/aleccrowell/diomedes/issues/14)).
- **Tyre compounds aren't in the data.** The lap model can't separate compound
  choice (for example, the 2014–21 rule that the top 10 qualifiers start on
  their Q2 tyres) from pace ([#12](https://github.com/aleccrowell/diomedes/issues/12)).
- **Rallying lacks covariates.** The WRC fit has no road order or surface
  terms, and its driver effects are relative to a 641-driver field that is
  mostly amateurs ([#33](https://github.com/aleccrowell/diomedes/issues/33), [#28](https://github.com/aleccrowell/diomedes/issues/28)).
