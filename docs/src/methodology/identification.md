# The question and what is identifiable

## Crossed effects

A result depends on the driver, the car and the day. With one result per
driver per race, the race-time models write the expected result of driver ``d``
in car-season ``m`` at race ``r`` as

```math
\mu_{i} = \gamma_r + \kappa_{s(r)}\, a_d + b_m - \overline{\bigl(\kappa_{s(r)} a + b\bigr)}_r ,
```

where ``\gamma_r`` is the race's level, ``a_d`` the driver effect, ``b_m`` the
car-season effect, ``\kappa_s`` a per-season pace scale, and the last term is
the mean of the driver and car parts over the race's entrants. Driver effects
are in average-season units (κ = 1); car effects, which never leave their
season, are in that season's units. Driver and car effects are *crossed*, not nested: a driver drives
several cars over a career and each car has two (or more) drivers. Team-mates
separate a driver from their car, and moves between teams link cars.

## What the data can and can't tell apart

Each constraint below removes a direction the likelihood can't see. Without
them the posterior has ridges that the sampler wanders along, and the raw
effects are meaningless even when predictions are fine.

- **Drivers sum to zero.** Adding ``c`` to every driver and subtracting it from
  every car leaves every prediction unchanged. Driver effects are therefore
  relative to the average driver in the data.
- **Cars sum to zero within each season.** With race intercepts, shifting all
  of one season's cars by ``c`` and all of that season's races by ``-c`` changes
  nothing. Only differences between cars in the same season are identified, and
  **car effects can't be compared across seasons**. Drivers *can*: a driver who
  raced in 1985 and 1995 links the two seasons.
- **Race intercepts are field-centred**, so ``\gamma_r`` is the expected gap of
  the race's average entrant, not of an arbitrary reference car.
- **Age, period and cohort.** A driver's age is the year minus their birth year.
  A linear age effect ``\beta\cdot\text{age} = \beta\cdot\text{year} -
  \beta\cdot\text{birth year}`` is absorbed by the race intercepts (year) and the
  driver effects (birth year). Only the *curvature* of a population age curve is
  identified: whether drivers peak and decline, not at what age they peak
  ([#22](https://github.com/aleccrowell/diomedes/issues/22)). For the same reason, per-driver career slopes are centred across
  drivers. They show how a driver's trajectory differed from the average
  driver's, not whether drivers improve with age ([#24](https://github.com/aleccrowell/diomedes/issues/24)).
- **In-season development** trends are centred within each season. An
  improvement shared by the whole field is absorbed by the race intercepts
  ([#27](https://github.com/aleccrowell/diomedes/issues/27)).
- **Circuit slopes** are centred the same way, and each car-season's circuit
  covariates are also centred over its own races. A slope then only describes
  how a car's form changed between circuits, never its overall level, and a car
  that raced one kind of circuit has no slope at all ([#81](https://github.com/aleccrowell/diomedes/issues/81)).
- **Chassis and engine walks** are centred within each season, like the car
  effects. Chassis and engine separate only where they cross: customer engines,
  a works changing engine supplier, and 1960s chassis that ran several engines.
  Where an engine had one user for a stretch, the split rests on the walks'
  priors. Engine levels mix in their customers' pace: Ferrari's engine walk sits
  slightly slow throughout, likely because its customers were slow teams
  ([#95](https://github.com/aleccrowell/diomedes/issues/95)).
- **The works-team term** is identified by works teams against customers of the
  same engine in the same seasons. Where a works team also had the best chassis,
  the walks and the term share the credit, which is one reason its interval is
  wide ([#100](https://github.com/aleccrowell/diomedes/issues/100)).
- **Driver incident proneness** is centred across drivers, so the season and
  track terms keep describing the average driver. It's identified from which
  drivers' results fall in the loss tail. A driver with few results shrinks
  towards the average ([#101](https://github.com/aleccrowell/diomedes/issues/101)).

## Measuring convergence on what is identified

The constraints leave directions that only the prior sees, such as the mean of
the standardised driver effects. Those mix slowly and make the raw R-hat and
ESS look alarming even when every identified quantity has converged. In a WRC
trial every standardised driver coordinate had ESS ≈ 21 and R-hat 1.26, while
the centred effects agreed across chains to about 0.02% ([#25](https://github.com/aleccrowell/diomedes/issues/25)).

Sampling directly in sum-to-zero (Helmert) coordinates left the posterior
unchanged (driver correlation 0.99987) but made sampling worse: 698 divergences
against 73, and min ESS 17 against 21 ([#25](https://github.com/aleccrowell/diomedes/issues/25)). Instead, convergence is reported
on the identified quantities ([`identified_convergence`](@ref)): centred
effects, per-race incident rates and pace scales, and scalars ([#26](https://github.com/aleccrowell/diomedes/issues/26)).

## Centred and non-centred parametrisation

Most F1 drivers and every car-season have many results, so their effects are
well determined by the data. Non-centred effects ``a = \sigma z`` then tie
``\sigma`` to every ``z``, creating a funnel.

- **Cars:** with the pace scale, non-centred cars had a ridge between ``\kappa``
  and the car spread (correlation down to −0.89) and needed 309 leapfrog steps
  per draw. Centred cars removed the ridge and cut the steps by 38% ([#20](https://github.com/aleccrowell/diomedes/issues/20)).
- **Drivers:** in WRC, non-centred drivers gave ``\sigma_\text{comp}`` R-hat 1.13
  and ESS 37. Centring raised the scalars' min ESS from 106 to 1,168 in a trial
  ([#28](https://github.com/aleccrowell/diomedes/issues/28)). One outlier driver, about 29% off the pace on almost every stage,
  stayed bimodal until drivers got a Student-t(3) prior ([#28](https://github.com/aleccrowell/diomedes/issues/28)).
