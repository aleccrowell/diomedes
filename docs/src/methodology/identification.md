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

## Ridges between a team's drivers and its car

Centring removes the directions no data can see, but some directions are seen only
weakly. In the lap model over 2021–2026, the slowest directions in the posterior were
driver–team clusters: Norris and Piastri against McLaren's race walk, Verstappen and
Pérez against Red Bull's. The data fix each car's form plus its drivers' levels, and
the split rests on few links (team moves, team-mates).

To find such directions, take each chain's posterior means, standardised by each
quantity's posterior sd, and run an SVD of their differences between chains. The
leading vectors name the quantities the chains disagree on together. For McLaren's
cluster it held 65 % of the between-chain spread
([#116](https://github.com/aleccrowell/diomedes/issues/116)).

The fix was to sample along the ridge rather than across it. `rwrel` samples each
walk state relative to its drivers, ``\tilde x = x + W a``, with ``W`` each
driver's share of the state's laps. This shear has Jacobian 1, so the model is
unchanged. Moving a team's drivers no longer drags its states with them, and the
minimum ESS of the drivers rose from 81 to 219 in the full fit. Adding the career
slopes to the shear (`rwrels`) moved the slow direction onto the age curve
instead, so it isn't used.

## Effects that rest on a few races

Effects measured on a small part of the data, such as wet-weather skill (about 5 %
of laps, from 16 races), can go wrong in three ways the well-measured effects don't
([#116](https://github.com/aleccrowell/diomedes/issues/116)):

- **They absorb what the races have in common.** Wet races differ: the lap model's
  per-race wet offsets spread with an sd of 2.4 % of lap time. Without them, a
  car-season with wet laps from one or two races took those races' severity:
  Williams 2023 came out at +2.6 % to +3.2 % against about −0.2 % in a raw same-lap
  comparison. Give such an effect a per-race term, partially pooled, so units are
  compared within a race.
- **Losses absorb weak signals.** In a pace + loss model a slower lap can be read as
  a slower limit or as a loss. A likelihood profile along a car's wet effect showed
  that with about 30 wet laps the losses take up a faster wet limit and the
  likelihood is nearly flat. Before trusting such an effect, profile the likelihood
  along it, and check that its posterior widens as its data shrink.
- **Centring a subset forces counterweights.** If only some units have the effect,
  centring within season makes the rest of the season's units balance one outlier.
  Centring over all units instead lets anything shared, such as a race's
  severity, load onto them. Choose the centring together with the terms that take up
  the shared part, and test it: the `wetallg` test checks that shifting every unit
  by a constant leaves the likelihood unchanged.

The lap model's wet-skill terms follow all three: per-race wet offsets (`wetrace`),
car-season effects for every car-season shrunk by a shared spread and centred
globally (`wetallg`), and a check that few-lap effects stay near zero with wide
intervals.
