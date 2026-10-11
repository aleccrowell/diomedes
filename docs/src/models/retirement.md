```@meta
DocTestSetup = :(using Diomedes; FIXTURE = joinpath(pkgdir(Diomedes), "test", "fixtures", "ergast"))
```

# Retirement model

[`retirement_effects`](@ref) / [`fit_retirement`](@ref): a competing-risks
survival model for how each start ends (#13). Each start is a record in laps:
`n` laps completed, then either the race ran out (censored) or the car retired
in lap `n + 1` with a cause ([`retirement_cause`](@ref)):

| Cause | Hazard has | Examples |
|---|---|---|
| `:mech` | a car-season **reliability** effect | engine, gearbox, hydraulics |
| `:incident` | a driver **incident-proneness** effect | accident, spun off |
| `:collision` | the driver effect with a learned loading `β_col` | contact with another car |
| `:other` | treated as censoring | out of fuel, driver unwell, "Retired" |

Non-starters and disqualifications are dropped. [`RETIRE_CAUSES`](@ref) lists the
modelled causes.

**Hazards** are per race distance, so short and long laps are comparable. Each
cause has a constant hazard over the race plus a first-lap jump (start crashes,
launch failures), both scaled by the start's linear predictor:

```
η_mech      = α_mech[season] + car-season reliability + race effect
η_incident  = α_inc[season]  + driver incident-proneness + race effect
η_collision = α_col[season]  + β_col · driver incident-proneness + race effect
```

The season levels `α` are random walks with Student-t(3) steps. Car effects are
centred within season, race effects within season, and driver effects overall.
Effects are on the **log-hazard scale**: a car effect of +0.5 means a mechanical
failure rate 1.65× its season's average.

```jldoctest
julia> prepare_retirements(fetch_results(ErgastCSV(FIXTURE), 2019))
RetireData(40 starts: 3 mech, 0 incident, 2 collision, 35 censored; 20 drivers, 10 machine-seasons, 2 races, 1 seasons)
```

`scripts/retire_fit.jl` fits the full F1 history and writes car reliability,
driver incident-proneness and per-season retirement rates.
