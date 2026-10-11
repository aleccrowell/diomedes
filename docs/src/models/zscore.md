```@meta
DocTestSetup = :(using Diomedes; FIXTURE = joinpath(pkgdir(Diomedes), "test", "fixtures", "ergast"))
```

# Z-score model

The original model, ported faithfully from the 2020 TensorFlow Probability version
(kept in `legacy/` as the reference; `scripts/legacy_parity.jl` checks the port).

```
z = competitor effect + machine-season effect + noise,   noise ~ N(0, σ_y)
```

- **Outcome:** each row's time standardised within its stage
  ([`standardise_times!`](@ref): population sd, as the legacy sklearn pipeline did).
- **Rows:** only rows with a time. In F1 that means **lead-lap finishers only**,
  which silently drops the slower half of most fields. The gap models fix this.
- **Effects:** non-centred, competitor effects sum to zero, no intercept by
  default (z has mean exactly 0 within every stage). `σ_y` is fixed at 1 as in the
  legacy model; `σ_y = nothing` learns it.
- **Speed:** the likelihood is computed from per-(driver, car-season) sufficient
  statistics, so a full F1 fit takes under 2 minutes on a Raspberry Pi 5.

```jldoctest
julia> d = prepare(fetch_results(ErgastCSV(FIXTURE), 2019))
ModelData(17 obs, 12 competitors, 8 machine-seasons)
```

```julia
chain = fit_effects(d; n_chains = 4)
convergence_summary(chain)          # worst R-hat and bulk ESS
eff = effects_table(chain, d)       # (; competitors, machines), in z units
```

See [`crossed_effects`](@ref) for every change from the legacy model and the
reasons for it.
