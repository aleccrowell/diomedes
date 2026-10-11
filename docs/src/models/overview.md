# Models overview

Every model in the package has the same core:

```
outcome = race (or stage) level + driver effect + machine-season effect + noise
```

They differ in what the outcome is, which rows they can use, and how they model
the noise.

| Model | Rows used | Noise | Status |
|---|---|---|---|
| [Z-score model](@ref) | lead-lap finishers | Gaussian, σ fixed at 1 | port of the original Python model, kept as the parity reference |
| [Gap model](@ref "Gap model: Student-t noise") | finishers, lapped cars as intervals | Student-t(4) or Gaussian | superseded by pace + loss |
| [Pace + loss](@ref "Pace + loss model") | as the gap model | Gaussian pace noise + exponential incident losses | **the main model**; many optional terms |
| [Retirement model](@ref) | every start | competing-risks hazards | complements the pace models |
| [Lap-time model](@ref) | clean laps, F1 1996– | pace + loss, per lap | in development (#12) |

## Which pace model to use

The best race-time model by PSIS-LOO (as of #14) is the pace + loss model with
every term on:

```julia
fit_paceloss(g; loss_duration = true, era = :rw, pace_scale = true, age = true,
             big_loss = true, slopes = true, dev = true)        # "pl_dur_rw_kappa_age_big_slope_dev"
```

Each term was added in its own issue and kept because it improved
out-of-sample fit (expected log predictive density, elpd). The issues hold the
comparisons:

| Term | Option | Model-name suffix | Issue |
|---|---|---|---|
| race duration in the loss | `loss_duration` | `_dur` | #9 |
| incident rate and size drift over seasons | `era = :rw` | `_rw` | #9 |
| pace scale by season | `pace_scale` | `_kappa` | #15 |
| population age curve | `age` | `_age` | #15 |
| big incident losses | `big_loss` | `_big` | #16 |
| per-driver career slopes | `slopes` | `_slope` | #15 |
| in-season development (+155.7 ± 16.4 elpd) | `dev` | `_dev` | #14 |

The scripts and the model-comparison helpers name models by these suffixes
(see [`model_spec`](@ref)). As the models grow, more rows get Pareto k > 0.7, so
PSIS-LOO becomes less reliable; confirm close calls with grouped K-fold
(see [Model comparison](@ref)).
