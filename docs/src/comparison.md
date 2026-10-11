# Model comparison

The pace models are compared by expected log predictive density (elpd) on the
same rows: every timed and lapped row of the data.

## PSIS-LOO

[`pointwise_loglik`](@ref) recomputes each row's log-likelihood for every
posterior draw. Turing can't do this itself, because the likelihood is added in
one fused call. The result is shaped (draws, chains, rows) for PosteriorStats'
`loo`. Models are named as in [`model_spec`](@ref), so a model's terms are read
from its name:

```julia
ll = pointwise_loglik(chain, g, "pl_dur_rw_kappa_age_big_slope_dev")
```

`scripts/loo_compare.jl` compares every fitted model with saved chains.

## Grouped K-fold

PSIS-LOO is unreliable for rows with Pareto k > 0.7: about 3% of rows, mostly
extreme results, and more as the models grow. K-fold refits the model without
each fold and scores the held-out rows exactly.

- [`kfold_folds`](@ref) deals each race's rows round-robin into K folds, so each
  race keeps most of its rows in training and its intercept stays determined.
- [`subset_gaps`](@ref) gives the training data. Its indices line up with the
  full data.
- [`heldout_loglik`](@ref) scores the held-out rows, and [`elpd_rows`](@ref)
  turns log-likelihood draws into per-row elpd.

`scripts/kfold.jl` runs folds as independent processes and compares models with
paired differences over the same folds.
