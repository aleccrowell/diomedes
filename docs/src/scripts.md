# Scripts

The scripts in `scripts/` run the package's analyses on the full data, write to
`output/` (gitignored), and document their arguments in a header comment. Run
them with `julia --project scripts/<name>.jl ...`.

| Script | Purpose |
|---|---|
| `legacy_parity.jl` | Checks the z-score port against the original Python pipeline |
| `gap_fit.jl` | Fits a gap or pace + loss model (F1 or WRC); `chain k` / `combine` |
| `compare_gap_fits.jl` | Lead-lap finishers only vs with lapped cars |
| `loo_compare.jl` | PSIS-LOO comparison of fitted gap models |
| `kfold.jl` | Grouped K-fold CV: `fold` runs, then `compare` |
| `mode_check.jl` | Whether a fit depends on where its chains start |
| `era_paths.jl` | Per-season incident rate, loss size and pace scale |
| `incident_diagnostic.jl` | Posterior incident probabilities per result, by season |
| `age_curve.jl` | The population career curve |
| `career_slopes.jl` | Drivers with the largest career slopes |
| `retire_fit.jl` | Fits the retirement model and writes reliability and incident tables |
| `selection_check.jl` | Whether dropping retirements biases the pace effects |
| `lap_fit.jl` | Fits the lap-time model, checkpointed, optionally warm-started |
| `lap_compare.jl` | Lap-model fits against each other or the race model |
| `export_site.jl` | Exports posterior summaries of a fitted race model for the site's Findings pages (`docs/src/assets/data/`) |

Model names (`pl_dur_rw_kappa`, ...) are parsed by [`model_spec`](@ref).
