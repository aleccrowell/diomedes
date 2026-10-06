"""
    Diomedes

Bayesian models of competitor vs machine performance in motorsport.

Data flows through three layers:

1. **Sources** (`src/sources/`) fetch raw data from a public API or local files and
   normalise it into the common long-format results schema (`src/schema.jl`).
2. **Preparation** (`src/prepare.jl`) turns a results table into model inputs:
   a per-stage standardised outcome plus integer indices for competitors and
   machine-seasons.
3. **Models** (`src/models.jl`) are Turing models over those inputs and know
   nothing about which series the data came from.

Adding a new series means writing one source adapter; nothing downstream changes.
"""
module Diomedes

using ADTypes: AutoReverseDiff
using CSV
using ChainRulesCore: ChainRulesCore, NoTangent
using DataFrames
using Dates
using HTTP
using JSON3
using LinearAlgebra
using LogExpFunctions: log1mexp, logaddexp, logistic, logsumexp
import MCMCDiagnosticTools
using Random
using Unicode
using ReverseDiff
using SHA
using Serialization: serialize, deserialize
using SpecialFunctions: erf, erfcx, loggamma
using Statistics
using StatsFuns: normlogcdf, normlogpdf, tdistcdf, tdistlogccdf
using Turing

include("schema.jl")
include("cache.jl")
include("sources/sources.jl")
include("sources/suspensions.jl")
include("sources/ergast_csv.jl")
include("sources/jolpica.jl")
include("sources/wrc.jl")
include("prepare.jl")
include("models.jl")
include("noise.jl")
include("gap_model.jl")
include("paceloss.jl")
include("likelihoods.jl")
include("loo.jl")
include("kfold.jl")
include("retirement.jl")

export RESULT_SCHEMA, empty_results, validate_results
export DataSource, fetch_results, ErgastCSV, JolpicaF1, WRCTiming, wrc_top_tiers, wrc_tier, wrc_machine_key
export ModelData, prepare, standardise_times!
export crossed_effects, fit_effects, effects_table, progress_logger,
       convergence_summary
export GapData, prepare_gaps, gap_effects, fit_gaps, gap_effects_table, NormalNoise, StudentTNoise, PriorScale
export paceloss_effects, fit_paceloss, LossCovariates, pointwise_loglik, identified_convergence, model_spec, REGIME_STARTS, race_pace_scale, AgeCurveBasis, age_curve,
       CareerRows, driver_offsets, career_slopes, DevRows, dev_trends, rehash_chain!
export kfold_folds, subset_gaps, heldout_loglik, elpd_rows
export RETIRE_CAUSES, retirement_cause, RetireData, prepare_retirements, retirement_effects, fit_retirement

end
