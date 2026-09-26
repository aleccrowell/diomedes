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
using DataFrames
using Dates
using HTTP
using JSON3
using LinearAlgebra
using Random
using ReverseDiff
using SHA
using Statistics
using Turing

include("schema.jl")
include("cache.jl")
include("sources/sources.jl")
include("sources/ergast_csv.jl")
include("sources/jolpica.jl")
include("sources/wrc.jl")
include("prepare.jl")
include("models.jl")

export RESULT_SCHEMA, empty_results, validate_results
export DataSource, fetch_results, ErgastCSV, JolpicaF1, WRCTiming
export ModelData, prepare, standardise_times!
export crossed_effects, fit_effects, effects_table, progress_logger,
       convergence_summary

end
