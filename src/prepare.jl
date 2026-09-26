# Turning a results table into model inputs.

"""
    standardise_times!(df; by = [:series, :event_id, :stage_id]) -> df

Add a column `:z` holding each row's `time_ms` standardised within its stage
(population standard deviation, as in the legacy sklearn pipeline). Rows without
a time get `missing`. If every timed row on a stage has the same time, their `z`
is 0.
"""
function standardise_times!(df::DataFrame; by = [:series, :event_id, :stage_id])
    transform!(groupby(df, by), :time_ms => zscore_skipmissing => :z)
    return df
end

function zscore_skipmissing(x)
    v = collect(skipmissing(x))
    isempty(v) && return fill(missing, length(x))
    μ = mean(v)
    s = std(v; corrected = false)
    s == 0 && (s = one(s))
    return [ismissing(xi) ? missing : (xi - μ) / s for xi in x]
end

"""
    ModelData

Inputs for the crossed-effects models: an outcome vector and integer indices
into sorted level labels.
"""
struct ModelData
    y::Vector{Float64}
    competitor::Vector{Int}
    machine::Vector{Int}
    competitors::Vector{String}   # level labels, indexed by `competitor`
    machines::Vector{String}      # level labels, indexed by `machine`
    rows::DataFrame               # the results rows used, aligned with `y`
end

Base.length(d::ModelData) = length(d.y)
function Base.show(io::IO, d::ModelData)
    print(io, "ModelData($(length(d)) obs, $(length(d.competitors)) competitors, ",
          "$(length(d.machines)) machine-seasons)")
end

"""
    prepare(results; machine_key) -> ModelData

Standardise times within each stage, keep rows with a time, and index
competitors and machine-seasons. `machine_key(row)` builds the machine grouping
label; it defaults to `"<machine_id>_<season>"`, i.e. a separate effect for each
car per season.

This reproduces the legacy pipeline: only rows with a recorded time are used,
which in F1 means lead-lap finishers only.
"""
function prepare(results::AbstractDataFrame;
                 machine_key = r -> string(r.machine_id, "_", r.season))
    df = standardise_times!(DataFrame(results))
    dropmissing!(df, :z)
    df.machine_key = [machine_key(r) for r in eachrow(df)]
    comp_levels = sort(unique(df.competitor_id))
    mach_levels = sort(unique(df.machine_key))
    comp_idx = Dict(l => i for (i, l) in enumerate(comp_levels))
    mach_idx = Dict(l => i for (i, l) in enumerate(mach_levels))
    return ModelData(
        Float64.(df.z),
        [comp_idx[c] for c in df.competitor_id],
        [mach_idx[m] for m in df.machine_key],
        comp_levels, mach_levels, df,
    )
end
