# Common long-format results schema.
#
# One row = one competitor's outcome on one timed stage of one event.
#   F1:  event = Grand Prix, stage = "race" (one stage per event)
#   WRC: event = rally,      stage = special stage (~15-25 per event)
#
# `machine_id` is whatever the series treats as "the car": the constructor in F1,
# the vehicle model in rallying (normalised text; see `manufacturer` and `entrant`
# for the cleaner rally identifiers). Models pair it with `season` to form a
# machine-season effect, since cars change year to year.

const RESULT_SCHEMA = [
    :series        => String,                   # "f1", "wrc", "erc", ...
    :season        => Int,
    :round         => Int,
    :event_id      => String,                   # unique within series
    :event_name    => String,
    :event_date    => Union{Missing,Date},      # race / rally start date
    :stage_id      => String,                   # unique within event
    :competitor_id => String,                   # driver; stable across seasons
    :competitor_name => String,                 # display name
    :competitor_birth => Union{Missing,Date},   # date of birth (for age effects)
    :codriver_id   => Union{Missing,String},    # rallying only
    :machine_id    => String,                   # constructor / vehicle model
    :class         => Union{Missing,String},    # e.g. "Rally1", "Rally2"
    :manufacturer  => Union{Missing,String},    # rallying: car maker (e.g. "Toyota"); missing in F1
    :entrant       => Union{Missing,String},    # rallying: entering team (works or privateer); missing in F1
    :time_ms       => Union{Missing,Float64},   # official stage/race time; missing if not timed
    :suspended_ms  => Union{Missing,Float64},   # red-flag suspension included in time_ms (per race); missing = unknown
    :position      => Union{Missing,Int},
    :laps          => Union{Missing,Int},       # laps completed (circuit racing); missing for rally stages
    :status        => String,                   # source status text ("Finished", "+1 Lap", "DNF", ...)
    :classified    => Bool,                     # counted as a finisher by the series
]

"""
    empty_results()

A zero-row `DataFrame` with the columns and element types of `RESULT_SCHEMA`.
"""
empty_results() = DataFrame([name => T[] for (name, T) in RESULT_SCHEMA])

"""
    validate_results(df)

Throw if `df` is missing a schema column or has a column whose element type is
not compatible with the schema. Returns `df` so it can be used inline.
"""
function validate_results(df::AbstractDataFrame)
    for (name, T) in RESULT_SCHEMA
        hasproperty(df, name) || throw(ArgumentError("results table missing column :$name"))
        S = eltype(df[!, name])
        S <: T || throw(ArgumentError("column :$name has eltype $S, expected subtype of $T"))
    end
    return df
end
