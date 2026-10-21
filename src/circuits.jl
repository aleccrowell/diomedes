# Circuit covariates for the lap model's car × circuit slopes (#14) and the race
# model's circuit terms (#81), from the hand-checked tables in reference/circuits/
# (see its README).

const CIRCUIT_DIR = joinpath(dirname(@__DIR__), "reference", "circuits")

"""
    circuit_table(dir = CIRCUIT_DIR) -> DataFrame

One row per F1 race in the reference tables (1950–2025): `event_id` (as in
`fetch_results`, "season-round"), `season`, `circuit_id`, `length_km` (the layout
raced), `street` (the street score, 0–1), its walls and surface criteria `W` and
`S` (0–2), `temp_c` (race-window air temperature) and `condition` (dry, mixed or
wet). Built from `race_weather.csv`, `layout_lengths.csv`, `street_scores.csv`
and `wet_races.csv`; pass it to `prepare_laps(…; circuits)` or the race model
(`RaceCircuits`).
"""
function circuit_table(dir::AbstractString = CIRCUIT_DIR)
    rd(f) = CSV.read(joinpath(dir, f), DataFrame; stringtype = String)
    races, lengths, street = rd("race_weather.csv"), rd("layout_lengths.csv"), rd("street_scores.csv")
    wet = Dict((r.season, r.round) => r.condition for r in eachrow(rd("wet_races.csv")))
    function lookup(tab, cid, y; race = nothing)
        rows = tab[(tab.circuit_id .== cid) .& (tab.first_year .<= y) .& (tab.last_year .>= y), :]
        # two layouts in one year (Bahrain/Sakhir 2020): the row naming the race
        if nrow(rows) > 1 && race !== nothing && hasproperty(rows, :race)
            named = rows[[!ismissing(r) && occursin(lowercase(r), lowercase(race)) for r in rows.race], :]
            rows = nrow(named) == 1 ? named : rows[ismissing.(rows.race), :]
        end
        nrow(rows) == 1 || throw(ArgumentError("$(nrow(rows)) rows for $cid $y in the circuit tables"))
        return rows[1, :]
    end
    out = DataFrame(event_id = String[], season = Int[], circuit_id = String[], length_km = Float64[],
                    street = Float64[], W = Int[], S = Int[], temp_c = Float64[], condition = String[])
    for r in eachrow(races)
        haskey(wet, (r.season, r.round)) || throw(ArgumentError("no condition for $(r.season)-$(r.round) in wet_races.csv"))
        sc = lookup(street, r.circuit_id, r.season)
        push!(out, (string(r.season, "-", lpad(r.round, 2, '0')), r.season, r.circuit_id,
                    lookup(lengths, r.circuit_id, r.season; race = r.race).length_km,
                    sc.street_score, sc.W, sc.S, r.temp_c, wet[(r.season, r.round)]))
    end
    return out
end

"""
    RaceCircuits(g::GapData, circuits)

Race-level circuit covariates for the race model (#81), one row per race of `g`,
from a table like `circuit_table()` matched by `event_id`:

- `x`: the pace covariates of the car-season circuit slopes, standardised across
  the races: average speed (km/h), rough (the surface criterion S, 0–2) and
  race-window air temperature (°C); raw values in `x_raw`. Speed is the layout
  length × the winner's laps ÷ the winner's race time, averaged over the dry
  races on the same layout (circuit and length; all its races if none was dry),
  so it describes the circuit rather than the day: safety cars and rain don't
  make a circuit look slow.
- `track`: the incident covariates on logit π and log λ: tight (the walls
  criterion W ÷ 2, 0–1) and indicators of a mixed and a wet race.
"""
struct RaceCircuits
    x::Matrix{Float64}            # races × pace covariates, standardised across races
    x_raw::Matrix{Float64}
    x_names::Vector{Symbol}       # :speed, :rough, :temp
    track::Matrix{Float64}        # races × incident covariates
    track_names::Vector{Symbol}   # :tight, :mixed, :wet
end
function RaceCircuits(g::GapData, circuits::AbstractDataFrame)
    ct = Dict(r.event_id => r for r in eachrow(circuits))
    event = Dict(r.race_key => r.event_id for r in eachrow(g.rows))
    evs = [event[k] for k in g.races]
    missing_races = filter(e -> !haskey(ct, e), evs)
    isempty(missing_races) || throw(ArgumentError("races missing from the circuit table: $(join(missing_races, ", "))"))
    # the winner's laps: the most laps completed in the race (partial-gap and lapped rows have fewer)
    laps = Dict{String,Int}()
    for r in eachrow(g.rows)
        ismissing(r.laps) || (laps[r.race_key] = max(get(laps, r.race_key, 0), r.laps))
    end
    raw = [haskey(laps, k) ? ct[e].length_km * laps[k] / (m / 60) : NaN
           for (k, e, m) in zip(g.races, evs, g.race_minutes)]
    lay = [(ct[e].circuit_id, ct[e].length_km) for e in evs]
    dry = [ct[e].condition == "dry" for e in evs]
    function layout_speed(l)
        idx = findall(i -> lay[i] == l && !isnan(raw[i]), eachindex(lay))
        isempty(idx) && throw(ArgumentError("no race with lap counts on layout $l"))
        d = filter(i -> dry[i], idx)
        return mean(raw[isempty(d) ? idx : d])
    end
    lm = Dict(l => layout_speed(l) for l in unique(lay))
    x_raw = hcat([lm[l] for l in lay], [Float64(ct[e].S) for e in evs], [Float64(ct[e].temp_c) for e in evs])
    sd = std(x_raw; dims = 1)
    x = (x_raw .- mean(x_raw; dims = 1)) ./ ifelse.(sd .> 0, sd, 1.0)     # a constant covariate is 0
    track = hcat([ct[e].W / 2 for e in evs], [Float64(ct[e].condition == "mixed") for e in evs],
                 [Float64(ct[e].condition == "wet") for e in evs])
    return RaceCircuits(x, x_raw, [:speed, :rough, :temp], track, [:tight, :mixed, :wet])
end

"""
    CircuitRows(g::GapData, rc::RaceCircuits; train = nothing)

Per-row data for the car-season circuit slopes (#81), rows in GapData order
(timed, then lapped): the machine level of each row (`mach`) and its race's pace
covariates minus their mean over that car-season's rows (`xc`, rows × covariates).

As with the development trends (see `DevRows`), centring per car-season makes a
slope orthogonal to the car-season effect: it measures how the car's form
changed across circuits, and a car that raced only one kind of circuit has no
slope term. `train` (rows used in a fit) restricts the means to those rows.
"""
struct CircuitRows
    mach::Vector{Int}
    xc::Matrix{Float64}
end
function CircuitRows(g::GapData, rc::RaceCircuits; train = nothing)
    races = vcat(g.t_race, g.c_race)
    mach = vcat(g.t_mach, g.c_mach)
    X = rc.x[races, :]
    s, n = zeros(length(g.machines), size(X, 2)), zeros(length(g.machines))
    for i in eachindex(mach)
        (train === nothing || train[i]) || continue
        s[mach[i], :] .+= X[i, :]; n[mach[i]] += 1
    end
    return CircuitRows(mach, X .- (s ./ max.(n, 1))[mach, :])
end

"""
    circuit_slopes(τ_cx, u_cx, g::GapData, season_counts) -> Vector

Car-season circuit slopes (% per standard deviation of each covariate), stacked
by covariate (`[(k - 1)·n_machines + m]`): τ_cx[k]·(u - season mean u), centred
within season like the development trends, since a slope shared by a season's
cars is absorbed by the race intercepts.
"""
function circuit_slopes(τ_cx, u_cx, g::GapData, season_counts)
    nm, K, ns = length(g.machines), length(τ_cx), length(season_counts)
    groups = reduce(vcat, [g.mach_season .+ (k - 1) * ns for k in 1:K])
    return τ_cx[repeat(1:K; inner = nm)] .* center_by(u_cx, groups, repeat(season_counts, K))
end

"Per-row circuit term Σ_k cx[k, car]·xc[row, k] (see `CircuitRows`, `circuit_slopes`)."
function circuit_offsets(cx, cr::CircuitRows)
    nm = length(cx) ÷ size(cr.xc, 2)
    return sum(cx[(k - 1) * nm .+ cr.mach] .* cr.xc[:, k] for k in axes(cr.xc, 2))
end
