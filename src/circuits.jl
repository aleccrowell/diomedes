# Circuit covariates for the lap model's car × circuit slopes (#14), from the hand-checked
# tables in reference/circuits/ (see its README).

const CIRCUIT_DIR = joinpath(dirname(@__DIR__), "reference", "circuits")

"""
    circuit_table(dir = CIRCUIT_DIR) -> DataFrame

One row per F1 race in the reference tables (1950–2025): `event_id` (as in
`fetch_results`, "season-round"), `season`, `circuit_id`, `length_km` (the layout
raced), `street` (the street score, 0–1) and `temp_c` (race-window air
temperature). Built from `race_weather.csv`, `layout_lengths.csv` and
`street_scores.csv`; pass it to `prepare_laps(…; circuits)`.
"""
function circuit_table(dir::AbstractString = CIRCUIT_DIR)
    rd(f) = CSV.read(joinpath(dir, f), DataFrame; stringtype = String)
    races, lengths, street = rd("race_weather.csv"), rd("layout_lengths.csv"), rd("street_scores.csv")
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
                    street = Float64[], temp_c = Float64[])
    for r in eachrow(races)
        push!(out, (string(r.season, "-", lpad(r.round, 2, '0')), r.season, r.circuit_id,
                    lookup(lengths, r.circuit_id, r.season; race = r.race).length_km,
                    lookup(street, r.circuit_id, r.season).street_score, r.temp_c))
    end
    return out
end
