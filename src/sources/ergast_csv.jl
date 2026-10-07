"""
    ErgastCSV(dir; include_indy500 = false)

F1 results from a local Ergast-format CSV dump (`results.csv`, `races.csv`,
`drivers.csv`, `constructors.csv`, `status.csv`). Useful offline and for checking
parity with the legacy Python pipeline, which used such a dump.

IDs use Ergast's `driverRef`/`constructorRef` strings so they match `JolpicaF1`.
If the dump has `lap_times.csv` (1996+), `suspended_ms` is each race's red-flag
suspension detected from lap times (see `suspension_ms`); it is `missing` for
races without lap times.
The Indianapolis 500 (1950–1960) is excluded unless `include_indy500 = true`
(see `INDY500`).
"""
struct ErgastCSV <: DataSource
    dir::String
    include_indy500::Bool
end
ErgastCSV(dir::AbstractString; include_indy500::Bool = false) = ErgastCSV(dir, include_indy500)

function fetch_results(src::ErgastCSV, seasons::AbstractVector{<:Integer})
    rd(name) = CSV.read(joinpath(src.dir, name), DataFrame; missingstring = "\\N")
    results = rd("results.csv")
    races = select(rd("races.csv"), :raceId, :year, :round, :name, :date)
    drivers = select(rd("drivers.csv"), :driverId, :driverRef, :forename, :surname, :dob)
    constructors = select(rd("constructors.csv"), :constructorId, :constructorRef)
    status = rd("status.csv")

    df = innerjoin(results, races; on = :raceId)
    filter!(:year => in(Set(seasons)), df)
    src.include_indy500 || filter!(:name => !=(INDY500), df)
    df = innerjoin(df, drivers; on = :driverId)
    df = innerjoin(df, constructors; on = :constructorId)
    df = innerjoin(df, status; on = :statusId)
    sort!(df, [:year, :round, :positionOrder])
    susp = ergast_suspensions(src.dir, Set(df.raceId))

    out = DataFrame(
        series = "f1",
        season = Int.(df.year),
        round = Int.(df.round),
        event_id = string.(df.year, "-", lpad.(df.round, 2, '0')),
        event_name = String.(df.name),
        event_date = Vector{Union{Missing,Date}}(maybedate.(df.date)),
        stage_id = "race",
        competitor_id = String.(df.driverRef),
        competitor_name = string.(df.forename, " ", df.surname),
        competitor_birth = Vector{Union{Missing,Date}}(maybedate.(df.dob)),
        codriver_id = Vector{Union{Missing,String}}(missing, nrow(df)),
        machine_id = String.(df.constructorRef),
        class = Vector{Union{Missing,String}}(missing, nrow(df)),
        manufacturer = Vector{Union{Missing,String}}(missing, nrow(df)),
        entrant = Vector{Union{Missing,String}}(missing, nrow(df)),
        time_ms = Vector{Union{Missing,Float64}}(maybefloat.(df.milliseconds)),
        suspended_ms = Vector{Union{Missing,Float64}}([get(susp, id, missing) for id in df.raceId]),
        position = Vector{Union{Missing,Int}}(maybeint.(df.position)),
        laps = Vector{Union{Missing,Int}}(maybeint.(df.laps)),
        status = String.(df.status),
        classified = .!ismissing.(df.position),
    )
    return validate_results(out)
end

# Suspended time per raceId from lap_times.csv, for races with lap data.
function ergast_suspensions(dir, race_ids)
    path = joinpath(dir, "lap_times.csv")
    isfile(path) || return Dict{Int,Float64}()
    lt = CSV.read(path, DataFrame; select = [:raceId, :driverId, :lap, :milliseconds])
    filter!(:raceId => in(race_ids), lt)
    sort!(lt, [:raceId, :driverId, :lap])
    out = Dict{Int,Float64}()
    for g in groupby(lt, :raceId)
        out[first(g.raceId)] = suspension_ms([Float64.(c.milliseconds) for c in groupby(g, :driverId)])
    end
    return out
end

# Lap times and pit stops (#12). Ergast has lap times from 1996 and pit stops
# from 2011. Ids as in `fetch_results`.
function ergast_event_ids(src::ErgastCSV, seasons)
    rd(name) = CSV.read(joinpath(src.dir, name), DataFrame; missingstring = "\\N")
    races = filter(:year => in(Set(seasons)), select(rd("races.csv"), :raceId, :year, :round, :name))
    src.include_indy500 || filter!(:name => !=(INDY500), races)
    drivers = select(rd("drivers.csv"), :driverId, :driverRef)
    return races, drivers
end

function fetch_laps(src::ErgastCSV, seasons::AbstractVector{<:Integer})
    races, drivers = ergast_event_ids(src, seasons)
    lt = CSV.read(joinpath(src.dir, "lap_times.csv"), DataFrame; missingstring = "\\N",
                  select = [:raceId, :driverId, :lap, :position, :milliseconds])
    df = innerjoin(innerjoin(lt, races; on = :raceId), drivers; on = :driverId)
    sort!(df, [:year, :round, :driverRef, :lap])
    return validate_schema(DataFrame(series = "f1", season = Int.(df.year),
                                     event_id = string.(df.year, "-", lpad.(df.round, 2, '0')),
                                     competitor_id = String.(df.driverRef), lap = Int.(df.lap),
                                     time_ms = Float64.(df.milliseconds),
                                     position = Vector{Union{Missing,Int}}(maybeint.(df.position))),
                           LAP_SCHEMA, "laps")
end

function fetch_pit_stops(src::ErgastCSV, seasons::AbstractVector{<:Integer})
    races, drivers = ergast_event_ids(src, seasons)
    ps = CSV.read(joinpath(src.dir, "pit_stops.csv"), DataFrame; missingstring = "\\N",
                  select = [:raceId, :driverId, :lap, :milliseconds])
    df = innerjoin(innerjoin(ps, races; on = :raceId), drivers; on = :driverId)
    sort!(df, [:year, :round, :driverRef, :lap])
    return validate_schema(DataFrame(series = "f1", season = Int.(df.year),
                                     event_id = string.(df.year, "-", lpad.(df.round, 2, '0')),
                                     competitor_id = String.(df.driverRef), lap = Int.(df.lap),
                                     duration_ms = Vector{Union{Missing,Float64}}(maybefloat.(df.milliseconds))),
                           PIT_SCHEMA, "pit stops")
end
