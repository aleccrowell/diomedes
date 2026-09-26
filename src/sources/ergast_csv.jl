"""
    ErgastCSV(dir)

F1 results from a local Ergast-format CSV dump (`results.csv`, `races.csv`,
`drivers.csv`, `constructors.csv`, `status.csv`). Useful offline and for checking
parity with the legacy Python pipeline, which used such a dump.

IDs use Ergast's `driverRef`/`constructorRef` strings so they match `JolpicaF1`.
"""
struct ErgastCSV <: DataSource
    dir::String
end

function fetch_results(src::ErgastCSV, seasons::AbstractVector{<:Integer})
    rd(name) = CSV.read(joinpath(src.dir, name), DataFrame; missingstring = "\\N")
    results = rd("results.csv")
    races = select(rd("races.csv"), :raceId, :year, :round, :name)
    drivers = select(rd("drivers.csv"), :driverId, :driverRef, :forename, :surname)
    constructors = select(rd("constructors.csv"), :constructorId, :constructorRef)
    status = rd("status.csv")

    df = innerjoin(results, races; on = :raceId)
    filter!(:year => in(Set(seasons)), df)
    df = innerjoin(df, drivers; on = :driverId)
    df = innerjoin(df, constructors; on = :constructorId)
    df = innerjoin(df, status; on = :statusId)
    sort!(df, [:year, :round, :positionOrder])

    out = DataFrame(
        series = "f1",
        season = Int.(df.year),
        round = Int.(df.round),
        event_id = string.(df.year, "-", lpad.(df.round, 2, '0')),
        event_name = String.(df.name),
        stage_id = "race",
        competitor_id = String.(df.driverRef),
        competitor_name = string.(df.forename, " ", df.surname),
        codriver_id = Vector{Union{Missing,String}}(missing, nrow(df)),
        machine_id = String.(df.constructorRef),
        class = Vector{Union{Missing,String}}(missing, nrow(df)),
        time_ms = Vector{Union{Missing,Float64}}(maybefloat.(df.milliseconds)),
        position = Vector{Union{Missing,Int}}(maybeint.(df.position)),
        laps = Vector{Union{Missing,Int}}(maybeint.(df.laps)),
        status = String.(df.status),
        classified = .!ismissing.(df.position),
    )
    return validate_results(out)
end
