"""
    JolpicaF1(; base_url, cache_dir, refresh=false)

F1 race results from the Jolpica API (https://github.com/jolpica/jolpica-f1),
the maintained successor to the Ergast API, with the same schema. Covers 1950
to the present.

Unauthenticated limits are 4 req/s burst and 500 req/hour sustained. A full
history is roughly 300 paged requests; responses are cached, so this is a
one-off cost. Pass `refresh=true` to re-fetch (e.g. the in-progress season).
"""
Base.@kwdef struct JolpicaF1 <: DataSource
    base_url::String = "https://api.jolpi.ca/ergast/f1"
    cache_dir::String = joinpath(default_cache_dir(), "jolpica")
    refresh::Bool = false
    limiter::RateLimiter = RateLimiter(0.3)
end

const JOLPICA_PAGE = 100  # API maximum

function fetch_results(src::JolpicaF1, seasons::AbstractVector{<:Integer})
    out = empty_results()
    for season in seasons
        append!(out, jolpica_season(src, season))
    end
    return validate_results(out)
end

function jolpica_season(src::JolpicaF1, season::Integer)
    rows = empty_results()
    offset, total = 0, 1
    while offset < total
        url = "$(src.base_url)/$season/results.json?limit=$JOLPICA_PAGE&offset=$offset"
        mr = cached_json(url; cache_dir = src.cache_dir, limiter = src.limiter,
                         refresh = src.refresh).MRData
        total = parse(Int, mr.total)
        # A race's results can be split across pages; each page repeats the race header.
        for race in mr.RaceTable.Races, r in race.Results
            push!(rows, jolpica_row(race, r))
        end
        offset += JOLPICA_PAGE
    end
    return rows
end

function jolpica_row(race, r)
    rnd = parse(Int, race.round)
    time = get(r, :Time, nothing)
    # positionText is the classified position, or a letter (R, D, W, ...) if not
    # classified. JSON `position` is always filled in, whereas the CSV dump leaves
    # it null for unclassified cars; follow the CSV so both sources agree.
    classified = tryparse(Int, r.positionText) !== nothing
    return (
        series = "f1",
        season = parse(Int, race.season),
        round = rnd,
        event_id = string(race.season, "-", lpad(rnd, 2, '0')),
        event_name = String(race.raceName),
        stage_id = "race",
        competitor_id = String(r.Driver.driverId),
        competitor_name = string(r.Driver.givenName, " ", r.Driver.familyName),
        codriver_id = missing,
        machine_id = String(r.Constructor.constructorId),
        class = missing,
        time_ms = time === nothing ? missing : maybefloat(get(time, :millis, nothing)),
        position = classified ? maybeint(r.position) : missing,
        laps = maybeint(get(r, :laps, nothing)),
        status = String(r.status),
        classified,
    )
end
