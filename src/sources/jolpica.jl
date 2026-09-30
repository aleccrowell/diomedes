"""
    JolpicaF1(; base_url, cache_dir, refresh=false)

F1 race results from the Jolpica API (https://github.com/jolpica/jolpica-f1),
the maintained successor to the Ergast API, with the same schema. Covers 1950
to the present.

Unauthenticated limits are 4 req/s burst and 500 req/hour sustained. A full
history is roughly 300 paged requests; responses are cached, so this is a
one-off cost. Pass `refresh=true` to re-fetch (e.g. the in-progress season).
`suspended_ms` (red-flag suspension included in official times, see
`suspension_ms`): races whose winner's average lap is anomalously slow against
the fastest lap (`suspect_suspension`) have their lap times fetched and checked,
a few extra requests per season; other races get 0. Races without fastest-lap
data (before 2004) get `missing`.
The Indianapolis 500 (1950–1960) is excluded unless `include_indy500 = true`
(see `INDY500`).
"""
Base.@kwdef struct JolpicaF1 <: DataSource
    base_url::String = "https://api.jolpi.ca/ergast/f1"
    cache_dir::String = joinpath(default_cache_dir(), "jolpica")
    refresh::Bool = false
    include_indy500::Bool = false
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
    fastest = Dict{String,Float64}()          # event_id => fastest lap (ms) in the race
    offset, total = 0, 1
    while offset < total
        url = "$(src.base_url)/$season/results.json?limit=$JOLPICA_PAGE&offset=$offset"
        mr = cached_json(url; cache_dir = src.cache_dir, limiter = src.limiter,
                         refresh = src.refresh).MRData
        total = parse(Int, mr.total)
        # A race's results can be split across pages; each page repeats the race header.
        for race in mr.RaceTable.Races, r in race.Results
            (src.include_indy500 || race.raceName != INDY500) || continue
            row = jolpica_row(race, r)
            push!(rows, row)
            fl = get(r, :FastestLap, nothing)
            if fl !== nothing && haskey(fl, :Time)
                ms = laptime_ms(String(fl.Time.time))
                fastest[row.event_id] = min(get(fastest, row.event_id, Inf), ms)
            end
        end
        offset += JOLPICA_PAGE
    end
    for g in groupby(rows, :event_id)
        eid = first(g.event_id)
        timed = findall(!ismissing, g.time_ms)
        if !haskey(fastest, eid) || isempty(timed)
            g.suspended_ms .= missing
            continue
        end
        w = timed[argmin(g.time_ms[timed])]
        g.suspended_ms .= suspect_suspension(g.time_ms[w], g.laps[w], fastest[eid]) ?
            suspension_ms(jolpica_laps(src, season, first(g.round))) : 0.0
    end
    return rows
end

# Lap times of one race, per car (vectors ordered by lap).
function jolpica_laps(src::JolpicaF1, season, round)
    laps = Dict{String,Vector{Tuple{Int,Float64}}}()
    offset, total = 0, 1
    while offset < total
        url = "$(src.base_url)/$season/$round/laps.json?limit=$JOLPICA_PAGE&offset=$offset"
        mr = cached_json(url; cache_dir = src.cache_dir, limiter = src.limiter,
                         refresh = src.refresh).MRData
        total = parse(Int, mr.total)
        for race in mr.RaceTable.Races, lap in race.Laps, t in lap.Timings
            push!(get!(laps, String(t.driverId), Tuple{Int,Float64}[]),
                  (parse(Int, lap.number), laptime_ms(String(t.time))))
        end
        offset += JOLPICA_PAGE
    end
    return [last.(sort(v)) for v in values(laps)]
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
        event_date = maybedate(get(race, :date, nothing)),
        stage_id = "race",
        competitor_id = String(r.Driver.driverId),
        competitor_name = string(r.Driver.givenName, " ", r.Driver.familyName),
        competitor_birth = maybedate(get(r.Driver, :dateOfBirth, nothing)),
        codriver_id = missing,
        machine_id = String(r.Constructor.constructorId),
        class = missing,
        manufacturer = missing,
        entrant = missing,
        time_ms = time === nothing ? missing : maybefloat(get(time, :millis, nothing)),
        suspended_ms = missing,                # filled per race in jolpica_season
        position = classified ? maybeint(r.position) : missing,
        laps = maybeint(get(r, :laps, nothing)),
        status = String(r.status),
        classified,
    )
end
