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
        timed = most_laps(g, timed)               # the winner as in prepare_gaps (see gap_model.jl)
        w = timed[argmin(g.time_ms[timed])]
        # From 2023 "Lapped" also covers classified cars that stopped before the flag
        # (Norris, Austria 2024: 64 of 71 laps). A lapped car crosses the line after
        # the winner, so its own time is at least the winner's; one below it stopped
        # early and is a retirement (cause not given), not a lapped finisher (#55).
        for i in eachindex(g.status)
            if is_lapped_status(g.status[i]) && !ismissing(g.own_time_ms[i]) && g.own_time_ms[i] < g.time_ms[w]
                g.status[i] = "Retired"
            end
        end
        g.suspended_ms .= if suspect_suspension(g.time_ms[w], g.laps[w], fastest[eid])
            car_laps = jolpica_laps(src, season, first(g.round))
            s = suspension_ms(values(car_laps))
            # Jolpica's lap times drop whole hours ("22:30.5" for a 1:22:30.5 lap with
            # a red flag in it: Bahrain 2020, Japan 2022). The winner's official time
            # minus its laps' sum recovers them; complete laps otherwise sum to it exactly.
            wl = get(car_laps, g.competitor_id[w], nothing)
            s + (wl === nothing || ismissing(g.laps[w]) || length(wl) != g.laps[w] ? 0.0 : lap_hours_lost(g.time_ms[w], wl))
        else
            0.0
        end
    end
    return rows
end

"""
    lap_hours_lost(official_ms, laps_ms) -> Float64

Whole hours missing from a car's lap times (ms), from its official race time:
Jolpica's lap time strings drop the hours of a lap with a red flag in it
("22:30.5" for 1:22:30.5). Complete lap times otherwise sum to the official time.
"""
lap_hours_lost(official_ms, laps_ms) = 3_600_000 * max(round((official_ms - sum(laps_ms)) / 3_600_000), 0)

# Lap times of one race, per car (driver id => vector ordered by lap).
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
    return Dict(k => last.(sort(v)) for (k, v) in laps)
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
        # Ergast convention: a race time only for cars that finished on the lead lap.
        # From 2023 Jolpica also times every other car over the laps it completed:
        # lapped cars at the flag, retired and disqualified cars when they stopped.
        # That is the car's own elapsed time, not a race time (read as one, a
        # retirement beat the winner's time and became the race's reference, #55), so
        # it goes to own_time_ms. prepare_gaps uses it for lapped cars' exact gaps; for
        # retirements it equals the cumulative lap time at the last lap (checked on
        # 2023-25), which partial_gaps already uses.
        time_ms = time === nothing || String(r.status) != "Finished" ? missing :
                  maybefloat(get(time, :millis, nothing)),
        own_time_ms = time === nothing || String(r.status) == "Finished" ? missing :
                      maybefloat(get(time, :millis, nothing)),
        suspended_ms = missing,                # filled per race in jolpica_season
        position = classified ? maybeint(r.position) : missing,
        grid = maybeint(get(r, :grid, nothing)),
        laps = maybeint(get(r, :laps, nothing)),
        status = String(r.status),
        classified,
    )
end

# Rounds of a season, from its schedule (one request), skipping the Indianapolis 500.
function jolpica_rounds(src::JolpicaF1, season::Integer)
    url = "$(src.base_url)/$season.json?limit=$JOLPICA_PAGE"
    mr = cached_json(url; cache_dir = src.cache_dir, limiter = src.limiter, refresh = src.refresh).MRData
    return [parse(Int, r.round) for r in mr.RaceTable.Races if src.include_indy500 || r.raceName != INDY500]
end

# Pages of one race's endpoint (laps or pitstops), concatenated.
function jolpica_race_pages(f, src::JolpicaF1, season, round, endpoint)
    offset, total = 0, 1
    while offset < total
        url = "$(src.base_url)/$season/$round/$endpoint.json?limit=$JOLPICA_PAGE&offset=$offset"
        mr = cached_json(url; cache_dir = src.cache_dir, limiter = src.limiter, refresh = src.refresh).MRData
        total = parse(Int, mr.total)
        foreach(f, mr.RaceTable.Races)
        offset += JOLPICA_PAGE
    end
end

"""
    fetch_laps(src::JolpicaF1, seasons) -> DataFrame

Lap times (`LAP_SCHEMA`) from Jolpica's laps endpoint, which has them from 1996:
about 11 paged requests per race (100 timings per page), cached. Ids match
`fetch_results(src, …)` and `ErgastCSV`, so seasons from the two sources can be
combined. Races without lap data are skipped.
"""
function fetch_laps(src::JolpicaF1, seasons::AbstractVector{<:Integer})
    out = DataFrame([name => T[] for (name, T) in LAP_SCHEMA])
    for season in seasons, round in jolpica_rounds(src, season)
        eid = string(season, "-", lpad(round, 2, '0'))
        jolpica_race_pages(src, season, round, "laps") do race
            for lap in race.Laps, t in lap.Timings
                push!(out, ("f1", season, eid, String(t.driverId), parse(Int, lap.number),
                            laptime_ms(String(t.time)), maybeint(get(t, :position, nothing))))
            end
        end
    end
    sort!(out, [:season, :event_id, :competitor_id, :lap])
    return validate_schema(out, LAP_SCHEMA, "laps")
end

"""
    fetch_pit_stops(src::JolpicaF1, seasons) -> DataFrame

Pit stops (`PIT_SCHEMA`) from Jolpica's pitstops endpoint (2011 on), one or two
requests per race, cached. `duration_ms` is `missing` when the duration isn't a
plain time (e.g. a stop during a red flag).
"""
function fetch_pit_stops(src::JolpicaF1, seasons::AbstractVector{<:Integer})
    out = DataFrame([name => T[] for (name, T) in PIT_SCHEMA])
    for season in seasons, round in jolpica_rounds(src, season)
        eid = string(season, "-", lpad(round, 2, '0'))
        jolpica_race_pages(src, season, round, "pitstops") do race
            for p in get(race, :PitStops, [])
                d = tryparse(Float64, String(get(p, :duration, "")))
                push!(out, ("f1", season, eid, String(p.driverId), parse(Int, p.lap),
                            d === nothing ? missing : 1000d))
            end
        end
    end
    sort!(out, [:season, :event_id, :competitor_id, :lap])
    return validate_schema(out, PIT_SCHEMA, "pit stops")
end
