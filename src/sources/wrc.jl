"""
    WRCTiming(; championship="World Rally Championship", series="wrc", ...)

Stage times from the timing API behind wrc.com (hosted by Red Bull). It is
undocumented; the endpoints follow OpenWRC (github.com/jixy2012/OpenWRC) and
rallydatajunkie.com's notes on the WRC live timing API.

Coverage: WRC 2018 onwards and the European Rally Championship 2022 onwards
(`WRCTiming(championship = "European Rally Championship", series = "erc")`).

Each completed special stage is one `stage_id`. `machine_id` is the vehicle model
(e.g. "GR Yaris Rally1") and `class` is the entry group (Rally1, Rally2, ...), so
factory and customer cars of the same model share a machine effect. Stage times
exclude penalties. Only stages with status "Completed" are included; cancelled
and interrupted stages (which get notional times) are skipped.

About 300 requests per season; responses are cached.
"""
Base.@kwdef struct WRCTiming <: DataSource
    championship::String = "World Rally Championship"
    series::String = "wrc"
    base_url::String = "https://p-p.redbull.com/rb-wrccom-lintegration-yv-prod/api"
    cache_dir::String = joinpath(default_cache_dir(), "wrc")
    refresh::Bool = false
    limiter::RateLimiter = RateLimiter(0.5)
end

wrc_json(src::WRCTiming, path; refresh = src.refresh) =
    cached_json(src.base_url * path; cache_dir = src.cache_dir, limiter = src.limiter,
                refresh)

function fetch_results(src::WRCTiming, seasons::AbstractVector{<:Integer})
    # The season list and calendar change, so always re-fetch them; events that
    # have not finished yet are re-fetched too (see `wrc_event`).
    all_seasons = wrc_json(src, "/seasons.json"; refresh = true)
    out = empty_results()
    for year in seasons
        idx = findfirst(s -> s.name == src.championship && s.year == year, all_seasons)
        idx === nothing && (@warn "no $(src.championship) season $year in WRC timing API"; continue)
        detail = wrc_json(src, "/season-detail.json?seasonId=$(all_seasons[idx].seasonId)";
                          refresh = true)
        for rnd in sort(collect(detail.seasonRounds); by = r -> r.order)
            finish = Date(rnd.event.finishDate[1:10])
            finish > today() && continue  # not started or still running: no final times
            live = finish >= today() - Day(2)  # recently finished: results may be amended
            append!(out, wrc_event(src, year, rnd.order, rnd.eventId; refresh = src.refresh || live))
        end
    end
    return validate_results(out)
end

function wrc_event(src::WRCTiming, season, round, event_id; refresh = src.refresh)
    api(path) = wrc_json(src, path; refresh)
    ev = api("/events/$event_id.json")
    rally = ev.rallies[something(findfirst(r -> r.isMain, ev.rallies), 1)]
    itin = api("/events/$event_id/itineraries/$(rally.itineraryId).json")
    entries = api("/events/$event_id/rallies/$(rally.rallyId)/entries.json")
    by_entry = Dict(e.entryId => e for e in entries)

    rows = empty_results()
    for leg in itin.itineraryLegs, section in leg.itinerarySections, stage in section.stages
        stage.status == "Completed" || continue
        times = api("/events/$event_id/stages/$(stage.stageId)/stagetimes.json" *
                           "?rallyId=$(rally.rallyId)")
        for t in times
            e = get(by_entry, t.entryId, nothing)
            e === nothing && continue
            push!(rows, (
                series = src.series,
                season,
                round,
                event_id = string(event_id),
                event_name = String(ev.name),
                event_date = maybedate(get(ev, :startDate, nothing)),
                stage_id = String(stage.code),
                competitor_id = string(e.driverId),
                competitor_name = String(e.driver.fullName),
                competitor_birth = missing,
                codriver_id = maybestring(get(e, :codriverId, nothing)),
                machine_id = something(maybestring(get(e, :vehicleModel, nothing)), "unknown"),
                class = e.group === nothing ? missing : String(e.group.name),
                time_ms = t.status == "Completed" ? maybefloat(t.elapsedDurationMs) : missing,
                suspended_ms = missing,
                position = t.status == "Completed" ? maybeint(t.position) : missing,
                laps = missing,
                status = String(t.status),
                classified = t.status == "Completed",
            ))
        end
    end
    return rows
end
