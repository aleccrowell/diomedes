# Per-lap conditions for the lap model (#116): tyre compound, track temperature and rainfall,
# from the F1 timing data (2021 on; scripts/conditions/fetch_conditions.py). Kept out of
# `LapData`, which chain checkpoints embed, as `LapContext` is.

"Tyre compounds of `LapConditions.comp`, in order; SOFT is the reference of the compound offsets."
const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
const WET_COMPOUNDS = (4, 5)

"""
    LapConditions(d::LapData, cond, numbers; wetdev = false)
    LapConditions()

Per-lap conditions and the switches for the lap model's #116 terms. `cond` is the table
written by `scripts/conditions/fetch_conditions.py` (`read_lap_conditions`), keyed by event,
car number and lap; `numbers` maps `(event_id, car number)` to the competitor
(`jolpica_driver_numbers`). For each of `d`'s laps:

- `comp`: index into `COMPOUNDS`, 0 if unknown. A lap the source leaves without a compound
  takes the one of the car's other laps in the same stint (`d.rows.stint` counts up from
  1 after each stop), when that is unique;
- `wet`: 1 on intermediate or wet tyres, else 0;
- `tt`: track temperature minus the race's mean over its laps, per 10 °C (0 if unknown):
  the race intercept takes the race's level, and the circuit term (`circuits`) its air
  temperature;
- `rain`: 1 if the weather station showed rain during the lap, else 0. Damp tracks without
  rain falling read 0 (2021 Istanbul, run on intermediates, has no rain sample), so the
  compound is the better wet signal.

Every race of `d` must have conditions data (it starts in 2021: fit those seasons).
`check` holds the coverage and a check of the join: the share of matched laps whose lap
time agrees with the source's within 0.5 s (a lap-number offset would fail it).

With `cond` (switched on in `lap_effects` when `on`), the lap's mean pace gets the compound
offsets against SOFT, compound-specific stint slopes, a track-temperature slope and a rain
effect; with `short` also wet-tyre and rain terms on the shortfall. With `wetdev`, car-season
and driver deviations on wet-tyre laps (see `lap_effects`); with `wetpool` too, the car-seasons'
are pooled through a wet walk over each works' seasons (the chassis walk's states, `CarWalks`)
instead of being independent (#116 step 1). With `wetcar = false`, drivers' deviations only
(`wetdrv`: the car-seasons' were not identified, see `lap_effects`). With `wetmin = n > 0`, only
car-seasons with at least `n` wet-tyre laps get a deviation (`wetcars`), with a tighter prior on
their spread (`wetshr`, #116 option 2: with ~30 wet laps the likelihood is nearly flat towards a
faster wet limit, because losses absorb it, and those car-seasons drifted to ±9 %).
"""
struct LapConditions
    comp::Vector{Int}
    wet::Vector{Float64}
    tt::Vector{Float64}
    rain::Vector{Float64}
    on::Bool
    wetdev::Bool
    wetpool::Bool
    wetcar::Bool
    wetmin::Int                 # with `wetdev`: car-seasons need this many wet-tyre laps for a deviation (0: all, no `wetshr`)
    wetcars::Vector{Int}        # those car-seasons (indices into d.machines), with wetmin > 0
    wetglobal::Bool             # their deviations centred over all of them, not within each season (`wetallg`)
    wetraces::Vector{Int}       # races (indices into d.races) with a wet-lap offset each (`wetrace`), else empty
    racedeg::Bool               # each race's own stint-age slope around the shared tyre curve (`racedeg`, #124)
    # `tyreage` (#124): the tyre's true age, counting the laps a used set had before the stint; else empty (stint age)
    tage::Vector{Float64}       # per lap, in laps (d.stint plus the set's earlier laps)
    tage_cd::Vector{Float64}    # tage/10 minus its mean over the driver's laps (as d.stint_cd)
    tage_cm::Vector{Float64}    # tage/10 minus its mean over the car-season's laps (as d.stint_cm)
    check::NamedTuple
end
LapConditions(comp, wet, tt, rain, on::Bool, wetdev::Bool, wetpool::Bool, wetcar::Bool, wetmin::Int, wetcars::Vector{Int},
              wetglobal::Bool, wetraces::Vector{Int}, racedeg::Bool, check::NamedTuple) =
    LapConditions(comp, wet, tt, rain, on, wetdev, wetpool, wetcar, wetmin, wetcars, wetglobal, wetraces, racedeg,
                  Float64[], Float64[], Float64[], check)
LapConditions(comp, wet, tt, rain, on::Bool, wetdev::Bool, wetpool::Bool, wetcar::Bool, wetmin::Int, wetcars::Vector{Int},
              wetglobal::Bool, wetraces::Vector{Int}, check::NamedTuple) =
    LapConditions(comp, wet, tt, rain, on, wetdev, wetpool, wetcar, wetmin, wetcars, wetglobal, wetraces, false, check)

const TYREAGE_MAX_EXTRA = 10   # more earlier laps than this: the sources disagree about the stint (`true_tyre_age`)

# A lap's tyre age (laps) and its driver- and car-centred versions /10: the true age with `tyreage`, else the stint age
@inline lap_age(c::LapConditions, d, j) = isempty(c.tage) ? d.stint[j] : c.tage[j]
@inline lap_age_cd(c::LapConditions, d, j) = isempty(c.tage) ? d.stint_cd[j] : c.tage_cd[j]
@inline lap_age_cm(c::LapConditions, d, j) = isempty(c.tage) ? d.stint_cm[j] : c.tage_cm[j]

"""
    true_tyre_age(stint, src_age, segs) -> (tage, extra, base)

The tyre's true age per lap (#124): the stint age (`stint`, laps since the out-lap) plus the laps
the set had before the stint. A stint's offset is the median over its laps of the source's tyre
age (`src_age`, NaN if unknown) minus the stint age; `base`, the offset of a new set (the mode
over the stints: the source also counts the out-lap), is subtracted, and negative remainders
are 0. `segs` gives each lap's stint (laps of one car in one race with one stint label). Stints
without a source age keep their stint age, as do stints with more than `maxextra` earlier laps:
a set run in qualifying or practice has a few (2021–2026: mostly 2–8), and after the pit stops
are reconciled with the timing data (`reconcile_pit_stops`, #151) more than 10 means the two
still disagree about the stint (2025-06: Jolpica's stops on laps 26–27, the timing data's tyre
change on lap 33), so its age is treated as unknown. `extra` holds each stint's earlier laps;
`dropped` counts the stints set to unknown that way.
"""
function true_tyre_age(stint::AbstractVector, src_age::AbstractVector, segs::AbstractVector; maxextra::Real = TYREAGE_MAX_EXTRA)
    offs = Dict{Any,Float64}()
    for s in unique(segs)
        js = findall(==(s), segs)
        v = [src_age[j] - stint[j] for j in js if !isnan(src_age[j])]
        isempty(v) || (offs[s] = round(median(v)))
    end
    isempty(offs) && return (Float64.(stint), Dict{Any,Float64}(), 0.0, 0)
    cnt = Dict{Float64,Int}()
    for v in values(offs); cnt[v] = get(cnt, v, 0) + 1; end
    base = argmax(cnt)
    extra = Dict(s => max(o - base, 0.0) for (s, o) in offs)
    dropped = count(>(maxextra), values(extra))
    filter!(p -> last(p) <= maxextra, extra)
    return ([stint[j] + get(extra, segs[j], 0.0) for j in eachindex(stint)], extra, base, dropped)
end
LapConditions(comp, wet, tt, rain, on::Bool, wetdev::Bool, check::NamedTuple) =
    LapConditions(comp, wet, tt, rain, on, wetdev, false, true, check)
LapConditions(comp, wet, tt, rain, on::Bool, wetdev::Bool, wetpool::Bool, check::NamedTuple) =
    LapConditions(comp, wet, tt, rain, on, wetdev, wetpool, true, check)
LapConditions(comp, wet, tt, rain, on::Bool, wetdev::Bool, wetpool::Bool, wetcar::Bool, check::NamedTuple) =
    LapConditions(comp, wet, tt, rain, on, wetdev, wetpool, wetcar, 0, Int[], false, check)
LapConditions(comp, wet, tt, rain, on::Bool, wetdev::Bool, wetpool::Bool, wetcar::Bool, wetmin::Int, wetcars::Vector{Int},
              check::NamedTuple) = LapConditions(comp, wet, tt, rain, on, wetdev, wetpool, wetcar, wetmin, wetcars, false, check)
LapConditions(comp, wet, tt, rain, on::Bool, wetdev::Bool, wetpool::Bool, wetcar::Bool, wetmin::Int, wetcars::Vector{Int},
              wetglobal::Bool, check::NamedTuple) =
    LapConditions(comp, wet, tt, rain, on, wetdev, wetpool, wetcar, wetmin, wetcars, wetglobal, Int[], check)
LapConditions() = LapConditions(Int[], Float64[], Float64[], Float64[], false, false, false, true, (;))

# `wetshr` (#116 option 2): the minimum wet-tyre laps for a car-season's deviation, and the prior sd of their spread
const WETSHR_MIN = 100
const WETSHR_SD = 0.5

"""
    wet_cars(mach, wet, nm, wetmin) -> Vector{Int}

The car-seasons (of `nm`, by the laps' `mach`) with at least `wetmin` laps on wet tyres (`wet`).
"""
wet_cars(mach::AbstractVector{Int}, wet::AbstractVector, nm::Int, wetmin::Int) =
    [m for m in 1:nm if count(j -> mach[j] == m && wet[j] > 0, eachindex(mach)) >= wetmin]

"""
    read_lap_conditions(path = data/conditions/lap_conditions.csv) -> DataFrame

The per-lap conditions table of `scripts/conditions/fetch_conditions.py`.
"""
function read_lap_conditions(path::AbstractString = joinpath(dirname(@__DIR__), "data", "conditions", "lap_conditions.csv"))
    isfile(path) || throw(ArgumentError("no $path: run scripts/conditions/fetch_conditions.py first"))
    return CSV.read(path, DataFrame; types = Dict(:event_id => String, :compound => String),
                    missingstring = "")
end

"""
    wet_green_laps(cond; share = 0.5) -> Set{Tuple{String,Int}}

Race laps `(event_id, lap)` of the conditions table run in the wet under green flag: at least
`share` of the cars with a known compound on intermediates or wets, and neither the lap nor the one
before it neutralised (safety car, VSC or red flag; `neutral`, from the timing data's track status
or race-control messages). `prepare_laps(…; keep_laps)` exempts them from the slow-lap cut, which
otherwise drops them with the neutralisations (2021–2026: Monaco 2023 laps 54–63, Zandvoort 2023
laps 3, 61 and 63, Suzuka 2022 laps 4–5, #116).
"""
function wet_green_laps(cond::AbstractDataFrame; share::Real = 0.5)
    hasproperty(cond, :neutral) || throw(ArgumentError("no `neutral` column: re-run scripts/conditions/fetch_conditions.py"))
    ok = filter(r -> !ismissing(r.compound) && r.compound in COMPOUNDS, cond)
    wet = Dict{Tuple{String,Int},Vector{Bool}}()
    for r in eachrow(ok)
        push!(get!(wet, (r.event_id, r.lap), Bool[]), r.compound in ("INTERMEDIATE", "WET"))
    end
    neutral = Set((r.event_id, r.lap) for r in eachrow(cond) if coalesce(r.neutral, 0) == 1)
    return Set(k for (k, v) in wet if mean(v) >= share && k ∉ neutral && (k[1], k[2] - 1) ∉ neutral)
end

"""
    reconcile_pit_stops(pits, cond, numbers) -> (; stops, noage, report)

Jolpica's recorded pit stops (`pits`, `fetch_pit_stops`) reconciled with the timing data's tyre
stints (`cond`, `read_lap_conditions`; `numbers`, `jolpica_driver_numbers`), #151. For each car
in a race with timing data:

1. a tyre change in the timing data (its tyre age falls, or the compound changes, from lap L-1
   to L) is a stop on lap L-1 if Jolpica has none on laps L-2 … L: tyre changes in the pit lane
   at a start procedure or red flag, and stops Jolpica lacks;
2. stops on consecutive laps (double records) are checked together, as one stop from the first
   lap s to the last s′;
3. a Jolpica stop the timing data covers (laps s-1 … s′+2) without a tyre change on laps s … s′+2
   is kept outside a neutralisation (the timing data misses some same-compound swaps), and the
   timing data's tyre age is then unknown from it to the next stop (`noage`);
4. … and dropped under a neutralisation (any of laps s … s′+1): a red-flag entry to the pit lane
   without a tyre change.

Races without timing data keep Jolpica's stops. `stops` has `pits`' columns (added stops have a
missing duration); `noage` holds the `(event_id, competitor_id, lap)` whose timing tyre age is
not to be used; `report` counts each case.
"""
function reconcile_pit_stops(pits::AbstractDataFrame, cond::AbstractDataFrame, numbers::AbstractDict)
    have = Set(cond.event_id)
    src = Dict{Tuple{String,String},Vector{Tuple{Int,Float64,String}}}()
    for r in eachrow(cond)
        c = get(numbers, (r.event_id, r.driver_number), nothing)
        (c === nothing || ismissing(r.tyre_age)) && continue
        push!(get!(src, (r.event_id, c), Tuple{Int,Float64,String}[]), (r.lap, Float64(r.tyre_age), coalesce(r.compound, "")))
    end
    neutral = hasproperty(cond, :neutral) ? Set((r.event_id, r.lap) for r in eachrow(cond) if coalesce(r.neutral, 0) == 1) :
              Set{Tuple{String,Int}}()
    jol = Dict{Tuple{String,String},Vector{Int}}()
    for r in eachrow(pits); push!(get!(jol, (r.event_id, r.competitor_id), Int[]), r.lap); end
    out = similar(pits, 0)
    rowof = Dict((r.event_id, r.competitor_id, r.lap) => i for (i, r) in enumerate(eachrow(pits)))
    noage = Set{Tuple{String,String,Int}}()
    rep = Dict(:jolpica => nrow(pits), :added => 0, :kept_unconfirmed => 0, :dropped_neutral => 0, :unchecked => 0)
    keys_all = union(Set(keys(jol)), Set(k for k in keys(src) if k[1] in have))
    for k in keys_all
        js = sort(get(jol, k, Int[]))
        if !(k[1] in have) || !haskey(src, k)            # no timing data: Jolpica's stops as they are
            foreach(s -> push!(out, pits[rowof[(k..., s)], :]), js)
            rep[:unchecked] += length(js); continue
        end
        v = sort(src[k])
        lapset = Set(first.(v))
        ch = [v[i][1] for i in 2:length(v) if v[i][1] == v[i - 1][1] + 1 && (v[i][2] < v[i - 1][2] || v[i][3] != v[i - 1][3])]
        keep_s, unconfirmed = Int[], Int[]
        # stops on consecutive laps (double records) are checked together, as one stop from the first to the last
        groups = Vector{Int}[]
        for s in js
            !isempty(groups) && s == last(last(groups)) + 1 ? push!(last(groups), s) : push!(groups, [s])
        end
        for g in groups
            s1, s2 = first(g), last(g)
            covered = all(l -> l in lapset, (s1 - 1):(s2 + 2))
            if !covered || any(L -> s1 <= L <= s2 + 2, ch)
                append!(keep_s, g)                       # unchecked, or confirmed by a tyre change
            elseif any(l -> (k[1], l) in neutral, s1:(s2 + 1))
                rep[:dropped_neutral] += length(g)       # rule 4
            else
                append!(keep_s, g); push!(unconfirmed, s2)   # rule 3
                rep[:kept_unconfirmed] += length(g)
            end
        end
        added = [L - 1 for L in ch if !any(s -> L - 2 <= s <= L, js)]   # rule 1
        rep[:added] += length(added)
        all_s = sort(unique(vcat(keep_s, added)))
        for s in all_s
            i = get(rowof, (k..., s), nothing)
            if i === nothing
                row = Dict{Symbol,Any}(n => missing for n in propertynames(pits))
                ser = first(pits.series); row[:series] = ser; row[:season] = parse(Int, k[1][1:4])
                row[:event_id] = k[1]; row[:competitor_id] = k[2]; row[:lap] = s
                push!(out, row)
            else
                push!(out, pits[i, :])
            end
        end
        for s in unconfirmed                             # the timing age is unknown until the next stop
            nxt = something(findfirst(>(s), all_s), 0)
            hi = nxt == 0 ? maximum(lapset) : all_s[nxt]
            foreach(l -> push!(noage, (k[1], k[2], l)), (s + 1):hi)
        end
    end
    rep[:stops] = nrow(out)
    return (; stops = sort!(out, [:event_id, :competitor_id, :lap]), noage, report = rep)
end

function LapConditions(d::LapData, cond::AbstractDataFrame, numbers::AbstractDict; wetdev::Bool = false,
                       wetpool::Bool = false, wetcar::Bool = true, wetmin::Int = 0, wetglobal::Bool = false,
                       wetrace::Bool = false, racedeg::Bool = false, tyreage::Bool = false,
                       noage::AbstractSet = Set{Tuple{String,String,Int}}())
    wetpool && !wetdev && throw(ArgumentError("wetpool pools the wet deviations: it needs wetdev"))
    wetpool && !wetcar && throw(ArgumentError("wetpool pools the car-seasons' wet deviations: it needs wetcar"))
    wetmin > 0 && !(wetdev && wetcar && !wetpool) &&
        throw(ArgumentError("wetmin selects car-seasons for independent wet deviations: it needs wetdev, without wetpool or wetcar = false"))
    wetglobal && wetmin == 0 && throw(ArgumentError("wetglobal centres the wetmin car-seasons' deviations: it needs wetmin > 0"))
    wetrace && wetmin == 0 && throw(ArgumentError("wetrace adds race offsets to the wetmin deviations: it needs wetmin > 0"))
    ci = Dict(c => k for (k, c) in enumerate(COMPOUNDS))
    have = Set(cond.event_id)
    nod = filter(e -> e ∉ have, d.races)
    isempty(nod) || throw(ArgumentError("no lap conditions for $(length(nod)) races ($(join(first(nod, 5), ", "))…); " *
                                        "the data starts in 2021"))
    row = Dict{Tuple{String,String,Int},Int}()
    unknown_cars = Set{Tuple{String,Int}}()            # cars of d's races with no competitor for their number
    races = Set(d.races)
    for (i, r) in enumerate(eachrow(cond))
        r.event_id in races || continue
        c = get(numbers, (r.event_id, r.driver_number), nothing)
        c === nothing && (push!(unknown_cars, (r.event_id, r.driver_number)); continue)
        row[(r.event_id, c, r.lap)] = i
    end
    n = length(d.y)
    comp, ttraw, rain, srcage = zeros(Int, n), fill(NaN, n), zeros(n), fill(NaN, n)
    matched, agree, timed = 0, 0, 0
    for (j, r) in enumerate(eachrow(d.rows))
        i = get(row, (r.event_id, r.competitor_id, r.lap), 0)
        i == 0 && continue
        matched += 1
        x = cond[i, :]
        comp[j] = ismissing(x.compound) ? 0 : get(ci, x.compound, 0)
        ismissing(x.track_temp) || (ttraw[j] = x.track_temp)
        rain[j] = coalesce(x.rainfall, 0.0) > 0 ? 1.0 : 0.0
        tyreage && hasproperty(x, :tyre_age) && !ismissing(x.tyre_age) && (r.event_id, r.competitor_id, r.lap) ∉ noage &&
            (srcage[j] = Float64(x.tyre_age))
        if !ismissing(x.lap_time_ms)
            timed += 1
            abs(x.lap_time_ms - r.time_ms) <= 500 && (agree += 1)
        end
    end
    # unknown compounds: the car's compound in the same stint, if its known laps agree
    filled = 0
    segid = zeros(Int, n)                                # each lap's stint, numbered over all cars and races
    nseg = 0
    for g in groupby(DataFrame(j = 1:n, e = d.rows.event_id, c = d.rows.competitor_id, lap = d.rows.lap,
                               stint = d.rows.stint), [:e, :c])
        js = g.j[sortperm(g.lap)]
        seg = cumsum([k == 1 || d.rows.stint[js[k]] <= d.rows.stint[js[k - 1]] ? 1 : 0 for k in eachindex(js)])
        for s in unique(seg)
            jj = js[seg .== s]
            nseg += 1; segid[jj] .= nseg
            known = unique(comp[jj][comp[jj] .> 0])
            length(known) == 1 || continue
            for j in jj
                comp[j] == 0 && (comp[j] = known[1]; filled += 1)
            end
        end
    end
    # track temperature: minus the race's mean over its laps, per 10 °C
    tt = zeros(n)
    for r in eachindex(d.races)
        js = findall(==(r), d.race)
        ok = filter(j -> !isnan(ttraw[j]), js)
        isempty(ok) && continue
        m = mean(ttraw[ok])
        tt[ok] .= (ttraw[ok] .- m) ./ 10
    end
    wet = [Float64(k in WET_COMPOUNDS) for k in comp]
    check = (; laps = n, matched = matched / n, time_agrees = timed > 0 ? agree / timed : NaN,
             compound_known = count(>(0), comp) / n, compound_filled = filled, wet = sum(wet) / n, rain = sum(rain) / n,
             track_temp_known = count(!isnan, ttraw) / n, unknown_cars = length(unknown_cars),
             compounds = Dict((k == 0 ? "unknown" : COMPOUNDS[k]) => count(==(k), comp) for k in 0:length(COMPOUNDS)))
    wcs = wetmin > 0 ? wet_cars(d.mach, wet, length(d.machines), wetmin) : Int[]
    wetmin > 0 && isempty(wcs) && throw(ArgumentError("no car-season has $wetmin wet-tyre laps"))
    wrs = wetrace ? sort(unique(d.race[wet .> 0])) : Int[]
    # `tyreage` (#124): the true tyre age, and its driver- and car-centred versions as in `prepare_laps`
    tage, tcd, tcm = Float64[], Float64[], Float64[]
    if tyreage
        tage, extra, base, dropped = true_tyre_age(d.stint, srcage, segid)
        s10 = tage ./ 10
        nd, nm = length(d.competitors), length(d.machines)
        sd_, nd_, sm_, nm_ = zeros(nd), zeros(nd), zeros(nm), zeros(nm)
        for (k, m, s) in zip(d.comp, d.mach, s10); sd_[k] += s; nd_[k] += 1; sm_[m] += s; nm_[m] += 1; end
        tcd, tcm = s10 .- (sd_ ./ max.(nd_, 1))[d.comp], s10 .- (sm_ ./ max.(nm_, 1))[d.mach]
        ex = collect(values(extra))
        check = merge(check, (; tyre_age_base = base, stints_aged = length(ex), stints_used = count(>(0), ex), stints_dropped = dropped,
                               used_median_laps = isempty(filter(>(0), ex)) ? 0.0 : median(filter(>(0), ex)),
                               laps_aged = count(j -> tage[j] > d.stint[j], 1:n) / n))
    end
    return LapConditions(comp, wet, tt, rain, true, wetdev, wetpool, wetcar, wetmin, wcs, wetglobal, wrs, racedeg,
                         tage, tcd, tcm, check)
end

function Base.show(io::IO, c::LapConditions)
    c.on || return print(io, "LapConditions(off)")
    k = c.check
    print(io, "LapConditions($(k.laps) laps: matched $(round(k.matched; digits = 3)), lap time agrees ",
          "$(round(k.time_agrees; digits = 3)), compound known $(round(k.compound_known; digits = 3)) ",
          "($(k.compound_filled) filled from the stint), wet tyres $(round(k.wet; digits = 3)), rain ",
          "$(round(k.rain; digits = 3)), track temp known $(round(k.track_temp_known; digits = 3))",
          c.wetdev ? (!c.wetcar ? "; drivers' wet deviations" : c.wetpool ? "; wet deviations, cars pooled by works" :
                      c.wetmin > 0 ? "; wet deviations, $(length(c.wetcars)) car-seasons with ≥ $(c.wetmin) wet laps" *
                                     (c.wetglobal ? ", centred over all" : ", centred within season") *
                                     (isempty(c.wetraces) ? "" : "; wet-lap offsets for $(length(c.wetraces)) races") :
                      "; wet deviations") : "", c.racedeg ? "; a stint slope per race" : "",
          isempty(c.tage) ? "" : "; true tyre age: $(k.stints_used) of $(k.stints_aged) stints on used sets " *
                                 "(median $(k.used_median_laps) earlier laps; new-set offset $(k.tyre_age_base); " *
                                 "$(k.stints_dropped) with > $(TYREAGE_MAX_EXTRA) treated as unknown), " *
                                 "$(round(100k.laps_aged; digits = 1))% of laps aged", ")")
end

# Shared condition effects (`cv` in `lap_loglik`): names and prior sds, laid out as
# compound offsets against SOFT (%), compound stint slopes (% per 10 laps, on top of the
# shared stint curve), track temperature (% per 10 °C), rain (%), and with `short`
# wet tyres and rain on the shortfall's logit π and log λ.
const COND_OFFSET = 1:4        # MEDIUM, HARD, INTERMEDIATE, WET against SOFT
const COND_SLOPE = 5:8
const COND_TT, COND_RAIN = 9, 10
const COND_KAPPA = 11:14       # wet π, wet λ, rain π, rain λ
cond_names(short::Bool) = vcat(COMPOUNDS[2:end], "stint·" .* COMPOUNDS[2:end], "track_temp", "rain",
                               short ? ["κ_wet_π", "κ_wet_λ", "κ_rain_π", "κ_rain_λ"] : String[])
cond_sd(short::Bool) = vcat([2.0, 2.0, 5.0, 5.0], fill(1.0, 4), 1.0, 2.0, short ? fill(1.0, 4) : Float64[])

# Group of each wet deviation (`wv`): car-seasons by season (centred within it), then the drivers (one group)
wet_groups(d::LapData) = vcat(d.mach_season, fill(length(d.seasons) + 1, length(d.competitors)))
wet_counts(d::LapData) = vcat([count(==(k), d.mach_season) for k in eachindex(d.seasons)], length(d.competitors))
wet_blocks(d::LapData) = vcat(fill(1, length(d.machines)), fill(2, length(d.competitors)))

# Wet tyres and rain on lap j's shortfall: the shifts of logit q and log λ_lap (#116, with `short`)
@inline cond_short(cv, cond::LapConditions, j) =
    (cv[COND_KAPPA[1]] * cond.wet[j] + cv[COND_KAPPA[3]] * cond.rain[j],
     cv[COND_KAPPA[2]] * cond.wet[j] + cv[COND_KAPPA[4]] * cond.rain[j])

# The two blocks of the wet deviations as one vector (car-seasons, then drivers): `vcat` of
# tracked arrays is ambiguous for ReverseDiff, so it has its own rule (#116, `wetpool`)
wet_join(a, b) = vcat(a, b)
function ChainRulesCore.rrule(::typeof(wet_join), a, b)
    na = length(a)
    pullback(ȳ) = (y = collect(ChainRulesCore.unthunk(ȳ)); (NoTangent(), y[1:na], y[(na + 1):end]))
    return wet_join(a, b), pullback
end
ReverseDiff.@grad_from_chainrules wet_join(a::ReverseDiff.TrackedArray, b::ReverseDiff.TrackedArray)

# `x` placed at `idx` of a zero vector of length n (the `wetshr` car-seasons' deviations), with its own rule for ReverseDiff
function wet_scatter(x, idx::Vector{Int}, n::Int)
    y = zeros(eltype(x), n)
    y[idx] .= x
    return y
end
function ChainRulesCore.rrule(::typeof(wet_scatter), x, idx::Vector{Int}, n::Int)
    pullback(ȳ) = (NoTangent(), collect(ChainRulesCore.unthunk(ȳ))[idx], NoTangent(), NoTangent())
    return wet_scatter(x, idx, n), pullback
end
ReverseDiff.@grad_from_chainrules wet_scatter(x::ReverseDiff.TrackedArray, idx::Vector{Int}, n::Int)

# Season group and group sizes of the `wetshr` car-seasons, for centring them within season
wet_car_groups(d::LapData, wcs::Vector{Int}) = (g = d.mach_season[wcs]; (g, [count(==(k), g) for k in eachindex(d.seasons)]))
# … or one group of all of them with `wetglobal` (`wetallg`; with `wetall`, centring within a season of two car-seasons with 1 and 4 wet
# laps pinned them to ±0.44 %, and in 2023 four few-lap cars drifting fast forced Williams to +2.6 %, probe 116f, #116)
wet_car_groups(d::LapData, c::LapConditions) =
    c.wetglobal ? (fill(1, length(c.wetcars)), [length(c.wetcars)]) : wet_car_groups(d, c.wetcars)

# Pooled car-season wet deviations (#116 step 1): a walk per works over its seasons on the
# chassis walk's states, each works' first season N(0, σ0), a season's step N(0, τ·√gap),
# centred within season (the shared wet offset takes a season's mean); each car-season takes
# its works' state.
function wet_car_walk(σ0, τ, z, cw)
    x = walk_path(walk_steps(τ, σ0, 0.0, 0.0, z, cw.chassis), cw.chassis.prev)
    return center_by(x, cw.chassis.season, cw.chassis.counts)[cw.row_ch]
end
