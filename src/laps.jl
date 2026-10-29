# Lap-time model (#12): per-lap pace instead of whole-race times.
#
# Outcome: y = 100·log(t / m_race), a lap time in % of the race's median clean
# lap, on the same % scale as the race-time models. Laps kept are "clean" green-
# flag racing laps. Excluded:
#   - lap 1 (standing start),
#   - race-wide slow laps (the field's median lap > SLOW_RACE × the race median:
#     safety car, VSC, red flag) and the lap after each (the restart),
#   - a car's laps over STOPPAGE × its own median lap (a red-flag stoppage is in
#     one lap per car, at different lap numbers; also very long stops), unless
#     `stoppage = false`,
#   - in- and out-laps of pit stops, recorded (Ergast pit stops, 2011 on) or
#     inferred from lap times (see `infer_pit_stops`).
# Other slow laps (traffic, mistakes, damage) stay in: the pace + loss noise
# treats them as incident losses, as in the race-time model. Inferred stops and
# the per-car STOPPAGE cut drop a car's laps by their size, so they also drop its
# largest incidents (#108: about 5 per 1,000 laps in 2011–2026, more for
# incident-prone drivers). With recorded stops and `stoppage = false`, no lap is
# dropped for its own size: only stops, race-wide neutralisations and lap 1 (its
# own model) are left out, and the big-loss component (`big`) takes the largest
# losses.
#
# Covariates: the lap's fraction of the race distance (fuel burn, track
# evolution; a slope per race), the tyres' stint age in laps since the last
# stop (`stint`, 1 on the first lap after the out-lap), the driver's age in the
# season minus their mean age over their laps (`agec`, decades; for career
# slopes, as in the race-time model), and the race's position in its season
# minus its car's mean position (`posc`; for in-season car development, #14),
# and whether the car led the race on that lap (`lead`, from its position at the
# end of the lap; #66).

# Seasons with in-race refuelling (1994–2009): the fuel load jumps back up at each
# stop, so within a stint fuel burn follows stint age as tyre wear does. The shared
# stint-age curve is separate for these seasons and the rest (#66).
const REFUEL_SEASONS = 1994:2009

const SLOW_RACE = 1.25      # race-wide slow lap: field median over this × race median lap
const STOPPAGE = 2.0        # a car's lap over this × its median lap is excluded
const PIT_EXCESS_MS = 12_000.0   # inferred stop: in-lap + out-lap this much slower than neighbouring laps

"""
    infer_pit_stops(times::Vector{Float64}, skip::AbstractSet{Int}) -> Vector{Int}

In-laps of a car's pit stops inferred from its lap times (`times[l]` = lap l):
lap l is a stop if its excess and the next lap's, over the median of the car's
laps within ±5 (leaving out lap 1 and `skip`, the race-wide slow laps), sum to
more than `PIT_EXCESS_MS`, the pit-lane loss. Overlapping candidates keep the
largest. Against Ergast's recorded stops 2011–2019 (within ±1 lap): recall
0.986, precision 0.85; most "false" stops have a stop-sized excess (median
23 s), e.g. drive-through penalties, which Ergast does not list as stops.
"""
function infer_pit_stops(times::AbstractVector{<:Real}, skip::AbstractSet{Int})
    n = length(times)
    ex = fill(NaN, n)
    for l in 2:n
        l in skip && continue
        nb = [times[k] for k in max(1, l - 5):min(n, l + 5) if k != l && k != 1 && !(k in skip)]
        length(nb) >= 3 && (ex[l] = times[l] - median(nb))
    end
    cand = [(ex[l] + (l < n && !isnan(ex[l + 1]) ? ex[l + 1] : 0.0), l) for l in 2:n
            if !isnan(ex[l]) && !((l + 1) in skip) && !((l - 1) in skip)]
    stops = Int[]
    for (s, l) in sort(filter(c -> c[1] > PIT_EXCESS_MS, cand); rev = true)
        any(abs(l - c) <= 1 for c in stops) || push!(stops, l)
    end
    return sort(stops)
end

"""
    LapData

Clean laps for `lap_effects`: per lap the outcome `y` (% of the race's median
clean lap), driver, machine-season and race indices, race-distance fraction
`frac`, stint age `stint`, the driver's centred season age `agec` (decades; 0
when the age is unknown), the race's centred season position `posc` and `lead`
(1 if the car was leading at the end of the lap). With ages, `abin` is the lap's
age bin (the driver's season age in whole years, an index into `age_years`; 0
when unknown) for the population age curve; with circuits, `race_x` holds each
race's standardised circuit covariates (see `prepare_laps`).
"""
struct LapData
    y::Vector{Float64}
    comp::Vector{Int}
    mach::Vector{Int}
    race::Vector{Int}
    frac::Vector{Float64}
    fracc::Vector{Float64}        # frac minus its mean over the race's laps
    stint::Vector{Float64}
    agec::Vector{Float64}         # season age minus the driver's mean age over their laps, decades
    posc::Vector{Float64}         # race's season position (0 first round … 1 last) minus its car's mean
    lead::Vector{Float64}         # 1 if the car led the race at the end of the lap, else 0 (#66)
    stint_cd::Vector{Float64}     # stint/10 minus its mean over the driver's laps (tyre management, #66)
    stint_cm::Vector{Float64}     # stint/10 minus its mean over the car-season's laps
    era::Vector{Int}              # index into `eras` of the lap's season: refuelling or not (#66)
    eras::Vector{Symbol}          # eras present, in order: :refuel (1994–2009), :norefuel
    # lap 1 (starts, #66), one row per car that started from the grid:
    s_y::Vector{Float64}          # lap-1 time in % of the race's median lap-1 time
    s_comp::Vector{Int}
    s_mach::Vector{Int}
    s_race::Vector{Int}
    s_grid::Vector{Float64}       # (grid slot - 1) / 10
    s_race_n::Vector{Int}         # starts per race
    mach_season::Vector{Int}      # season index of each machine level
    race_season::Vector{Int}      # season index of each race
    seasons::Vector{Int}
    competitors::Vector{String}
    machines::Vector{String}
    races::Vector{String}
    race_n::Vector{Int}           # laps per race
    rows::DataFrame               # laps used
    excluded::Dict{Symbol,Int}    # laps dropped, by reason
    # population age curve (#12): one-year bins of the drivers' season ages
    abin::Vector{Int}             # per lap: index into age_years, 0 if the age is unknown
    s_abin::Vector{Int}           # per start
    age_years::Vector{Int}        # the bins (whole years of age)
    age_rows::Vector{Float64}     # laps per bin
    # circuit covariates (#14): per race, standardised across the races
    race_x::Matrix{Float64}       # races × covariates (0 columns without circuits)
    race_x_raw::Matrix{Float64}   # the same, unstandardised
    race_x_names::Vector{Symbol}  # :speed (km/h), :street (score 0–1), :temp (°C)
    # incident covariates (#108, as the race model's track terms, #81): per race
    race_track::Matrix{Float64}   # races × covariates (0 columns without circuits)
    race_track_names::Vector{Symbol}  # :tight (walls score / 2), :mixed, :wet
end

"""
    RaceWalk(d::LapData, cars = car_table())

Race-level chassis walk states for the lap model (#108): one state per works per
race it ran, in race order, so a car's form can move from race to race (upgrades,
B-specs, fades) as well as across seasons. `prev` is the works' previous state (0
for its first); `boundary` marks a step across an off-season, with `own` and `reg`
for a change of ownership and a major chassis rule change in between; `race` and
`counts` give each state's race and the states per race (the states are centred
within race); `lap` and `start` index each lap's and each start's state. A car's
works is its car-season's F1DB constructor's works (see `CarWalks`). `RaceWalk()`
is the empty walk (no states), for a model without it.
"""
struct RaceWalk
    prev::Vector{Int}
    first::Vector{Bool}
    boundary::Vector{Float64}
    own::Vector{Float64}
    reg::Vector{Float64}
    race::Vector{Int}
    counts::Vector{Int}
    names::Vector{String}
    lap::Vector{Int}
    start::Vector{Int}
end
RaceWalk() = RaceWalk(Int[], Bool[], Float64[], Float64[], Float64[], Int[], Int[], String[], Int[], Int[])

function Base.show(io::IO, d::LapData)
    print(io, "LapData($(length(d.y)) laps, $(length(d.competitors)) drivers, $(length(d.machines)) ",
          "machine-seasons, $(length(d.races)) races; excluded ",
          join(("$k $v" for (k, v) in sort(collect(d.excluded))), ", "), ")")
end

"""
    prepare_laps(laps, results; pits = :recorded, pit_stops = nothing, machine_key, circuits = nothing) -> LapData

Clean laps (see the file header) from `laps` (`LAP_SCHEMA`) with each car's
machine from `results` (`RESULT_SCHEMA`). `pits = :recorded` takes in-laps from
`pit_stops` (`PIT_SCHEMA`); `pits = :inferred` infers them from the lap times
(`infer_pit_stops`). Races without lap data for their winner are skipped.

`circuits` (e.g. `circuit_table()`): one row per race with `event_id`,
`length_km`, `street` and `temp_c` (and optionally `circuit_id`). Each race then
gets three covariates (#14): average lap speed (layout length ÷ the median clean
lap, km/h), the street score and the race-window air temperature, standardised
across the races (`race_x`; raw values in `race_x_raw`). With `circuit_id`, the
speed is averaged over the races on the same layout (circuit and length), so it
describes the circuit rather than the day: a wet race isn't a slow circuit.
Every race must be in the table.
"""
function prepare_laps(laps::AbstractDataFrame, results::AbstractDataFrame; pits::Symbol = :recorded,
                      pit_stops = nothing, machine_key = r -> string(r.machine_id, "_", r.season),
                      circuits = nothing, stoppage::Bool = true)
    pits in (:recorded, :inferred) || throw(ArgumentError("pits must be :recorded or :inferred"))
    pits === :recorded && pit_stops === nothing && throw(ArgumentError("pits = :recorded needs pit_stops"))
    mkey = Dict((r.event_id, r.competitor_id) => machine_key(r) for r in eachrow(results))
    gridof = Dict((r.event_id, r.competitor_id) => (hasproperty(results, :grid) ? r.grid : missing) for r in eachrow(results))
    hasage = hasproperty(results, :competitor_birth) && hasproperty(results, :event_date)
    age = Dict((r.event_id, r.competitor_id) =>
               hasage && !ismissing(r.competitor_birth) && !ismissing(r.event_date) ?
               Dates.value(r.event_date - r.competitor_birth) / 365.2425 : NaN for r in eachrow(results))
    recorded = Dict{Tuple{String,String},Set{Int}}()
    if pits === :recorded
        for r in eachrow(pit_stops)
            push!(get!(recorded, (r.event_id, r.competitor_id), Set{Int}()), r.lap)
        end
    end
    excl = Dict(:lap1 => 0, :slow_race => 0, :stoppage => 0, :pit => 0, :no_machine => 0)
    # lap 1 (starts, #66): kept apart, with the car's grid slot; pit-lane starts and
    # unknown grid slots are left out
    start = DataFrame(event_id = String[], competitor_id = String[], machine_key = String[], time_ms = Float64[],
                      grid = Int[])
    keep = DataFrame(season = Int[], event_id = String[], competitor_id = String[], machine_key = String[],
                     lap = Int[], time_ms = Float64[], frac = Float64[], stint = Int[], lead = Float64[])
    for ev in groupby(laps, :event_id)
        L = maximum(ev.lap)
        base = median(ev.time_ms)
        bylap = Dict{Int,Vector{Float64}}()
        for r in eachrow(ev); push!(get!(bylap, r.lap, Float64[]), r.time_ms); end
        slow = Set(l for (l, v) in bylap if median(v) > SLOW_RACE * base)
        skip = union(slow, Set(l + 1 for l in slow))           # restart laps too
        for car in groupby(ev, :competitor_id)
            k = (first(car.event_id), first(car.competitor_id))
            if !haskey(mkey, k)
                excl[:no_machine] += nrow(car); continue
            end
            car = sort(car, :lap)
            car.lap == 1:nrow(car) || continue                  # gaps in a car's laps: unusable
            times = car.time_ms
            pos = hasproperty(car, :position) ? car.position : fill(missing, nrow(car))
            stops = pits === :recorded ? sort(collect(get(recorded, k, Set{Int}()))) : infer_pit_stops(times, skip)
            pitlaps = Set(vcat(stops, stops .+ 1))
            med = median(times)
            g0 = gridof[k]
            if !ismissing(g0) && g0 >= 1 && !isempty(times)
                push!(start, (k[1], k[2], mkey[k], times[1], g0))
            end
            for l in eachindex(times)
                reason = l == 1 ? :lap1 : l in skip ? :slow_race : stoppage && times[l] > STOPPAGE * med ? :stoppage :
                         l in pitlaps ? :pit : nothing
                if reason !== nothing
                    excl[reason] += 1; continue
                end
                last = findlast(<(l), stops)
                stint = last === nothing ? l - 1 : l - stops[last] - 1     # laps since the out-lap, from 1
                push!(keep, (first(car.season), k[1], k[2], mkey[k], l, times[l], l / L, stint,
                             coalesce(pos[l], 0) == 1 ? 1.0 : 0.0))
            end
        end
    end
    # outcome relative to each race's median clean lap
    keep.y = zeros(nrow(keep))
    for g in groupby(keep, :event_id)
        g.y .= 100 .* log.(g.time_ms ./ median(g.time_ms))
    end
    seasons = sort(unique(keep.season))
    si = Dict(s => i for (i, s) in enumerate(seasons))
    comps, machs, races = sort(unique(keep.competitor_id)), sort(unique(keep.machine_key)), sort(unique(keep.event_id))
    ci = Dict(l => i for (i, l) in enumerate(comps))
    mi = Dict(l => i for (i, l) in enumerate(machs))
    ri = Dict(l => i for (i, l) in enumerate(races))
    mach_season, race_season = zeros(Int, length(machs)), zeros(Int, length(races))
    for r in eachrow(keep)
        mach_season[mi[r.machine_key]] = si[r.season]
        race_season[ri[r.event_id]] = si[r.season]
    end
    race = [ri[e] for e in keep.event_id]
    race_n = [count(==(r), race) for r in eachindex(races)]
    # centred age (career slopes): the driver's age in the season (mean over their
    # laps that season) minus their mean age over all their laps. Ages to the day
    # vary by only ~0.05 decades within a season, so within-season pace changes
    # (learning, car development, mid-season moves) became huge per-decade slopes
    # for short careers (#12); season ages leave the slopes only between-season
    # change, and a one-season driver none.
    comp = [ci[c] for c in keep.competitor_id]
    ages = [age[(e, c)] for (e, c) in zip(keep.event_id, keep.competitor_id)]
    ssum, sn = Dict{Tuple{Int,Int},Float64}(), Dict{Tuple{Int,Int},Int}()
    for (k, s, a) in zip(comp, keep.season, ages)
        isnan(a) && continue
        ssum[(k, s)] = get(ssum, (k, s), 0.0) + a; sn[(k, s)] = get(sn, (k, s), 0) + 1
    end
    sage = [isnan(a) ? NaN : ssum[(k, s)] / sn[(k, s)] for (k, s, a) in zip(comp, keep.season, ages)]
    asum, an = zeros(length(comps)), zeros(length(comps))
    for (k, a) in zip(comp, sage)
        isnan(a) || (asum[k] += a; an[k] += 1)
    end
    amean = asum ./ max.(an, 1)
    agec = [isnan(a) ? 0.0 : (a - amean[k]) / 10 for (k, a) in zip(comp, sage)]
    # season position of each race (0 at the season's first round in the data, 1 at
    # its last), centred on each car's mean over its laps (in-season development)
    rnd = Dict(e => parse(Int, split(e, "-")[end]) for e in races)
    lo, hi = Dict{Int,Int}(), Dict{Int,Int}()
    for e in races
        s = race_season[ri[e]]
        lo[s] = min(get(lo, s, typemax(Int)), rnd[e]); hi[s] = max(get(hi, s, 0), rnd[e])
    end
    rpos = [(s = race_season[r]; hi[s] > lo[s] ? (rnd[e] - lo[s]) / (hi[s] - lo[s]) : 0.5) for (r, e) in enumerate(races)]
    mach = [mi[m] for m in keep.machine_key]
    psum, pn = zeros(length(machs)), zeros(length(machs))
    for (m, r) in zip(mach, race); psum[m] += rpos[r]; pn[m] += 1; end
    posc = [rpos[r] - psum[m] / pn[m] for (m, r) in zip(mach, race)]
    fbar = zeros(length(races))
    for (r, f) in zip(race, keep.frac); fbar[r] += f; end
    fbar ./= race_n
    # stint age centred per driver and per car-season (tyre-management slopes, #66):
    # uncentred, a driver's slope would also shift its level (mean stint ≈ 12 laps)
    # and trade off against its effect, as an uncentred age did for career slopes
    s10 = keep.stint ./ 10
    sd_, nd_ = zeros(length(comps)), zeros(length(comps)); sm_, nm_ = zeros(length(machs)), zeros(length(machs))
    for (k, m, s) in zip(comp, mach, s10); sd_[k] += s; nd_[k] += 1; sm_[m] += s; nm_[m] += 1; end
    stint_cd = s10 .- (sd_ ./ max.(nd_, 1))[comp]
    stint_cm = s10 .- (sm_ ./ max.(nm_, 1))[mach]
    # lap-1 rows for the drivers, cars and races that have clean laps
    filter!(r -> haskey(ci, r.competitor_id) && haskey(mi, r.machine_key) && haskey(ri, r.event_id), start)
    s_y = zeros(nrow(start))
    for g in groupby(start, :event_id)
        g.time_ms .= 100 .* log.(g.time_ms ./ median(g.time_ms))      # reused as the outcome
    end
    s_y = Float64.(start.time_ms)
    s_race = [ri[e] for e in start.event_id]
    s_race_n = [count(==(r), s_race) for r in eachindex(races)]
    # age bins (population age curve, #12): the driver's season age in whole years
    sage_ks = Dict(k => ssum[k] / sn[k] for k in keys(ssum))           # (driver, season) => season age
    lapbin = [isnan(a) ? -1 : floor(Int, a) for a in sage]
    age_years = sort(unique(filter(>=(0), lapbin)))
    abi = Dict(y => i for (i, y) in enumerate(age_years))
    abin = [y < 0 ? 0 : abi[y] for y in lapbin]
    age_rows = [Float64(count(==(i), abin)) for i in eachindex(age_years)]
    s_abin = [begin
                  a = get(sage_ks, (ci[c], seasons[race_season[ri[e]]]), NaN)
                  isnan(a) ? 0 : get(abi, floor(Int, a), 0)
              end for (c, e) in zip(start.competitor_id, start.event_id)]
    # circuit covariates (#14), one row per race
    if circuits === nothing
        race_x_raw, race_x_names = zeros(length(races), 0), Symbol[]
    else
        ct = Dict(r.event_id => r for r in eachrow(circuits))
        missing_races = filter(e -> !haskey(ct, e), races)
        isempty(missing_races) || throw(ArgumentError("races missing from the circuit table: $(join(missing_races, ", "))"))
        medms = zeros(length(races))
        for g in groupby(keep, :event_id); medms[ri[first(g.event_id)]] = median(g.time_ms); end
        speed = [ct[e].length_km / (medms[i] / 3.6e6) for (i, e) in enumerate(races)]
        if hasproperty(circuits, :circuit_id)               # per layout: mean over its races
            lay = [(ct[e].circuit_id, ct[e].length_km) for e in races]
            lm = Dict(l => mean(speed[lay .== Ref(l)]) for l in unique(lay))
            speed = [lm[l] for l in lay]
        end
        race_x_raw = hcat(speed,
                          [Float64(ct[e].street) for e in races], [Float64(ct[e].temp_c) for e in races])
        race_x_names = [:speed, :street, :temp]
    end
    race_track, race_track_names = circuits === nothing || !hasproperty(circuits, :condition) ? (zeros(length(races), 0), Symbol[]) :
        (hcat([ct[e].W / 2 for e in races], [Float64(ct[e].condition == "mixed") for e in races],
              [Float64(ct[e].condition == "wet") for e in races]), [:tight, :mixed, :wet])
    race_x = copy(race_x_raw)
    for k in axes(race_x, 2)
        sdk = std(race_x[:, k])
        race_x[:, k] .= sdk > 0 ? (race_x[:, k] .- mean(race_x[:, k])) ./ sdk : 0.0
    end
    eralab = [s in REFUEL_SEASONS ? :refuel : :norefuel for s in keep.season]
    eras = [e for e in (:refuel, :norefuel) if e in eralab]
    era = [findfirst(==(e), eras) for e in eralab]
    return LapData(keep.y, comp, mach,
                   race, keep.frac, keep.frac .- fbar[race], Float64.(keep.stint), agec, posc, keep.lead, stint_cd, stint_cm,
                   era, eras, s_y, [ci[c] for c in start.competitor_id], [mi[m] for m in start.machine_key], s_race,
                   (start.grid .- 1) ./ 10, s_race_n, mach_season, race_season, seasons, comps, machs, races, race_n, keep, excl,
                   abin, s_abin, age_years, age_rows, race_x, race_x_raw, race_x_names, race_track, race_track_names)
end

"""
    lap_loglik(γ, δ, a, b, β, sl, dv, η, td, tm, fage, cx, xr, σ, πr, λr, πd, ρ, λ2, ηt, φ, sh, κ, rwk, ctx, d::LapData)

Log-likelihood of the laps in `d` under pace + loss noise (`paceloss_logpdf`),
with incident probability `πr[race]` shifted on the logit scale by the driver's
offset `πd[driver]` (#108; zeros without incident proneness) and mean loss
`λr[race]` (#108: per race with the track terms, else the same for every race),
with a share `ρ` of incidents being big losses of mean `λ2` (#108, as the race
model's #16: `pacebig_logpdf`; ρ = 0 is the single-loss form), and per-lap mean

    μ = γ[race] + δ[race]·(frac - mean frac of the race) + h - (mean h of the race),
    h = a[driver] + sl[driver]·agec + b[machine] + dv[machine]·posc + η[1]·lead
        + β[2e-1]·s + β[2e]·s² + td[driver]·s_d + tm[machine]·s_m
        + fage[age bin] + Σ_k cx[machine, k]·x[race, k],   s = stint/10, e = the lap's era

(stint age in tens of laps, which keeps β's two terms on similar scales; `sl`
are the drivers' career slopes in % per decade of age and `dv` the cars'
in-season development trends in % per season; η the lead-lap effect in %; `td`
and `tm` the drivers' and cars' deviations from the shared tyre slope, in % per
10 laps of stint age, on s_d and s_m: s centred on the driver's and on the
car-season's mean; `fage` the population age curve per age bin, 0 for laps of
unknown age; `cx` the car-seasons' slopes on the race's circuit covariates
`d.race_x`, stored column by column (covariate k of machine m at
(k - 1)·n_machines + m); see `lap_effects`). As in
the race-time model, centring within each race makes γ the race's mean lap, so
the intercepts are not correlated with the slopes and effects: with ~1,000 laps
per race the uncentred form had near-degenerate directions that kept NUTS at
its maximum tree depth. Has a fused reverse rule: per-lap derivatives from
ForwardDiff duals, scattered into the parameter vectors.

#113 terms, switched on by `ctx` (`LapContext`): `ηt`, the traffic bins' effects
(h += ηt[bin] for each of the lap's `ctx.tbin` bins); `φ`, each car-season's slope on
the field pace (h += φ[machine]·fp); and `sh`, the shortfall offsets, laid out as
[driver logit π; car-season logit π; driver log λ; car-season log λ]. With them the
lap's incident probability and loss mean are
    logit q = logit πr[race] (+ πd[driver] without big losses) + sh_π[driver] + sh_π[machine],
    log λ_lap = log λr[race] + sh_λ[driver] + sh_λ[machine],
(with `ctx.fpace` also + κ[1]·fp on logit q and + κ[2]·fp on log λ_lap: the field
pace moves the shortfall too, so slow laps of a wet spell are conditions as well as
losses), and μ is the lap's mean pace rather than its clean-lap location: the pace noise is
centred on μ - q_b·λ_b, the edge the shortfall is measured from, with q_b and λ_b the
lap's q and λ_lap without the field-pace term. Mean pace is what the laps pin down; the
edge is split from it only by the shape of the distribution, a ridge the sampler would
otherwise have to follow (#113). The field pace's part is left out of the edge: in wet
races (λ ≈ 6 %, fp up to ±9 %) it moved the edge by tens of % of lap time per unit κ,
and the 2011–2026 fit diverged on nearly every iteration. Off, these are unused.
"""
function lap_loglik(γ, δ, a, b, β, sl, dv, η, td, tm, fage, cx, xr, σ, πr, λr, πd, ρ, λ2, ηt, φ, sh, κ, rwk::RaceWalk,
                    ctx, d::LapData)
    h, hbar = lap_h(a, b, β, sl, dv, η, td, tm, fage, cx, xr, ηt, φ, rwk, ctx, d)
    nc, nm = length(d.competitors), length(d.machines)
    s = 0.0
    for j in eachindex(d.y)
        r = d.race[j]
        μ = γ[r] + δ[r] * d.fracc[j] + h[j] - hbar[r]
        if ctx.short        # driver and car-season shortfall offsets; μ the mean pace (#113)
            c, m = d.comp[j], d.mach[j]
            f = ctx.fpace ? ctx.fp[j] : 0.0
            xb = logit(πr[r]) + (iszero(ρ) ? πd[c] : zero(eltype(πd))) + sh[c] + sh[nc + m]
            λb = λr[r] * exp(sh[nc + nm + c] + sh[2nc + nm + m])
            q, λl = logistic(xb + κ[1] * f), λb * exp(κ[end] * f)
            e = μ - logistic(xb) * λb       # the edge from the base shortfall: the field pace only shapes the losses
            s += iszero(ρ) ? paceloss_logpdf(d.y[j] - e, σ, q, λl) :
                 pacebig_logpdf(d.y[j] - e, σ, q, λl, logistic(logit(ρ) + πd[c]), λ2)
        elseif iszero(ρ)    # single loss: the driver's offset is on the incident probability
            s += paceloss_logpdf(d.y[j] - μ, σ, row_π(πr, r, πd, d.comp[j]), λr[r])
        else                # with big losses it is on the big-loss share (#108)
            s += pacebig_logpdf(d.y[j] - μ, σ, πr[r], λr[r], logistic(logit(ρ) + πd[d.comp[j]]), λ2)
        end
    end
    return s
end

# The circuit term of machine m at race r: Σ_k cx[(k - 1)·n_machines + m]·x[r, k]
@inline function circuit_term(cx, x, m, r, nm)
    v = zero(eltype(cx))
    for k in axes(x, 2)
        v += cx[(k - 1) * nm + m] * x[r, k]
    end
    return v
end

# Per-lap h = driver + career slope + car + car development + lead lap + tyre terms
# + age curve + circuit slopes (+ traffic bins and field-pace slopes, #113), and its mean per race.
function lap_h(a, b, β, sl, dv, η, td, tm, fage, cx, xr, ηt, φ, rwk::RaceWalk, ctx, d::LapData)
    T = promote_type(eltype(a), eltype(b), eltype(β), eltype(sl), eltype(dv), eltype(η), eltype(td), eltype(tm),
                     eltype(fage), eltype(cx), eltype(xr), eltype(ηt), eltype(φ))
    rw = !isempty(rwk.lap)
    h = Vector{T}(undef, length(d.y))
    hbar = zeros(T, length(d.races))
    nm = length(d.machines)
    for j in eachindex(d.y)
        st = d.stint[j] / 10
        h[j] = a[d.comp[j]] + sl[d.comp[j]] * d.agec[j] + b[d.mach[j]] + dv[d.mach[j]] * d.posc[j] +
               η[1] * d.lead[j] + β[2d.era[j] - 1] * st + β[2d.era[j]] * st^2 +
               td[d.comp[j]] * d.stint_cd[j] + tm[d.mach[j]] * d.stint_cm[j] +
               (d.abin[j] > 0 ? fage[d.abin[j]] : zero(T)) + circuit_term(cx, d.race_x, d.mach[j], d.race[j], nm) +
               (rw ? xr[rwk.lap[j]] : zero(T))        # the car's works at this race (race-level chassis walk)
        if ctx.traffic
            for g in axes(ctx.tbin, 2)
                k = ctx.tbin[j, g]
                k > 0 && (h[j] += ηt[k])
            end
        end
        ctx.fpace && (h[j] += φ[d.mach[j]] * ctx.fp[j])
        hbar[d.race[j]] += h[j]
    end
    hbar ./= d.race_n
    return h, hbar
end

function ChainRulesCore.rrule(::typeof(lap_loglik), γ, δ, a, b, β, sl, dv, η, td, tm, fage, cx, xr, σ, πr, λr, πd, ρ,
                              λ2, ηt, φ, sh, κ, rwk::RaceWalk, ctx, d::LapData)
    h, hbar = lap_h(a, b, β, sl, dv, η, td, tm, fage, cx, xr, ηt, φ, rwk, ctx, d)
    gxr, rw = zeros(length(xr)), !isempty(rwk.lap)
    gηt, gφ, gsh, gκ = zeros(length(ηt)), zeros(length(φ)), zeros(length(sh)), zeros(length(κ))
    κπ, κλ = ctx.fpace ? (κ[1], κ[end]) : (0.0, 0.0)
    nc = length(d.competitors)
    N = length(d.y)
    gμ = zeros(N)
    gγ, gδ, ga, gb, gβ = zeros(length(γ)), zeros(length(δ)), zeros(length(a)), zeros(length(b)), zeros(length(β))
    gsl, gdv = zeros(length(sl)), zeros(length(dv))
    gη, gtd, gtm = zeros(length(η)), zeros(length(td)), zeros(length(tm))
    gfage, gcx = zeros(length(fage)), zeros(length(cx))
    nm = length(d.machines)
    gσ, gρ, gλ2, val = 0.0, 0.0, 0.0, 0.0
    big = !iszero(ρ)
    lρ = big ? (log1p(-ρ), log(ρ)) : (0.0, 0.0)
    gπr, gλr, gπd = zeros(length(πr)), zeros(length(λr)), zeros(length(πd))
    # per race: logit π and the log mixture weights, so a lap without a driver offset
    # costs no extra exp or log
    Lr = [log(p / (1 - p)) for p in πr]
    lqr = [(log1p(-p), log(p)) for p in πr]
    noπd = all(iszero, πd)
    # with big losses the drivers' offsets are on the big-loss share (#108), once per driver
    ρd = big && !noπd ? [logistic(logit(ρ) + o) for o in πd] : Float64[]
    lρd = big && !noπd ? [(log1p(-p), log(p)) for p in ρd] : Tuple{Float64,Float64}[]
    # without: on π; a driver's laps in a race are consecutive, so their π is computed once
    rc_last, q, lq = (0, 0), 0.0, (0.0, 0.0)
    λl, xb, λb = 0.0, 0.0, 0.0
    for j in 1:N
        r, y, c = d.race[j], d.y[j], d.comp[j]
        μ = γ[r] + δ[r] * d.fracc[j] + h[j] - hbar[r]
        if ctx.short        # driver and car-season shortfall offsets, μ the mean pace (#113; see lap_loglik)
            m = d.mach[j]
            if (r, c) != rc_last                    # a driver's laps in a race: consecutive, one machine
                xb = Lr[r] + (big ? 0.0 : πd[c]) + sh[c] + sh[nc + m]
                λb = λr[r] * exp(sh[nc + nm + c] + sh[2nc + nm + m])
                rc_last = (r, c)
            end
            f = ctx.fpace ? ctx.fp[j] : 0.0         # the field pace on the shortfall, per lap
            x = xb + κπ * f
            q = logistic(x)
            lq = (-log1pexp(x), x - log1pexp(x))
            λl = λb * exp(κλ * f)
            qb = logistic(xb)
            e = μ - qb * λb                         # the edge from the base shortfall (see lap_loglik)
            if big
                ρl, lρl = noπd ? (ρ, lρ) : (ρd[c], lρd[c])
                v, pμ, pσ, pπ, pλ, pρ, pλ2 = pacebig_row_vg(y, 0.0, 0.0, false, e, σ, q, λl, ρl, λ2, lq, lρl)
                gλ2 += pλ2
                if noπd
                    gρ += pρ
                else
                    w = pρ * ρl * (1 - ρl)
                    gρ += w / (ρ * (1 - ρ)); gπd[c] += w
                end
            else
                v, pμ, pσ, pπ, pλ = paceloss_row_vg(y, e, σ, q, λl, lq)
            end
            # e = μ - q_b·λ_b: the base shortfall also moves the location, the field pace's part doesn't
            wqm, wλm = pπ * q * (1 - q), pλ * λl     # through the mixture: ∂ℓ/∂(logit q), ∂ℓ/∂(log λ_lap)
            wq = wqm - pμ * λb * qb * (1 - qb)       # ∂ℓ/∂(base logit q)
            wλ = wλm - pμ * qb * λb                  # ∂ℓ/∂(base log λ)
            gπr[r] += wq / (πr[r] * (1 - πr[r]))
            big || (gπd[c] += wq)
            gsh[c] += wq; gsh[nc + m] += wq
            gsh[nc + nm + c] += wλ; gsh[2nc + nm + m] += wλ
            ctx.fpace && (gκ[1] += wqm * f; gκ[end] += wλm * f)
            val += v
            gμ[j] = pμ
            gγ[r] += pμ; gδ[r] += pμ * d.fracc[j]
            gσ += pσ; gλr[r] += wλ / λr[r]
            continue
        end
        if noπd || big
            q, lq = πr[r], lqr[r]
        elseif (r, c) != rc_last
            x = Lr[r] + πd[c]
            q = logistic(x)
            lq = (-log1pexp(x), x - log1pexp(x))
            rc_last = (r, c)
        end
        if big
            ρl, lρl = noπd ? (ρ, lρ) : (ρd[c], lρd[c])
            v, pμ, pσ, pπ, pλ, pρ, pλ2 = pacebig_row_vg(y, 0.0, 0.0, false, μ, σ, q, λr[r], ρl, λ2, lq, lρl)
            gλ2 += pλ2
            if noπd
                gρ += pρ
            else
                w = pρ * ρl * (1 - ρl)               # ∂ℓ/∂(logit of the lap's big-loss share)
                gρ += w / (ρ * (1 - ρ)); gπd[c] += w
            end
            gπr[r] += pπ
        else
            v, pμ, pσ, pπ, pλ = paceloss_row_vg(y, μ, σ, q, λr[r], lq)
            w = pπ * q * (1 - q)                     # ∂ℓ/∂(logit of the lap's π)
            gπr[r] += w / (πr[r] * (1 - πr[r])); gπd[c] += w
        end
        val += v
        gμ[j] = pμ
        gγ[r] += pμ; gδ[r] += pμ * d.fracc[j]
        gσ += pσ; gλr[r] += pλ
    end
    # h enters every lap of its race through the race mean: dμ_k/dh_j = [k = j] - 1/n_race
    for j in 1:N
        gh = gμ[j] - gγ[d.race[j]] / d.race_n[d.race[j]]
        st = d.stint[j] / 10
        ga[d.comp[j]] += gh; gb[d.mach[j]] += gh
        gsl[d.comp[j]] += gh * d.agec[j]
        gdv[d.mach[j]] += gh * d.posc[j]
        gβ[2d.era[j] - 1] += gh * st; gβ[2d.era[j]] += gh * st^2
        gη[1] += gh * d.lead[j]
        gtd[d.comp[j]] += gh * d.stint_cd[j]; gtm[d.mach[j]] += gh * d.stint_cm[j]
        d.abin[j] > 0 && (gfage[d.abin[j]] += gh)
        for k in axes(d.race_x, 2)
            gcx[(k - 1) * nm + d.mach[j]] += gh * d.race_x[d.race[j], k]
        end
        rw && (gxr[rwk.lap[j]] += gh)
        if ctx.traffic
            for g in axes(ctx.tbin, 2)
                k = ctx.tbin[j, g]
                k > 0 && (gηt[k] += gh)
            end
        end
        ctx.fpace && (gφ[d.mach[j]] += gh * ctx.fp[j])
    end
    # the closure captures the scalar sums through `gs`, bound once: capturing gσ, gρ,
    # gλ2 themselves (reassigned in the loop) boxed them, and the loop's updates were
    # dynamic, ~40% of a gradient (#102)
    gs = (gσ, gρ, gλ2)
    pullback(Δ) = (NoTangent(), Δ .* gγ, Δ .* gδ, Δ .* ga, Δ .* gb, Δ .* gβ, Δ .* gsl, Δ .* gdv, Δ .* gη, Δ .* gtd, Δ .* gtm,
                   Δ .* gfage, Δ .* gcx, Δ .* gxr, Δ * gs[1], Δ .* gπr, Δ .* gλr, Δ .* gπd, Δ * gs[2], Δ * gs[3],
                   Δ .* gηt, Δ .* gφ, Δ .* gsh, Δ .* gκ, NoTangent(), NoTangent(), NoTangent())
    return val, pullback
end
# β (tyres), sl (career slopes), dv (car development), η (lead laps), td, tm
# (tyre management), fage (age curve) and cx (circuit slopes) are tracked when
# sampled and constant zero vectors when the model leaves them out
# (`lap_effects(d; tyres, slopes, dev, lead, tyre_mgmt, age, circuits)`)
const _TA_OR_CONST = (ReverseDiff.TrackedArray, Vector{Float64})
for Tβ in _TA_OR_CONST, Tsl in _TA_OR_CONST, Tdv in _TA_OR_CONST, Tη in _TA_OR_CONST, Tt in _TA_OR_CONST,
    Tf in _TA_OR_CONST, Tc in _TA_OR_CONST, Td in _TA_OR_CONST
    @eval ReverseDiff.@grad_from_chainrules lap_loglik(γ::ReverseDiff.TrackedArray, δ::ReverseDiff.TrackedArray,
                                                       a::ReverseDiff.TrackedArray, b::ReverseDiff.TrackedArray,
                                                       β::$Tβ, sl::$Tsl, dv::$Tdv, η::$Tη, td::$Tt, tm::$Tt,
                                                       fage::$Tf, cx::$Tc, xr::ReverseDiff.TrackedArray,
                                                       σ::ReverseDiff.TrackedReal,
                                                       πr::ReverseDiff.TrackedArray, λr::ReverseDiff.TrackedArray,
                                                       πd::$Td, ρ::ReverseDiff.TrackedReal, λ2::ReverseDiff.TrackedReal,
                                                       ηt::ReverseDiff.TrackedArray, φ::ReverseDiff.TrackedArray,
                                                       sh::ReverseDiff.TrackedArray, κ::ReverseDiff.TrackedArray,
                                                       rwk::RaceWalk, ctx, d::LapData)
end

"""
    start_loglik(γ1, a, b, sd, sc, g, fage, cx, σ, π, λ, πd, d::LapData)

Log-likelihood of the lap-1 times in `d` (#66) under pace + loss noise, with
per-start mean

    μ = γ1[race] + h - (mean h of the race's starts),
    h = a[driver] + b[machine] + sd[driver] + sc[machine] + g[1]·r + g[2]·r²
        + fage[age bin] + Σ_k cx[machine, k]·x[race, k],   r = (grid slot - 1)/10

`a` and `b` are the drivers' and cars' pace effects shared with the clean laps,
as are the age curve `fage` and circuit slopes `cx` (see `lap_loglik`);
`sd` and `sc` their start effects (lap 1 against their pace), and `g` the cost of
starting further back. Centred within race as `lap_loglik`, with a fused
reverse rule.
"""
function start_loglik(γ1, a, b, sd, sc, g, fage, cx, xr, σ, π, λ, πd, rwk::RaceWalk, d::LapData)
    h, hbar = start_h(a, b, sd, sc, g, fage, cx, xr, rwk, d)
    s = 0.0
    for j in eachindex(d.s_y)
        r = d.s_race[j]
        q = logistic(logit(π) + πd[d.s_comp[j]])          # the driver's lap-1 incident offset (#108)
        s += paceloss_logpdf(d.s_y[j] - (γ1[r] + h[j] - hbar[r]), σ, q, λ)
    end
    return s
end

function start_h(a, b, sd, sc, g, fage, cx, xr, rwk::RaceWalk, d::LapData)
    T = promote_type(eltype(a), eltype(b), eltype(sd), eltype(sc), eltype(g), eltype(fage), eltype(cx), eltype(xr))
    rw = !isempty(rwk.start)
    h = Vector{T}(undef, length(d.s_y))
    hbar = zeros(T, length(d.races))
    nm = length(d.machines)
    for j in eachindex(d.s_y)
        gr = d.s_grid[j]
        h[j] = a[d.s_comp[j]] + b[d.s_mach[j]] + sd[d.s_comp[j]] + sc[d.s_mach[j]] + g[1] * gr + g[2] * gr^2 +
               (d.s_abin[j] > 0 ? fage[d.s_abin[j]] : zero(T)) + circuit_term(cx, d.race_x, d.s_mach[j], d.s_race[j], nm) +
               (rw ? xr[rwk.start[j]] : zero(T))
        hbar[d.s_race[j]] += h[j]
    end
    for r in eachindex(hbar); d.s_race_n[r] > 0 && (hbar[r] /= d.s_race_n[r]); end
    return h, hbar
end

function ChainRulesCore.rrule(::typeof(start_loglik), γ1, a, b, sd, sc, g, fage, cx, xr, σ, π, λ, πd, rwk::RaceWalk,
                              d::LapData)
    h, hbar = start_h(a, b, sd, sc, g, fage, cx, xr, rwk, d)
    gxr, rw = zeros(length(xr)), !isempty(rwk.start)
    N = length(d.s_y)
    gμ = zeros(N)
    gγ, ga, gb, gsd, gsc, gg = zeros(length(γ1)), zeros(length(a)), zeros(length(b)), zeros(length(sd)),
                               zeros(length(sc)), zeros(2)
    gfage, gcx = zeros(length(fage)), zeros(length(cx))
    nm = length(d.machines)
    gσ, gπ, gλ, val = 0.0, 0.0, 0.0, 0.0
    gπd = zeros(length(πd))
    L = log(π / (1 - π))
    for j in 1:N
        r, y, c = d.s_race[j], d.s_y[j], d.s_comp[j]
        μ = γ1[r] + h[j] - hbar[r]
        x = L + πd[c]                                  # the driver's lap-1 incident offset (#108)
        q = logistic(x)
        v, pμ, pσ, pπ, pλ = paceloss_row_vg(y, μ, σ, q, λ, (-log1pexp(x), x - log1pexp(x)))
        val += v
        gμ[j] = pμ; gγ[r] += pμ
        gσ += pσ; gλ += pλ
        w = pπ * q * (1 - q)                           # ∂ℓ/∂(logit of the start's π)
        gπ += w / (π * (1 - π)); gπd[c] += w
    end
    for j in 1:N
        gh = gμ[j] - gγ[d.s_race[j]] / d.s_race_n[d.s_race[j]]
        gr = d.s_grid[j]
        ga[d.s_comp[j]] += gh; gb[d.s_mach[j]] += gh
        gsd[d.s_comp[j]] += gh; gsc[d.s_mach[j]] += gh
        gg[1] += gh * gr; gg[2] += gh * gr^2
        d.s_abin[j] > 0 && (gfage[d.s_abin[j]] += gh)
        for k in axes(d.race_x, 2)
            gcx[(k - 1) * nm + d.s_mach[j]] += gh * d.race_x[d.s_race[j], k]
        end
        rw && (gxr[rwk.start[j]] += gh)
    end
    gs = (gσ, gπ, gλ)          # bound once, so the closure does not box the loop's sums (see lap_loglik)
    pullback(Δ) = (NoTangent(), Δ .* gγ, Δ .* ga, Δ .* gb, Δ .* gsd, Δ .* gsc, Δ .* gg, Δ .* gfage, Δ .* gcx, Δ .* gxr,
                   Δ * gs[1], Δ * gs[2], Δ * gs[3], Δ .* gπd, NoTangent(), NoTangent())
    return val, pullback
end
for Tf in _TA_OR_CONST, Tc in _TA_OR_CONST, Td in _TA_OR_CONST
    @eval ReverseDiff.@grad_from_chainrules start_loglik(γ1::ReverseDiff.TrackedArray, a::ReverseDiff.TrackedArray,
                                                         b::ReverseDiff.TrackedArray, sd::ReverseDiff.TrackedArray,
                                                         sc::ReverseDiff.TrackedArray, g::ReverseDiff.TrackedArray,
                                                         fage::$Tf, cx::$Tc, xr::ReverseDiff.TrackedArray,
                                                         σ::ReverseDiff.TrackedReal, π::ReverseDiff.TrackedReal,
                                                         λ::ReverseDiff.TrackedReal, πd::$Td, rwk::RaceWalk, d::LapData)
end

"""
    lap_effects(d::LapData; tyres = true, slopes = false, dev = false, lead = false, tyre_mgmt = false, starts = false,
                age = false, circuits = false)

Lap time (% of the race's median clean lap) = race intercept + race slope·frac
+ driver + car-season + tyre(stint age) + pace noise + incident loss.

- Driver effects `a_comp ~ N(0, σ_comp)` and car-season effects
  `b_mach ~ N(0, σ_mach)`, centred (see the model body); in the likelihood the
  drivers sum to zero and the cars sum to zero within each season (with race
  intercepts only within-season car differences are identified), as in `gap_effects`.
- Race intercepts γ ~ N(μ_γ, 2), with μ_γ learned: each race's mean lap (the
  other terms are centred within the race, see `lap_loglik`). Race slopes
  δ ~ N(μ_δ, τ_δ): the change in lap time from the start to the end of a race
  (fuel burn, track evolution), partially pooled.
- Tyres: β[1]·s + β[2]·s² with s = stint age / 10 laps, in %, one curve per era
  present (`d.eras`: refuelling 1994–2009, or not; β has 2 entries per era).
  With refuelling, fuel burn within a stint follows stint age too, so the
  curve is a within-stint pace change (tyres and fuel), kept out of the driver
  and car effects (#66). With
  `tyres = false` there are no tyre terms (β = 0): a check of whether they move
  the effects at all.
- Career slopes (`slopes = true`, #15 in the race model): a linear trend per
  driver over their age, sl[driver]·agec with agec the lap's age minus the
  driver's mean age in decades, so it is orthogonal to the driver's constant
  effect. Over a long span of seasons a driver's pace isn't constant (Kubica's
  comeback, a young driver's improvement), and a constant effect then misplaces
  them. The slopes are centred across drivers (a common slope is not identified:
  the age–period–cohort problem, see `CareerRows`), sl = τ_slope·(u - mean u),
  u ~ N(0, 1), τ_slope ~ half-N(0, 0.5) in % per decade. Ages are per season
  (see `prepare_laps`), so the slopes fit only between-season change.
- In-season car development (`dev = true`, #14 in the race model): a linear
  trend per car-season over the season, dv[machine]·posc with posc the race's
  position in its season (0 first round, 1 last) minus the car's mean position.
  A trend shared by the whole field is absorbed by the race intercepts, so the
  trends are centred within season: dv = τ_dev·(u - season mean u), u ~ N(0, 1),
  τ_dev ~ half-N(0, 1) in % per season. Without it, within-season car changes
  are attributed to the drivers (their slopes, #12).
- Lead laps (`lead = true`, #66): η·lead, a shared effect of leading the race
  on that lap (in %). It can't separate clean air from pace management, which
  coincide; with the driver and car effects in the model it is the within-driver
  difference between leading and not. η ~ N(0, 1).
- Tyre management (`tyre_mgmt = true`, #66): driver and car-season deviations
  from the shared tyre slope, td[driver]·s_d + tm[machine]·s_m, with the stint
  age s centred on the driver's (s_d) and the car-season's (s_m) mean, so a slope
  doesn't also move the level (as agec for the career slopes). Centred across
  drivers and within season, so they don't absorb the shared slope β[1]: td = τ_td·(u - mean u),
  tm = τ_tm·(u - season mean u), u ~ N(0, 1), τ ~ half-N(0, 0.5) in % per 10
  laps. Without compound data they mix tyre management with compound choice.
- Starts (`starts = true`, #66): lap 1 as its own term (`start_loglik`), with
  its own race intercepts γ1 ~ N(0, 2), the drivers' and cars' pace effects
  plus start effects sd (drivers, centred) and sc (car-seasons, centred within
  season), sd = τ_sd·(u - mean u), sc = τ_sc·(u - season mean u), τ ~ half-N(0, 1),
  a grid-slot curve g[1]·r + g[2]·r² (r = (slot - 1)/10, g ~ N(0, 2)), and its own
  pace + loss noise (σ1, a_π1, a_λ1). Pit-lane starts are left out.
- Population age curve (`age = true`, #12; `AgeCurveBasis` as in the race
  model): a shared curve over the drivers' season ages in one-year bins,
  fage = proj·walk·(τ_age·e_age), e_age ~ N(0, 1), τ_age ~ half-N(0, 0.2) in %
  per year of age. Only its shape is identified: the constant goes into the
  driver effects and the linear part into the race intercepts and career slopes
  (age–period–cohort), so it is projected onto curves with neither. With it, the
  career slopes only explain how a driver's trajectory differs from the common
  one, which keeps a team's improvement from passing for both its drivers'
  steep slopes (McLaren 2023–25, #12). Laps of unknown age get no curve.
- Circuit slopes (`circuits = true`, #14; needs `prepare_laps(…; circuits)`):
  each car-season's slope on each of the race's standardised circuit covariates
  (average lap speed, street score, race temperature), Σ_k cx[machine, k]·x[race, k].
  The race intercepts take what is shared by the whole field, so the slopes are
  centred within season like the development trends: cx[:, k] = τ_cx[k]·(u - season
  mean u), u ~ N(0, 1), τ_cx ~ half-N(0, 0.5) in % per standard deviation of the
  covariate. A covariate that doesn't matter shrinks τ_cx[k] to zero.
- Noise: pace + loss (`paceloss_logpdf`) with one σ, incident probability π
  and mean loss λ: traffic, mistakes and damage are losses, as in the race model.
- Maximum pace and consistency (#113, switched on by `ctx`, a `LapContext`):
  `traffic`, shared effects of the traffic bins, η_tr ~ N(0, 1) in %; `fpace`, each
  car-season's slope on the field pace, φ = β_fp + τ_fp·(u - season mean u),
  β_fp ~ N(1, 0.5), τ_fp ~ half-N(0, 0.5), and with `short` the field pace on the
  shortfall's logit π and log λ too, κ_fp ~ N(0, 1) (so a wet spell's slow laps aren't
  only losses, #113); `short`, driver and car-season offsets
  on the shortfall's logit π and log λ (four blocks, each τ_sh[k]·(u - mean u),
  τ_sh ~ half-N(0, 0.5)), with the driver and car effects then on the mean pace and
  the clean-lap edge below it by the expected shortfall (see `lap_loglik`).
"""
@model function lap_effects(d::LapData, mach_counts::Vector{Int}, age_basis, tyres::Bool, slopes::Bool, dev::Bool,
                            lead::Bool, tyre_mgmt::Bool, starts::Bool, age::Bool, circuits::Bool, track::Bool = false,
                            dinc::Bool = false, cw = nothing, works::Bool = false, big::Bool = false,
                            rw = nothing, sdrv::Bool = true, dinc1::Bool = false, ctx = LapContext())
    σ_comp ~ truncated(Normal(0, 2); lower = 0)
    rw === nothing && (σ_mach ~ truncated(Normal(0, 2); lower = 0))   # no car-season residual with the race walk
    σ ~ truncated(Normal(0, 1); lower = 0)
    # centred effects (as `_cdrv`, #28): each driver and car has thousands of laps,
    # so the non-centred form a = σ·z would tie σ to every z (a funnel)
    a_comp ~ IsoNormalVec(0.0, σ_comp, length(d.competitors))
    rw === nothing && (b_mach ~ IsoNormalVec(0.0, σ_mach, length(d.machines)))
    μ_γ ~ Normal(0, 2)
    γ ~ IsoNormalVec(μ_γ, 2.0, length(d.races))
    μ_δ ~ Normal(0, 2)
    τ_δ ~ truncated(Normal(0, 2); lower = 0)
    δ ~ IsoNormalVec(μ_δ, τ_δ, length(d.races))
    if tyres
        β ~ StdNormalVec(2length(d.eras))
    else
        β = zeros(2length(d.eras))
    end
    if slopes
        τ_slope ~ truncated(Normal(0, 0.5); lower = 0)
        u_slope ~ StdNormalVec(length(d.competitors))
        sl = career_slopes(τ_slope, u_slope)
    else
        sl = zeros(length(d.competitors))
    end
    if dev
        τ_dev ~ truncated(Normal(0, 1); lower = 0)
        u_dev ~ StdNormalVec(length(d.machines))
        dv = τ_dev .* center_by(u_dev, d.mach_season, mach_counts)
    else
        dv = zeros(length(d.machines))
    end
    if lead
        η ~ StdNormalVec(1)
    else
        η = zeros(1)
    end
    if tyre_mgmt
        τ_td ~ truncated(Normal(0, 0.5); lower = 0)
        u_td ~ StdNormalVec(length(d.competitors))
        τ_tm ~ truncated(Normal(0, 0.5); lower = 0)
        u_tm ~ StdNormalVec(length(d.machines))
        td = career_slopes(τ_td, u_td)                         # τ·(u - mean u)
        tm = τ_tm .* center_by(u_tm, d.mach_season, mach_counts)
    else
        td, tm = zeros(length(d.competitors)), zeros(length(d.machines))
    end
    if starts
        γ1 ~ IsoNormalVec(0.0, 2.0, length(d.races))
        if sdrv             # per-driver start effects (off in #108: barely identified)
            τ_sd ~ truncated(Normal(0, 1); lower = 0)
            u_sd ~ StdNormalVec(length(d.competitors))
        end
        τ_sc ~ truncated(Normal(0, 1); lower = 0)
        u_sc ~ StdNormalVec(length(d.machines))
        g1 ~ IsoNormalVec(0.0, 2.0, 2)
        σ1 ~ truncated(Normal(0, 2); lower = 0)
        a_π1 ~ Normal(-1.5, 1)
        a_λ1 ~ Normal(0, 1)
    end
    if age
        τ_age ~ truncated(Normal(0, 0.2); lower = 0)
        e_age ~ StdNormalVec(length(d.age_years) - 1)
        fage = age_curve(τ_age, e_age, age_basis)
    else
        fage = zeros(length(d.age_years))
    end
    nm, nk = length(d.machines), size(d.race_x, 2)
    if circuits
        τ_cx ~ filldist(truncated(Normal(0, 0.5); lower = 0), nk)
        u_cx ~ StdNormalVec(nm * nk)
        # centred within season, covariate by covariate, in one call (each covariate's
        # seasons as their own groups); slicing would put every element on the tape
        ns = length(mach_counts)
        cx = τ_cx[repeat(1:nk; inner = nm)] .*
             center_by(u_cx, reduce(vcat, [d.mach_season .+ (k - 1) * ns for k in 1:nk]), repeat(mach_counts, nk))
    else
        cx = zeros(nm * nk)
    end
    a_π ~ Normal(-1.5, 1)
    a_λ ~ Normal(0, 1)
    nr = length(d.races)
    if track                # walls, mixed and wet races on logit π and log λ (#108, as #81)
        nt = size(d.race_track, 2)
        β_track_π ~ StdNormalVec(nt)
        β_track_λ ~ StdNormalVec(nt)
        πr = logistic.(a_π .+ d.race_track * β_track_π)
        λr = exp.(a_λ .+ d.race_track * β_track_λ)
    else                    # the same for every race (a vector, so the likelihood has one form)
        πr = logistic.(a_π .+ zeros(nr))
        λr = exp.(a_λ .+ zeros(nr))
    end
    if dinc                 # per-driver incident proneness, centred (#108, as #101): on the big-loss share
                            # with big losses, else on logit π (see lap_loglik)
        τ_dinc ~ truncated(Normal(0, 0.5); lower = 0)
        u_dinc ~ StdNormalVec(length(d.competitors))
        πd = τ_dinc .* (u_dinc .- mean(u_dinc))
    else
        πd = zeros(length(d.competitors))
    end
    a = sum_to_zero(a_comp)
    if rw === nothing
        b = center_by(b_mach, d.mach_season, mach_counts)
        xr = 0 .* a_π .+ zeros(1)            # no race walk: a tracked placeholder, unused
        rwk = RaceWalk()
    else                    # race-level chassis walk (#108)
        rwk = rw.rwk
        τ_cr ~ truncated(Normal(0, 0.5); lower = 0)           # chassis step per race, %
        τ_co ~ truncated(Normal(0, 1); lower = 0)             # chassis step across an off-season
        σ0_ch ~ truncated(Normal(0, 2); lower = 0)            # spread of a works' first race
        # The states are sampled directly, with the walk as their prior (as the race model's
        # `_wst`, #102): in the step form each race's well-observed state was a sum of all
        # the steps before it, which held probe 108c at maximum tree depth.
        xs_cr ~ FlatVec(length(rwk.prev))
        δ_own ~ truncated(Normal(0, 1); lower = 0)            # larger off-season steps: ownership, regulations
        δ_reg ~ truncated(Normal(0, 1); lower = 0)
        # (as a prior term: a plain number would count towards the log likelihood, and so LOO)
        Turing.@addlogprob! (; logprior = walk_states_logpdf(xs_cr, race_walk_sd(τ_cr, τ_co, σ0_ch, δ_own, δ_reg, rwk), rwk.prev))
        xr = center_by(xs_cr, rwk.race, rwk.counts)
        # Each race's mean state is absorbed by its intercept, so only the walk prior sets it: a
        # 326-race random walk, too ill-conditioned for a diagonal metric (probe 108d, depth 10).
        # Pinned softly near zero (sd RACE_MEAN_PIN per state); the likelihood doesn't see it.
        Turing.@addlogprob! (; logprior = -sum(abs2, xs_cr .- xr) / (2 * RACE_MEAN_PIN^2))
        # No engine walk: beside a works' chassis walk, which takes each season's change, only
        # customer engines told it apart, and its scales went to zero (probe 108f, all chains).
        b = 0 .* a_π .+ zeros(length(d.machines))             # no car-season part: a tracked placeholder
    end
    if cw !== nothing       # chassis and engine walks (#108, as #95): b_mach is then each car-season's residual
        τ_ch ~ truncated(Normal(0, 1); lower = 0)             # step sd per year, %
        σ0_ch ~ truncated(Normal(0, 2); lower = 0)            # spread of a works' first season
        z_ch ~ StdNormalVec(length(cw.chassis.prev))
        τ_en ~ truncated(Normal(0, 1); lower = 0)
        σ0_en ~ truncated(Normal(0, 2); lower = 0)
        z_en ~ StdNormalVec(length(cw.engine.prev))
        δ_own ~ truncated(Normal(0, 1); lower = 0)            # larger steps: ownership, regulations
        δ_reg ~ truncated(Normal(0, 1); lower = 0)            # one for chassis and engine rules (#97)
        xc, xe = car_walk_states((; τ_ch, σ0_ch, z_ch, τ_en, σ0_en, z_en, δ_own, δ_reg), cw)
        b = b .+ walk_offsets(xc, cw.row_ch, xe, cw.row_en)
        if works            # the engine builder's own team (#100)
            β_works ~ Normal(0, 1)
            b = b .+ β_works .* cw.row_works
        end
    end
    if big                  # big incident losses (#108, as the race model's #16)
        a_ρ ~ Normal(log(0.05 / 0.95), 1)                       # share of incidents that are big, ~5%
        δ_λ2 ~ truncated(Normal(log(10), 1); lower = 0)         # λ2 = λ_baseline·e^δ > λ_baseline
        ρ, λ2 = logistic(a_ρ), exp(a_λ + δ_λ2)
    else                    # tracked zeros, so the likelihood has one form (ρ = 0: single loss)
        ρ, λ2 = 0 * a_π, 1 + 0 * a_π
    end
    # #113 terms (see lap_loglik); tracked placeholders when off, so the likelihood has one form
    nc = length(d.competitors)
    if ctx.traffic          # traffic bins, shared, in %
        η_tr ~ StdNormalVec(length(ctx.tnames))
        ηt = η_tr
    else
        ηt = 0 .* a_π .+ zeros(1)
    end
    if ctx.fpace            # car-season slopes on the field pace: the field moves 1:1, cars about it
        β_fp ~ Normal(1, 0.5)
        τ_fp ~ truncated(Normal(0, 0.5); lower = 0)
        u_fp ~ StdNormalVec(nm)
        φ = β_fp .+ τ_fp .* center_by(u_fp, d.mach_season, mach_counts)
        κ_fp ~ StdNormalVec(2)  # the field pace on the shortfall's logit π and log λ (used with `short`)
        κ = κ_fp
    else
        φ = 0 .* a_π .+ zeros(1)
        κ = 0 .* a_π .+ zeros(2)
    end
    if ctx.short            # shortfall offsets on logit π and log λ, drivers and car-seasons, centred
        # one vector in four blocks (drivers' π, cars' π, drivers' λ, cars' λ), each centred
        # and with its own scale, in one call (vcat of tracked arrays is ambiguous)
        τ_sh ~ filldist(truncated(Normal(0, 0.5); lower = 0), 4)
        u_sh ~ StdNormalVec(2nc + 2nm)
        bsh = short_blocks(nc, nm)
        sh = τ_sh[bsh] .* center_by(u_sh, bsh, [nc, nm, nc, nm])
    else
        sh = 0 .* a_π .+ zeros(1)
    end
    @addlogprob! lap_loglik(γ, δ, a, b, β, sl, dv, η, td, tm, fage, cx, xr, σ, πr, λr, πd, ρ, λ2, ηt, φ, sh, κ, rwk, ctx, d)
    if starts
        sd = sdrv ? career_slopes(τ_sd, u_sd) : 0 .* a_π .+ zeros(length(d.competitors))   # τ·(u - mean u)
        sc = τ_sc .* center_by(u_sc, d.mach_season, mach_counts)
        if dinc1            # lap-1 incident proneness, separate from the green-flag one (#108; off: barely identified)
            τ_dinc1 ~ truncated(Normal(0, 0.5); lower = 0)
            u_dinc1 ~ StdNormalVec(length(d.competitors))
            πd1 = τ_dinc1 .* (u_dinc1 .- mean(u_dinc1))
        else
            πd1 = zeros(length(d.competitors))
        end
        @addlogprob! start_loglik(γ1, a, b, sd, sc, g1, fage, cx, xr, σ1, logistic(a_π1), exp(a_λ1), πd1, rwk, d)
    end
end

"""
    CarWalks(d::LapData, cars = car_table())

Chassis and engine walk states for the lap model's car-seasons (#108): one row
per machine level of `d`, with its car-season's most common F1DB chassis and
engine (the lap model has no per-row pace scale, so the walks enter as
car-season offsets).
"""
function CarWalks(d::LapData, cars = car_table())
    seas = [d.seasons[s] for s in d.mach_season]
    rows = DataFrame(season = seas, round = fill(-1, length(seas)), competitor_id = fill("", length(seas)),
                     machine_id = [replace(m, Regex("_$(y)\$") => "") for (m, y) in zip(d.machines, seas)])
    return CarWalks(rows, first(d.seasons), last(d.seasons), cars)
end


function RaceWalk(d::LapData, cars = car_table())
    cw = CarWalks(d, cars)
    works = [first(rsplit(cw.chassis.names[cw.row_ch[m]], " "; limit = 2)) for m in eachindex(d.machines)]
    keys = sort(unique(vcat([(works[m], r) for (m, r) in zip(d.mach, d.race)],
                            [(works[m], r) for (m, r) in zip(d.s_mach, d.s_race)])))
    idx = Dict(k => i for (i, k) in enumerate(keys))
    own = Set{Tuple{String,Int}}((String(r.works_id), Int(r.season)) for r in eachrow(cars.ownership))
    regs = cars.regulations.season[cars.regulations.chassis .== 1]
    season(r) = d.seasons[d.race_season[r]]
    n = length(keys)
    prev, fst, bnd, ow, rg = zeros(Int, n), falses(n), zeros(n), zeros(n), zeros(n)
    for (i, (w, r)) in enumerate(keys)
        if i > 1 && keys[i - 1][1] == w
            prev[i] = i - 1
            y0, y = season(keys[i - 1][2]), season(r)
            if y != y0
                bnd[i] = 1.0
                ow[i] = (w, y) in own ? 1.0 : 0.0
                rg[i] = any(x -> y0 < x <= y, regs) ? 1.0 : 0.0
            end
        else
            fst[i] = true
        end
    end
    race = [r for (_, r) in keys]
    return RaceWalk(prev, fst, bnd, ow, rg, race, [count(==(r), race) for r in eachindex(d.races)],
                    ["$(w) $(d.races[r])" for (w, r) in keys],
                    [idx[(works[m], r)] for (m, r) in zip(d.mach, d.race)],
                    [idx[(works[m], r)] for (m, r) in zip(d.s_mach, d.s_race)])
end

# Step sd of each race-level chassis state: a works' first race σ0; within a season
# τ_cr per race; across an off-season τ_co·exp(δ_own·own + δ_reg·reg)
const RACE_MEAN_PIN = 0.02

# Log density of walk states x whose state k is N(x[prev[k]], sd[k]) (N(0, sd[k]) at a walk's start).
walk_states_logpdf(x, sd, prev::Vector{Int}) = sum(normlogpdf.(walk_diffs(x, prev) ./ sd)) - sum(log.(sd))

function race_walk_sd(τ_cr, τ_co, σ0, δ_own, δ_reg, rw::RaceWalk)
    f = Float64.(rw.first)
    return f .* σ0 .+ (1 .- f) .* (rw.boundary .* τ_co .* exp.(δ_own .* rw.own .+ δ_reg .* rw.reg) .+ (1 .- rw.boundary) .* τ_cr)
end

lap_mach_counts(d::LapData) = [count(==(k), d.mach_season) for k in eachindex(d.seasons)]
function lap_effects(d::LapData; tyres::Bool = true, slopes::Bool = false, dev::Bool = false, lead::Bool = false,
                     tyre_mgmt::Bool = false, starts::Bool = false, age::Bool = false, circuits::Bool = false,
                     track::Bool = false, dinc::Bool = false, walk::Bool = false, works::Bool = false,
                     big::Bool = false, rwalk::Bool = false, sdrv::Bool = true, dinc1::Bool = false,
                     cars = walk || rwalk ? car_table() : nothing, ctx = LapContext())
    (ctx.traffic || ctx.fpace) && size(ctx.tbin, 1) != length(d.y) &&
        throw(ArgumentError("ctx was not built from these laps (LapContext(d, laps; …))"))
    age && length(d.age_years) < 3 && throw(ArgumentError("the age curve needs ages spanning at least 3 years"))
    circuits && size(d.race_x, 2) == 0 && throw(ArgumentError("circuits = true needs prepare_laps(…; circuits)"))
    track && size(d.race_track, 2) == 0 && throw(ArgumentError("track = true needs prepare_laps(…; circuits)"))
    works && !walk && throw(ArgumentError("works needs the chassis and engine walks (walk = true)"))
    rwalk && (walk || dev) && throw(ArgumentError("rwalk replaces the season walks and the development trends (walk, dev)"))
    dinc1 && !(dinc && starts) && throw(ArgumentError("dinc1 needs dinc and starts"))
    basis = age ? AgeCurveBasis(d.age_years, d.age_rows) : nothing
    return lap_effects(d, lap_mach_counts(d), basis, tyres, slopes, dev, lead, tyre_mgmt, starts, age, circuits, track, dinc,
                       walk ? CarWalks(d, cars) : nothing, works, big,
                       rwalk ? (; rwk = RaceWalk(d, cars), cw = CarWalks(d, cars)) : nothing, sdrv, dinc1, ctx)
end

"""
    fit_laps(d::LapData; n_samples=1000, n_chains=1, checkpoint=nothing, seed=1, ...)

Sample `lap_effects`, starting near the prior centre with race intercepts and
slopes from a per-race least-squares line (jittered per chain). Options as in
`fit_paceloss`, including `checkpoint` (resumable blocks, one chain), and
`tyres`, `slopes`, `dev`, `lead`, `tyre_mgmt`, `starts`, `age` and `circuits` as in `lap_effects`. With
`warm_start = lap_warm_start(...)` (checkpointed fits), NUTS starts from that
metric and step size (`lap_warm_sampler`) and adapts for `n_adapts` iterations.
`δ` is NUTS's target acceptance rate (default 0.8). Lowering it a little enlarges
the adapted step size; on large fits that can keep every chain at the same tree
depth when some would otherwise adapt just past a doubling (#62).
"""
function fit_laps(d::LapData; n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                  sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                  progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100,
                  checkpoint::Union{Nothing,AbstractString} = nothing, seed::Int = 1,
                  warm_start = nothing, n_adapts::Int = 500, tyres::Bool = true, slopes::Bool = false,
                  dev::Bool = false, lead::Bool = false, tyre_mgmt::Bool = false, starts::Bool = false,
                  age::Bool = false, circuits::Bool = false, track::Bool = false, dinc::Bool = false,
                  walk::Bool = false, works::Bool = false, big::Bool = false, rwalk::Bool = false, sdrv::Bool = true,
                  dinc1::Bool = false, cars = walk || rwalk ? car_table() : nothing, ctx = LapContext(), δ::Real = 0.8,
                  kwargs...)
    checkpoint === nothing || n_chains == 1 || throw(ArgumentError("checkpointed fits run one chain per call"))
    warm_start === nothing || checkpoint !== nothing ||
        throw(ArgumentError("warm_start is implemented for checkpointed fits"))
    model = lap_effects(d; tyres, slopes, dev, lead, tyre_mgmt, starts, age, circuits, track, dinc, walk, works, big, rwalk,
                        sdrv, dinc1, cars, ctx)
    cw = model.args.cw
    rwm = model.args.rw
    # per-race line through the laps, as starting values
    γ0, δ0 = zeros(length(d.races)), zeros(length(d.races))
    for r in eachindex(d.races)
        sel = d.race .== r
        f, y = d.fracc[sel], d.y[sel]
        δ0[r] = var(f) > 0 ? cov(f, y) / var(f) : 0.0
        γ0[r] = mean(y)
    end
    # stint terms and pace noise from the residuals of those lines (a large dataset
    # concentrates the posterior: generic starting values put NUTS's initial step
    # size at ~1e-4 and every iteration at maximum tree depth)
    res0 = d.y .- γ0[d.race] .- δ0[d.race] .* d.fracc
    sv = d.stint ./ 10
    X = tyres ? reduce(hcat, [hcat(sv .* (d.era .== e), sv .^ 2 .* (d.era .== e)) for e in eachindex(d.eras)]) :
        zeros(length(sv), 0)
    β0 = tyres ? (X' * X) \ (X' * res0) : Float64[]
    e0 = res0 .- X * β0
    σ0 = 1.4826 * median(abs.(e0 .- median(e0)))
    init_all() = merge((; σ_comp = 0.3 + 0.4 * rand(rng), σ_mach = 0.3 + 0.4 * rand(rng),
                             σ = σ0 * (0.9 + 0.2 * rand(rng)),
                             a_comp = 0.05 .* randn(rng, length(d.competitors)),
                             b_mach = 0.05 .* randn(rng, length(d.machines)),
                             μ_γ = mean(γ0), γ = γ0 .+ 0.05 .* randn(rng, length(γ0)),
                             μ_δ = mean(δ0), τ_δ = std(δ0) * (0.9 + 0.2 * rand(rng)),
                             δ = δ0 .+ 0.05 .* randn(rng, length(δ0)),
                             a_π = -1.5 + 0.1 * randn(rng), a_λ = 0.1 * randn(rng)),
                            tyres ? (; β = β0 .+ 0.01 .* randn(rng, length(β0))) : (;),
                            slopes ? (; τ_slope = 0.1 + 0.1 * rand(rng),
                                        u_slope = 0.1 .* randn(rng, length(d.competitors))) : (;),
                            dev ? (; τ_dev = 0.1 + 0.1 * rand(rng), u_dev = 0.1 .* randn(rng, length(d.machines))) : (;),
                            lead ? (; η = 0.01 .* randn(rng, 1)) : (;),
                            tyre_mgmt ? (; τ_td = 0.05 + 0.05 * rand(rng), u_td = 0.1 .* randn(rng, length(d.competitors)),
                                           τ_tm = 0.05 + 0.05 * rand(rng), u_tm = 0.1 .* randn(rng, length(d.machines))) : (;),
                            starts ? (; γ1 = 0.05 .* randn(rng, length(d.races)), τ_sd = 0.1 + 0.1 * rand(rng),
                                        u_sd = 0.1 .* randn(rng, length(d.competitors)), τ_sc = 0.1 + 0.1 * rand(rng),
                                        u_sc = 0.1 .* randn(rng, length(d.machines)), g1 = 0.1 .* randn(rng, 2),
                                        σ1 = (isempty(d.s_y) ? 1.0 : 1.4826 * median(abs.(d.s_y .- median(d.s_y)))) * (0.9 + 0.2 * rand(rng)),
                                        a_π1 = -1.5 + 0.1 * randn(rng), a_λ1 = 0.1 * randn(rng)) : (;),
                            age ? (; τ_age = 0.02 + 0.02 * rand(rng), e_age = 0.1 .* randn(rng, length(d.age_years) - 1)) : (;),
                            circuits ? (; τ_cx = 0.05 .+ 0.05 .* rand(rng, size(d.race_x, 2)),
                                          u_cx = 0.1 .* randn(rng, length(d.machines) * size(d.race_x, 2))) : (;),
                            track ? (; β_track_π = 0.1 .* randn(rng, size(d.race_track, 2)),
                                       β_track_λ = 0.1 .* randn(rng, size(d.race_track, 2))) : (;),
                            dinc ? (; τ_dinc = 0.05 + 0.05 * rand(rng), u_dinc = 0.1 .* randn(rng, length(d.competitors))) : (;),
                            walk ? (; τ_ch = 0.1 + 0.1 * rand(rng), σ0_ch = 0.3 + 0.4 * rand(rng),
                                      z_ch = 0.1 .* randn(rng, length(cw.chassis.prev)),
                                      τ_en = 0.05 + 0.05 * rand(rng), σ0_en = 0.1 + 0.2 * rand(rng),
                                      z_en = 0.1 .* randn(rng, length(cw.engine.prev)),
                                      δ_own = 0.1 + 0.1 * rand(rng), δ_reg = 0.1 + 0.1 * rand(rng)) : (;),
                            works ? (; β_works = 0.1 * randn(rng)) : (;),
                            big ? (; a_ρ = log(0.05 / 0.95) + 0.1 * randn(rng), δ_λ2 = log(10) + 0.1 * randn(rng)) : (;),
                            dinc1 ? (; τ_dinc1 = 0.05 + 0.05 * rand(rng),
                                       u_dinc1 = 0.1 .* randn(rng, length(d.competitors))) : (;),
                            rwalk ? (; τ_cr = 0.05 + 0.05 * rand(rng), τ_co = 0.2 + 0.2 * rand(rng),
                                       σ0_ch = 0.3 + 0.4 * rand(rng), xs_cr = 0.1 .* randn(rng, length(rwm.rwk.prev)),
                                       δ_own = 0.1 + 0.1 * rand(rng), δ_reg = 0.1 + 0.1 * rand(rng)) : (;),
                            ctx.traffic ? (; η_tr = 0.05 .* randn(rng, length(ctx.tnames))) : (;),
                            ctx.fpace ? (; β_fp = 1 + 0.05 * randn(rng), τ_fp = 0.05 + 0.05 * rand(rng),
                                           u_fp = 0.1 .* randn(rng, length(d.machines)), κ_fp = 0.05 .* randn(rng, 2)) : (;),
                            ctx.short ? (; τ_sh = 0.05 .+ 0.05 .* rand(rng, 4),
                                           u_sh = 0.1 .* randn(rng, 2length(d.competitors) + 2length(d.machines))) : (;))
    # leave out what the model doesn't sample (no residual with the race walk, no driver start effects)
    drop = (rwalk ? (:σ_mach, :b_mach) : ())..., (sdrv ? () : (:τ_sd, :u_sd))...
    init() = InitFromParams(NamedTuple(k => v for (k, v) in pairs(init_all()) if k ∉ drop))
    external = warm_start === nothing ? nothing : lap_warm_sampler(warm_start.Minv, warm_start.ϵ; δ)
    checkpoint === nothing ||
        return run_nuts_checkpointed(model, init(), checkpoint; n_samples, n_adapts, seed, progress_log, log_every,
                                     external, δ, kwargs...)
    return run_nuts(model, [init() for _ in 1:n_chains]; n_samples, n_chains, ensemble, sampler, rng,
                    progress, progress_log, log_every, kwargs...)
end

# Warm start (#12): the 1996–2019 fit spends most of its warm-up at maximum tree
# depth before the first mass-matrix window (~80 s per iteration on 470k laps).
# Starting from the step size and diagonal metric adapted on a smaller fit
# (2011–2019) skips that phase.

const START_PARAMS = (:γ1, :τ_sd, :u_sd, :τ_sc, :u_sc, :g1, :σ1, :a_π1, :a_λ1)
const AGE_PARAMS = (:τ_age, :e_age)
const CIRCUIT_PARAMS = (:τ_cx, :u_cx)
const TRACK_PARAMS = (:β_track_π, :β_track_λ)
const DINC_PARAMS = (:τ_dinc, :u_dinc)
const WALK_PARAMS = (:τ_ch, :σ0_ch, :z_ch, :τ_en, :σ0_en, :z_en, :δ_own, :δ_reg)
const FPACE_PARAMS = (:β_fp, :τ_fp, :u_fp, :κ_fp)
const SHORT_PARAMS = (:τ_sh, :u_sh)
# block of each shortfall offset (#113): drivers' logit π, car-seasons' logit π, drivers' log λ, car-seasons' log λ
short_blocks(nc, nm) = vcat(fill(1, nc), fill(2, nm), fill(3, nc), fill(4, nm))

"Parameter layout of `lap_effects` in the sampler's (unconstrained) vector: name => (length, level labels or nothing)."
lap_layout(d::LapData; tyres::Bool = true, slopes::Bool = false, dev::Bool = false, lead::Bool = false,
           tyre_mgmt::Bool = false, starts::Bool = false, age::Bool = false, circuits::Bool = false,
           track::Bool = false, dinc::Bool = false, walk::Bool = false, works::Bool = false, big::Bool = false,
           rwalk::Bool = false, sdrv::Bool = true, dinc1::Bool = false,
           cw = walk || rwalk ? CarWalks(d) : nothing, rwk = rwalk ? RaceWalk(d) : nothing, ctx = LapContext()) =
    filter(p -> (tyres || first(p) !== :β) && (slopes || first(p) ∉ (:τ_slope, :u_slope)) &&
                (dev || first(p) ∉ (:τ_dev, :u_dev)) && (lead || first(p) !== :η) &&
                (tyre_mgmt || first(p) ∉ (:τ_td, :u_td, :τ_tm, :u_tm)) &&
                (starts || first(p) ∉ START_PARAMS) && (age || first(p) ∉ AGE_PARAMS) &&
                (circuits || first(p) ∉ CIRCUIT_PARAMS) && (track || first(p) ∉ TRACK_PARAMS) &&
                (dinc || first(p) ∉ DINC_PARAMS) && (works || first(p) !== :β_works) &&
                (walk || first(p) ∉ (:τ_ch, :z_ch)) && (rwalk || first(p) ∉ (:τ_cr, :τ_co, :xs_cr)) &&
                (walk || rwalk || first(p) ∉ WALK_PARAMS) && (!rwalk || first(p) ∉ (:σ_mach, :b_mach, :τ_en, :σ0_en, :z_en)) &&
                (sdrv || first(p) ∉ (:τ_sd, :u_sd)) &&
                (dinc1 || first(p) ∉ (:τ_dinc1, :u_dinc1)) && (big || first(p) ∉ (:a_ρ, :δ_λ2)) &&
                (ctx.traffic || first(p) !== :η_tr) && (ctx.fpace || first(p) ∉ FPACE_PARAMS) &&
                (ctx.short || first(p) ∉ SHORT_PARAMS),
                         [:σ_comp => (1, nothing), :σ_mach => (1, nothing), :σ => (1, nothing),
                          :a_comp => (length(d.competitors), d.competitors), :b_mach => (length(d.machines), d.machines),
                          :μ_γ => (1, nothing), :γ => (length(d.races), d.races), :μ_δ => (1, nothing),
                          :τ_δ => (1, nothing), :δ => (length(d.races), d.races),
                          :β => (2length(d.eras), [string(e, "_", t) for e in d.eras for t in ("s", "s2")]),
                          :τ_slope => (1, nothing), :u_slope => (length(d.competitors), d.competitors),
                          :τ_dev => (1, nothing), :u_dev => (length(d.machines), d.machines),
                          :η => (1, nothing),
                          :τ_td => (1, nothing), :u_td => (length(d.competitors), d.competitors),
                          :τ_tm => (1, nothing), :u_tm => (length(d.machines), d.machines),
                          :γ1 => (length(d.races), d.races),
                          :τ_sd => (1, nothing), :u_sd => (length(d.competitors), d.competitors),
                          :τ_sc => (1, nothing), :u_sc => (length(d.machines), d.machines),
                          :g1 => (2, nothing), :σ1 => (1, nothing), :a_π1 => (1, nothing), :a_λ1 => (1, nothing),
                          # age-curve steps are labelled by the bin they lead to, circuit slopes by
                          # covariate and car-season, so warm starts map them across fits
                          :τ_age => (1, nothing), :e_age => (max(length(d.age_years) - 1, 0), string.(d.age_years[2:end])),
                          :τ_cx => (size(d.race_x, 2), string.(d.race_x_names)),
                          :u_cx => (length(d.machines) * size(d.race_x, 2),
                                    [string(k, "|", m) for k in d.race_x_names for m in d.machines]),
                          :a_π => (1, nothing), :a_λ => (1, nothing),
                          :β_track_π => (size(d.race_track, 2), string.(d.race_track_names)),
                          :β_track_λ => (size(d.race_track, 2), string.(d.race_track_names)),
                          :τ_dinc => (1, nothing), :u_dinc => (length(d.competitors), d.competitors),
                          # walk states are labelled "works season", so warm starts map them across fits
                          :τ_cr => (1, nothing), :τ_co => (1, nothing),
                          :τ_ch => (1, nothing), :σ0_ch => (1, nothing),
                          :xs_cr => rwk === nothing ? (0, nothing) : (length(rwk.prev), rwk.names),
                          :z_ch => cw === nothing ? (0, nothing) : (length(cw.chassis.prev), cw.chassis.names),
                          :τ_en => (1, nothing), :σ0_en => (1, nothing),
                          :z_en => cw === nothing ? (0, nothing) : (length(cw.engine.prev), cw.engine.names),
                          :δ_own => (1, nothing), :δ_reg => (1, nothing), :β_works => (1, nothing),
                          :a_ρ => (1, nothing), :δ_λ2 => (1, nothing),
                          # #113 terms, sampled after the big losses and before the lap-1 incident offsets
                          :η_tr => (length(ctx.tnames), ctx.tnames),
                          :β_fp => (1, nothing), :τ_fp => (1, nothing), :u_fp => (length(d.machines), d.machines),
                          :κ_fp => (2, ["π", "λ"]),
                          :τ_sh => (4, ["π_drv", "π_car", "λ_drv", "λ_car"]),
                          :u_sh => (2length(d.competitors) + 2length(d.machines),
                                    vcat("π|" .* d.competitors, "π|" .* d.machines, "λ|" .* d.competitors, "λ|" .* d.machines)),
                          :τ_dinc1 => (1, nothing), :u_dinc1 => (length(d.competitors), d.competitors)])

# Inverse-metric entries for blocks a warm start's old fit doesn't have (career
# slopes or car development added to a fit without them), on the sampler's scale
# (log τ; standardised u). Kept small: a variance set too large forces a small
# global step size (the first slopes fit warmed up at ~197 s per iteration with
# u_slope at 0.5, #12), while one too small only slows that direction until
# adaptation corrects it.
const LAP_WARM_DEFAULT = Dict(:τ_slope => 0.01, :u_slope => 0.01, :τ_dev => 0.01, :u_dev => 0.01, :β => 1e-4,
                              :η => 1e-4, :τ_td => 0.01, :u_td => 0.01, :τ_tm => 0.01, :u_tm => 0.01,
                              :γ1 => 0.01, :τ_sd => 0.01, :u_sd => 0.01, :τ_sc => 0.01, :u_sc => 0.01, :g1 => 1e-3,
                              :σ1 => 1e-3, :a_π1 => 0.01, :a_λ1 => 0.01, :τ_age => 0.01, :e_age => 0.01,
                              :τ_cx => 0.01, :u_cx => 0.01, :β_track_π => 0.01, :β_track_λ => 0.01,
                              :τ_dinc => 0.01, :u_dinc => 0.01, :τ_ch => 0.01, :σ0_ch => 0.01, :z_ch => 0.01,
                              :τ_en => 0.01, :σ0_en => 0.01, :z_en => 0.01, :δ_own => 0.01, :δ_reg => 0.01,
                              :β_works => 1e-3, :τ_dinc1 => 0.01, :u_dinc1 => 0.01, :a_ρ => 0.01, :δ_λ2 => 0.01,
                              :τ_cr => 0.01, :τ_co => 0.01, :xs_cr => 0.01, :η_tr => 1e-4, :β_fp => 1e-4,
                              :τ_fp => 0.01, :u_fp => 0.01, :κ_fp => 1e-3, :τ_sh => 0.01, :u_sh => 0.01)

"""
    lap_warm_start(state, d_old::LapData, d_new::LapData; tyres_old = true, tyres_new = tyres_old,
                   slopes_old = false, slopes_new = slopes_old, dev_old = false, dev_new = dev_old,
                   lead_old = false, lead_new = lead_old, tyre_mgmt_old = false, tyre_mgmt_new = tyre_mgmt_old,
                   starts_old = false, starts_new = starts_old, age_old = false, age_new = age_old,
                   circuits_old = false, circuits_new = circuits_old, track_old = false, track_new = track_old,
                   dinc_old = false, dinc_new = dinc_old, walk_old = false, walk_new = walk_old,
                   works_old = false, works_new = works_old, big_old = false, big_new = big_old, rwalk_old = false,
                   rwalk_new = rwalk_old, sdrv_old = true, sdrv_new = sdrv_old, dinc1_old = false,
                   dinc1_new = dinc1_old, ctx_old = LapContext(), ctx_new = ctx_old, missing_old = ()) -> (; Minv, ϵ)

Diagonal inverse metric and step size for a fit to `d_new`, from the adapted
NUTS `state` of a fit to `d_old` (`only(Turing.loadstate(chain))`, or its
`lap_adapted` summary). Levels
(drivers, car-seasons, races) present in both keep their adapted variance; new
levels get their group's median. Scalars are scaled by the ratio of lap
counts (posterior variance ∝ 1/data). Adaptation still runs on the new fit;
this only gives it a good start. `tyres_*` and `slopes_*` say whether each fit
has the tyre terms and career slopes (`lap_effects`); a block the old fit
doesn't have starts from `LAP_WARM_DEFAULT`; `missing_old` names blocks its flags would
give it but it predates (e.g. `(:κ_fp,)` for a #113 fit from before that term).
"""
function lap_warm_start(state, d_old::LapData, d_new::LapData; tyres_old::Bool = true, tyres_new::Bool = tyres_old,
                        slopes_old::Bool = false, slopes_new::Bool = slopes_old,
                        dev_old::Bool = false, dev_new::Bool = dev_old, lead_old::Bool = false,
                        lead_new::Bool = lead_old, tyre_mgmt_old::Bool = false, tyre_mgmt_new::Bool = tyre_mgmt_old,
                        starts_old::Bool = false, starts_new::Bool = starts_old, age_old::Bool = false,
                        age_new::Bool = age_old, circuits_old::Bool = false, circuits_new::Bool = circuits_old,
                        track_old::Bool = false, track_new::Bool = track_old, dinc_old::Bool = false,
                        dinc_new::Bool = dinc_old, walk_old::Bool = false, walk_new::Bool = walk_old,
                        works_old::Bool = false, works_new::Bool = works_old, big_old::Bool = false,
                        big_new::Bool = big_old, rwalk_old::Bool = false, rwalk_new::Bool = rwalk_old,
                        sdrv_old::Bool = true, sdrv_new::Bool = sdrv_old, dinc1_old::Bool = false,
                        dinc1_new::Bool = dinc1_old, ctx_old = LapContext(), ctx_new = ctx_old, missing_old = ())
    ad = lap_adapted(state)
    Mo = ad.Minv
    lo = lap_layout(d_old; tyres = tyres_old, slopes = slopes_old, dev = dev_old, lead = lead_old, tyre_mgmt = tyre_mgmt_old,
                    starts = starts_old, age = age_old, circuits = circuits_old, track = track_old, dinc = dinc_old,
                    walk = walk_old, works = works_old, big = big_old, rwalk = rwalk_old, sdrv = sdrv_old, dinc1 = dinc1_old,
                    ctx = ctx_old)
    lo = filter(p -> first(p) ∉ missing_old, lo)
    ln = lap_layout(d_new; tyres = tyres_new, slopes = slopes_new, dev = dev_new, lead = lead_new, tyre_mgmt = tyre_mgmt_new,
                    starts = starts_new, age = age_new, circuits = circuits_new, track = track_new, dinc = dinc_new,
                    walk = walk_new, works = works_new, big = big_new, rwalk = rwalk_new, sdrv = sdrv_new, dinc1 = dinc1_new,
                    ctx = ctx_new)
    sum(first(last(p)) for p in lo) == length(Mo) || throw(DimensionMismatch("state does not match d_old"))
    scale = length(d_old.y) / length(d_new.y)
    old = Dict{Symbol,Tuple{UnitRange{Int},Any}}()
    io = 0
    for (name, (no, labels_old)) in lo
        old[name] = ((io + 1):(io + no), labels_old)
        io += no
    end
    Mn = Float64[]
    for (name, (nn, labels_new)) in ln
        if !haskey(old, name)
            append!(Mn, fill(LAP_WARM_DEFAULT[name], nn))
            continue
        end
        rng, labels_old = old[name]
        block = Mo[rng]
        if labels_old === nothing
            append!(Mn, block .* scale)
        else
            pos = Dict(l => k for (k, l) in enumerate(labels_old))
            med = median(block)
            append!(Mn, [haskey(pos, l) ? block[pos[l]] : med for l in labels_new])
        end
    end
    return (; Minv = Mn, ϵ = ad.ϵ * sqrt(scale))
end

"""
    lap_adapted(state) -> (; Minv, ϵ)

The adapted diagonal inverse metric and step size in a sampler state: Turing's
NUTS state, the `TuringState` wrapping an AdvancedHMC sampler (a fit that was
itself warm-started, see `lap_warm_sampler`), or an already extracted
`(; Minv, ϵ)`.
"""
function lap_adapted(st)
    st isa NamedTuple && haskey(st, :Minv) && return (; Minv = Vector{Float64}(st.Minv), ϵ = Float64(st.ϵ))
    s = hasproperty(st, :hamiltonian) || hasproperty(st, :metric) ? st : st.state
    hasproperty(s, :hamiltonian) &&
        return (; Minv = Vector{Float64}(s.hamiltonian.metric.M⁻¹), ϵ = Float64(s.kernel.τ.integrator.ϵ))
    return (; Minv = Vector{Float64}(s.metric.M⁻¹), ϵ = Float64(s.κ.τ.integrator.ϵ))     # AdvancedHMC.HMCState
end

"""
    lap_warm_sampler(Minv, ϵ; δ = 0.8)

NUTS (AdvancedHMC, via `externalsampler`, uncompiled ReverseDiff) starting from
diagonal inverse metric `Minv` and step size `ϵ`, with Stan's windowed
adaptation of both (as Turing's `NUTS`).
"""
function lap_warm_sampler(Minv::AbstractVector, ϵ::Real; δ::Real = 0.8)
    metric = AdvancedHMC.DiagEuclideanMetric(Vector{Float64}(Minv))
    integrator = AdvancedHMC.Leapfrog(Float64(ϵ))
    κ = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator,
                                                                               AdvancedHMC.GeneralisedNoUTurn()))
    adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(δ, integrator))
    return externalsampler(AdvancedHMC.HMCSampler(κ, metric, adaptor); adtype = gap_sampler().adtype)
end
