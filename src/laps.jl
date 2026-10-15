# Lap-time model (#12): per-lap pace instead of whole-race times.
#
# Outcome: y = 100·log(t / m_race), a lap time in % of the race's median clean
# lap, on the same % scale as the race-time models. Laps kept are "clean" green-
# flag racing laps. Excluded:
#   - lap 1 (standing start),
#   - race-wide slow laps (the field's median lap > SLOW_RACE × the race median:
#     safety car, VSC, red flag) and the lap after each (the restart),
#   - a car's laps over STOPPAGE × its own median lap (a red-flag stoppage is in
#     one lap per car, at different lap numbers; also very long stops),
#   - in- and out-laps of pit stops, recorded (Ergast pit stops, 2011 on) or
#     inferred from lap times (see `infer_pit_stops`).
# Other slow laps (traffic, mistakes, damage) stay in: the pace + loss noise
# treats them as incident losses, as in the race-time model.
#
# Covariates: the lap's fraction of the race distance (fuel burn, track
# evolution; a slope per race), the tyres' stint age in laps since the last
# stop (`stint`, 1 on the first lap after the out-lap), the driver's age in the
# season minus their mean age over their laps (`agec`, decades; for career
# slopes, as in the race-time model), and the race's position in its season
# minus its car's mean position (`posc`; for in-season car development, #14),
# and whether the car led the race on that lap (`lead`, from its position at the
# end of the lap; #66).

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
(1 if the car was leading at the end of the lap).
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
    mach_season::Vector{Int}      # season index of each machine level
    race_season::Vector{Int}      # season index of each race
    seasons::Vector{Int}
    competitors::Vector{String}
    machines::Vector{String}
    races::Vector{String}
    race_n::Vector{Int}           # laps per race
    rows::DataFrame               # laps used
    excluded::Dict{Symbol,Int}    # laps dropped, by reason
end

function Base.show(io::IO, d::LapData)
    print(io, "LapData($(length(d.y)) laps, $(length(d.competitors)) drivers, $(length(d.machines)) ",
          "machine-seasons, $(length(d.races)) races; excluded ",
          join(("$k $v" for (k, v) in sort(collect(d.excluded))), ", "), ")")
end

"""
    prepare_laps(laps, results; pits = :recorded, pit_stops = nothing, machine_key) -> LapData

Clean laps (see the file header) from `laps` (`LAP_SCHEMA`) with each car's
machine from `results` (`RESULT_SCHEMA`). `pits = :recorded` takes in-laps from
`pit_stops` (`PIT_SCHEMA`); `pits = :inferred` infers them from the lap times
(`infer_pit_stops`). Races without lap data for their winner are skipped.
"""
function prepare_laps(laps::AbstractDataFrame, results::AbstractDataFrame; pits::Symbol = :recorded,
                      pit_stops = nothing, machine_key = r -> string(r.machine_id, "_", r.season))
    pits in (:recorded, :inferred) || throw(ArgumentError("pits must be :recorded or :inferred"))
    pits === :recorded && pit_stops === nothing && throw(ArgumentError("pits = :recorded needs pit_stops"))
    mkey = Dict((r.event_id, r.competitor_id) => machine_key(r) for r in eachrow(results))
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
            for l in eachindex(times)
                reason = l == 1 ? :lap1 : l in skip ? :slow_race : times[l] > STOPPAGE * med ? :stoppage :
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
    return LapData(keep.y, comp, mach,
                   race, keep.frac, keep.frac .- fbar[race], Float64.(keep.stint), agec, posc, keep.lead, stint_cd, stint_cm,
                   mach_season, race_season, seasons, comps, machs, races, race_n, keep, excl)
end

"""
    lap_loglik(γ, δ, a, b, β, sl, dv, η, td, tm, σ, π, λ, d::LapData)

Log-likelihood of the laps in `d` under pace + loss noise (`paceloss_logpdf`),
with per-lap mean

    μ = γ[race] + δ[race]·(frac - mean frac of the race) + h - (mean h of the race),
    h = a[driver] + sl[driver]·agec + b[machine] + dv[machine]·posc + η[1]·lead
        + β[1]·s + β[2]·s² + td[driver]·s_d + tm[machine]·s_m,   s = stint/10

(stint age in tens of laps, which keeps β's two terms on similar scales; `sl`
are the drivers' career slopes in % per decade of age and `dv` the cars'
in-season development trends in % per season; η the lead-lap effect in %; `td`
and `tm` the drivers' and cars' deviations from the shared tyre slope, in % per
10 laps of stint age, on s_d and s_m: s centred on the driver's and on the
car-season's mean; see `lap_effects`). As in
the race-time model, centring within each race makes γ the race's mean lap, so
the intercepts are not correlated with the slopes and effects: with ~1,000 laps
per race the uncentred form had near-degenerate directions that kept NUTS at
its maximum tree depth. Has a fused reverse rule: per-lap derivatives from
ForwardDiff duals, scattered into the parameter vectors.
"""
function lap_loglik(γ, δ, a, b, β, sl, dv, η, td, tm, σ, π, λ, d::LapData)
    h, hbar = lap_h(a, b, β, sl, dv, η, td, tm, d)
    s = 0.0
    for j in eachindex(d.y)
        r = d.race[j]
        μ = γ[r] + δ[r] * d.fracc[j] + h[j] - hbar[r]
        s += paceloss_logpdf(d.y[j] - μ, σ, π, λ)
    end
    return s
end

# Per-lap h = driver + career slope + car + car development + lead lap + tyre terms, and its mean per race.
function lap_h(a, b, β, sl, dv, η, td, tm, d::LapData)
    T = promote_type(eltype(a), eltype(b), eltype(β), eltype(sl), eltype(dv), eltype(η), eltype(td), eltype(tm))
    h = Vector{T}(undef, length(d.y))
    hbar = zeros(T, length(d.races))
    for j in eachindex(d.y)
        st = d.stint[j] / 10
        h[j] = a[d.comp[j]] + sl[d.comp[j]] * d.agec[j] + b[d.mach[j]] + dv[d.mach[j]] * d.posc[j] +
               η[1] * d.lead[j] + β[1] * st + β[2] * st^2 + td[d.comp[j]] * d.stint_cd[j] + tm[d.mach[j]] * d.stint_cm[j]
        hbar[d.race[j]] += h[j]
    end
    hbar ./= d.race_n
    return h, hbar
end

function ChainRulesCore.rrule(::typeof(lap_loglik), γ, δ, a, b, β, sl, dv, η, td, tm, σ, π, λ, d::LapData)
    h, hbar = lap_h(a, b, β, sl, dv, η, td, tm, d)
    N = length(d.y)
    gμ = zeros(N)
    gγ, gδ, ga, gb, gβ = zeros(length(γ)), zeros(length(δ)), zeros(length(a)), zeros(length(b)), zeros(2)
    gsl, gdv = zeros(length(sl)), zeros(length(dv))
    gη, gtd, gtm = zeros(length(η)), zeros(length(td)), zeros(length(tm))
    gσ, gπ, gλ, val = 0.0, 0.0, 0.0, 0.0
    for j in 1:N
        r, y = d.race[j], d.y[j]
        μ = γ[r] + δ[r] * d.fracc[j] + h[j] - hbar[r]
        v, p = value_grad4((m, s, q, l) -> paceloss_logpdf(y - m, s, q, l), μ, σ, π, λ)
        val += v
        gμ[j] = p[1]
        gγ[r] += p[1]; gδ[r] += p[1] * d.fracc[j]
        gσ += p[2]; gπ += p[3]; gλ += p[4]
    end
    # h enters every lap of its race through the race mean: dμ_k/dh_j = [k = j] - 1/n_race
    for j in 1:N
        gh = gμ[j] - gγ[d.race[j]] / d.race_n[d.race[j]]
        st = d.stint[j] / 10
        ga[d.comp[j]] += gh; gb[d.mach[j]] += gh
        gsl[d.comp[j]] += gh * d.agec[j]
        gdv[d.mach[j]] += gh * d.posc[j]
        gβ[1] += gh * st; gβ[2] += gh * st^2
        gη[1] += gh * d.lead[j]
        gtd[d.comp[j]] += gh * d.stint_cd[j]; gtm[d.mach[j]] += gh * d.stint_cm[j]
    end
    pullback(Δ) = (NoTangent(), Δ .* gγ, Δ .* gδ, Δ .* ga, Δ .* gb, Δ .* gβ, Δ .* gsl, Δ .* gdv, Δ .* gη, Δ .* gtd, Δ .* gtm,
                   Δ * gσ, Δ * gπ, Δ * gλ, NoTangent())
    return val, pullback
end
# β (tyres), sl (career slopes), dv (car development), η (lead laps) and td, tm
# (tyre management) are tracked when sampled and constant zero vectors when the
# model leaves them out (`lap_effects(d; tyres, slopes, dev, lead, tyre_mgmt)`)
const _TA_OR_CONST = (ReverseDiff.TrackedArray, Vector{Float64})
for Tβ in _TA_OR_CONST, Tsl in _TA_OR_CONST, Tdv in _TA_OR_CONST, Tη in _TA_OR_CONST, Tt in _TA_OR_CONST
    @eval ReverseDiff.@grad_from_chainrules lap_loglik(γ::ReverseDiff.TrackedArray, δ::ReverseDiff.TrackedArray,
                                                       a::ReverseDiff.TrackedArray, b::ReverseDiff.TrackedArray,
                                                       β::$Tβ, sl::$Tsl, dv::$Tdv, η::$Tη, td::$Tt, tm::$Tt,
                                                       σ::ReverseDiff.TrackedReal,
                                                       π::ReverseDiff.TrackedReal, λ::ReverseDiff.TrackedReal, d::LapData)
end

"""
    lap_effects(d::LapData; tyres = true, slopes = false, dev = false, lead = false, tyre_mgmt = false)

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
- Tyres: β[1]·s + β[2]·s² with s = stint age / 10 laps, in %. With
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
- Noise: pace + loss (`paceloss_logpdf`) with one σ, incident probability π
  and mean loss λ: traffic, mistakes and damage are losses, as in the race model.
"""
@model function lap_effects(d::LapData, mach_counts::Vector{Int}, tyres::Bool, slopes::Bool, dev::Bool,
                            lead::Bool, tyre_mgmt::Bool)
    σ_comp ~ truncated(Normal(0, 2); lower = 0)
    σ_mach ~ truncated(Normal(0, 2); lower = 0)
    σ ~ truncated(Normal(0, 1); lower = 0)
    # centred effects (as `_cdrv`, #28): each driver and car has thousands of laps,
    # so the non-centred form a = σ·z would tie σ to every z (a funnel)
    a_comp ~ filldist(Normal(0, σ_comp), length(d.competitors))
    b_mach ~ filldist(Normal(0, σ_mach), length(d.machines))
    μ_γ ~ Normal(0, 2)
    γ ~ filldist(Normal(μ_γ, 2), length(d.races))
    μ_δ ~ Normal(0, 2)
    τ_δ ~ truncated(Normal(0, 2); lower = 0)
    δ ~ filldist(Normal(μ_δ, τ_δ), length(d.races))
    if tyres
        β ~ filldist(Normal(0, 1), 2)
    else
        β = zeros(2)
    end
    if slopes
        τ_slope ~ truncated(Normal(0, 0.5); lower = 0)
        u_slope ~ filldist(Normal(), length(d.competitors))
        sl = career_slopes(τ_slope, u_slope)
    else
        sl = zeros(length(d.competitors))
    end
    if dev
        τ_dev ~ truncated(Normal(0, 1); lower = 0)
        u_dev ~ filldist(Normal(), length(d.machines))
        dv = τ_dev .* center_by(u_dev, d.mach_season, mach_counts)
    else
        dv = zeros(length(d.machines))
    end
    if lead
        η ~ filldist(Normal(0, 1), 1)
    else
        η = zeros(1)
    end
    if tyre_mgmt
        τ_td ~ truncated(Normal(0, 0.5); lower = 0)
        u_td ~ filldist(Normal(), length(d.competitors))
        τ_tm ~ truncated(Normal(0, 0.5); lower = 0)
        u_tm ~ filldist(Normal(), length(d.machines))
        td = career_slopes(τ_td, u_td)                         # τ·(u - mean u)
        tm = τ_tm .* center_by(u_tm, d.mach_season, mach_counts)
    else
        td, tm = zeros(length(d.competitors)), zeros(length(d.machines))
    end
    a_π ~ Normal(-1.5, 1)
    a_λ ~ Normal(0, 1)
    a = sum_to_zero(a_comp)
    b = center_by(b_mach, d.mach_season, mach_counts)
    @addlogprob! lap_loglik(γ, δ, a, b, β, sl, dv, η, td, tm, σ, logistic(a_π), exp(a_λ), d)
end

lap_mach_counts(d::LapData) = [count(==(k), d.mach_season) for k in eachindex(d.seasons)]
lap_effects(d::LapData; tyres::Bool = true, slopes::Bool = false, dev::Bool = false, lead::Bool = false,
            tyre_mgmt::Bool = false) =
    lap_effects(d, lap_mach_counts(d), tyres, slopes, dev, lead, tyre_mgmt)

"""
    fit_laps(d::LapData; n_samples=1000, n_chains=1, checkpoint=nothing, seed=1, ...)

Sample `lap_effects`, starting near the prior centre with race intercepts and
slopes from a per-race least-squares line (jittered per chain). Options as in
`fit_paceloss`, including `checkpoint` (resumable blocks, one chain), and
`tyres`, `slopes`, `dev`, `lead` and `tyre_mgmt` as in `lap_effects`. With
`warm_start = lap_warm_start(...)` (checkpointed fits), NUTS starts from that
metric and step size (`lap_warm_sampler`) and adapts for `n_adapts` iterations.
"""
function fit_laps(d::LapData; n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                  sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                  progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100,
                  checkpoint::Union{Nothing,AbstractString} = nothing, seed::Int = 1,
                  warm_start = nothing, n_adapts::Int = 500, tyres::Bool = true, slopes::Bool = false,
                  dev::Bool = false, lead::Bool = false, tyre_mgmt::Bool = false, kwargs...)
    checkpoint === nothing || n_chains == 1 || throw(ArgumentError("checkpointed fits run one chain per call"))
    warm_start === nothing || checkpoint !== nothing ||
        throw(ArgumentError("warm_start is implemented for checkpointed fits"))
    model = lap_effects(d; tyres, slopes, dev, lead, tyre_mgmt)
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
    X = tyres ? hcat(sv, sv .^ 2) : zeros(length(sv), 0)
    β0 = tyres ? (X' * X) \ (X' * res0) : Float64[]
    e0 = res0 .- X * β0
    σ0 = 1.4826 * median(abs.(e0 .- median(e0)))
    init() = InitFromParams(merge((; σ_comp = 0.3 + 0.4 * rand(rng), σ_mach = 0.3 + 0.4 * rand(rng),
                             σ = σ0 * (0.9 + 0.2 * rand(rng)),
                             a_comp = 0.05 .* randn(rng, length(d.competitors)),
                             b_mach = 0.05 .* randn(rng, length(d.machines)),
                             μ_γ = mean(γ0), γ = γ0 .+ 0.05 .* randn(rng, length(γ0)),
                             μ_δ = mean(δ0), τ_δ = std(δ0) * (0.9 + 0.2 * rand(rng)),
                             δ = δ0 .+ 0.05 .* randn(rng, length(δ0)),
                             a_π = -1.5 + 0.1 * randn(rng), a_λ = 0.1 * randn(rng)),
                            tyres ? (; β = β0 .+ 0.01 .* randn(rng, 2)) : (;),
                            slopes ? (; τ_slope = 0.1 + 0.1 * rand(rng),
                                        u_slope = 0.1 .* randn(rng, length(d.competitors))) : (;),
                            dev ? (; τ_dev = 0.1 + 0.1 * rand(rng), u_dev = 0.1 .* randn(rng, length(d.machines))) : (;),
                            lead ? (; η = 0.01 .* randn(rng, 1)) : (;),
                            tyre_mgmt ? (; τ_td = 0.05 + 0.05 * rand(rng), u_td = 0.1 .* randn(rng, length(d.competitors)),
                                           τ_tm = 0.05 + 0.05 * rand(rng), u_tm = 0.1 .* randn(rng, length(d.machines))) : (;)))
    external = warm_start === nothing ? nothing : lap_warm_sampler(warm_start.Minv, warm_start.ϵ)
    checkpoint === nothing ||
        return run_nuts_checkpointed(model, init(), checkpoint; n_samples, n_adapts, seed, progress_log, log_every,
                                     external, kwargs...)
    return run_nuts(model, [init() for _ in 1:n_chains]; n_samples, n_chains, ensemble, sampler, rng,
                    progress, progress_log, log_every, kwargs...)
end

# Warm start (#12): the 1996–2019 fit spends most of its warm-up at maximum tree
# depth before the first mass-matrix window (~80 s per iteration on 470k laps).
# Starting from the step size and diagonal metric adapted on a smaller fit
# (2011–2019) skips that phase.

"Parameter layout of `lap_effects` in the sampler's (unconstrained) vector: name => (length, level labels or nothing)."
lap_layout(d::LapData; tyres::Bool = true, slopes::Bool = false, dev::Bool = false, lead::Bool = false,
           tyre_mgmt::Bool = false) =
    filter(p -> (tyres || first(p) !== :β) && (slopes || first(p) ∉ (:τ_slope, :u_slope)) &&
                (dev || first(p) ∉ (:τ_dev, :u_dev)) && (lead || first(p) !== :η) &&
                (tyre_mgmt || first(p) ∉ (:τ_td, :u_td, :τ_tm, :u_tm)),
                         [:σ_comp => (1, nothing), :σ_mach => (1, nothing), :σ => (1, nothing),
                          :a_comp => (length(d.competitors), d.competitors), :b_mach => (length(d.machines), d.machines),
                          :μ_γ => (1, nothing), :γ => (length(d.races), d.races), :μ_δ => (1, nothing),
                          :τ_δ => (1, nothing), :δ => (length(d.races), d.races), :β => (2, nothing),
                          :τ_slope => (1, nothing), :u_slope => (length(d.competitors), d.competitors),
                          :τ_dev => (1, nothing), :u_dev => (length(d.machines), d.machines),
                          :η => (1, nothing),
                          :τ_td => (1, nothing), :u_td => (length(d.competitors), d.competitors),
                          :τ_tm => (1, nothing), :u_tm => (length(d.machines), d.machines),
                          :a_π => (1, nothing), :a_λ => (1, nothing)])

# Inverse-metric entries for blocks a warm start's old fit doesn't have (career
# slopes or car development added to a fit without them), on the sampler's scale
# (log τ; standardised u). Kept small: a variance set too large forces a small
# global step size (the first slopes fit warmed up at ~197 s per iteration with
# u_slope at 0.5, #12), while one too small only slows that direction until
# adaptation corrects it.
const LAP_WARM_DEFAULT = Dict(:τ_slope => 0.01, :u_slope => 0.01, :τ_dev => 0.01, :u_dev => 0.01, :β => 1e-4,
                              :η => 1e-4, :τ_td => 0.01, :u_td => 0.01, :τ_tm => 0.01, :u_tm => 0.01)

"""
    lap_warm_start(state, d_old::LapData, d_new::LapData; tyres_old = true, tyres_new = tyres_old,
                   slopes_old = false, slopes_new = slopes_old, dev_old = false, dev_new = dev_old,
                   lead_old = false, lead_new = lead_old, tyre_mgmt_old = false, tyre_mgmt_new = tyre_mgmt_old) -> (; Minv, ϵ)

Diagonal inverse metric and step size for a fit to `d_new`, from the adapted
NUTS `state` of a fit to `d_old` (`only(Turing.loadstate(chain))`, or its
`lap_adapted` summary). Levels
(drivers, car-seasons, races) present in both keep their adapted variance; new
levels get their group's median. Scalars are scaled by the ratio of lap
counts (posterior variance ∝ 1/data). Adaptation still runs on the new fit;
this only gives it a good start. `tyres_*` and `slopes_*` say whether each fit
has the tyre terms and career slopes (`lap_effects`); a block the old fit
doesn't have starts from `LAP_WARM_DEFAULT`.
"""
function lap_warm_start(state, d_old::LapData, d_new::LapData; tyres_old::Bool = true, tyres_new::Bool = tyres_old,
                        slopes_old::Bool = false, slopes_new::Bool = slopes_old,
                        dev_old::Bool = false, dev_new::Bool = dev_old, lead_old::Bool = false,
                        lead_new::Bool = lead_old, tyre_mgmt_old::Bool = false, tyre_mgmt_new::Bool = tyre_mgmt_old)
    ad = lap_adapted(state)
    Mo = ad.Minv
    lo = lap_layout(d_old; tyres = tyres_old, slopes = slopes_old, dev = dev_old, lead = lead_old, tyre_mgmt = tyre_mgmt_old)
    ln = lap_layout(d_new; tyres = tyres_new, slopes = slopes_new, dev = dev_new, lead = lead_new, tyre_mgmt = tyre_mgmt_new)
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
