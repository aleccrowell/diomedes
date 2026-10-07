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
# evolution; a slope per race) and the tyres' stint age in laps since the last
# stop (`stint`, 1 on the first lap after the out-lap).

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
`frac` and stint age `stint`.
"""
struct LapData
    y::Vector{Float64}
    comp::Vector{Int}
    mach::Vector{Int}
    race::Vector{Int}
    frac::Vector{Float64}
    stint::Vector{Float64}
    mach_season::Vector{Int}      # season index of each machine level
    race_season::Vector{Int}      # season index of each race
    seasons::Vector{Int}
    competitors::Vector{String}
    machines::Vector{String}
    races::Vector{String}
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
    recorded = Dict{Tuple{String,String},Set{Int}}()
    if pits === :recorded
        for r in eachrow(pit_stops)
            push!(get!(recorded, (r.event_id, r.competitor_id), Set{Int}()), r.lap)
        end
    end
    excl = Dict(:lap1 => 0, :slow_race => 0, :stoppage => 0, :pit => 0, :no_machine => 0)
    keep = DataFrame(season = Int[], event_id = String[], competitor_id = String[], machine_key = String[],
                     lap = Int[], time_ms = Float64[], frac = Float64[], stint = Int[])
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
                push!(keep, (first(car.season), k[1], k[2], mkey[k], l, times[l], l / L, stint))
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
    return LapData(keep.y, [ci[c] for c in keep.competitor_id], [mi[m] for m in keep.machine_key],
                   [ri[e] for e in keep.event_id], keep.frac, Float64.(keep.stint), mach_season, race_season,
                   seasons, comps, machs, races, keep, excl)
end

"""
    lap_loglik(γ, δ, a, b, β, σ, π, λ, d::LapData)

Log-likelihood of the laps in `d` under pace + loss noise (`paceloss_logpdf`),
with per-lap mean

    μ = γ[race] + δ[race]·frac + a[driver] + b[machine] + β[1]·stint + β[2]·stint²/10

(stint age in laps). Has a fused reverse rule: per-lap derivatives from
ForwardDiff duals, scattered into the parameter vectors.
"""
function lap_loglik(γ, δ, a, b, β, σ, π, λ, d::LapData)
    s = 0.0
    for j in eachindex(d.y)
        r, st = d.race[j], d.stint[j]
        μ = γ[r] + δ[r] * d.frac[j] + a[d.comp[j]] + b[d.mach[j]] + β[1] * st + β[2] * st^2 / 10
        s += paceloss_logpdf(d.y[j] - μ, σ, π, λ)
    end
    return s
end

function ChainRulesCore.rrule(::typeof(lap_loglik), γ, δ, a, b, β, σ, π, λ, d::LapData)
    gγ, gδ, ga, gb, gβ = zeros(length(γ)), zeros(length(δ)), zeros(length(a)), zeros(length(b)), zeros(2)
    gσ, gπ, gλ, val = 0.0, 0.0, 0.0, 0.0
    for j in eachindex(d.y)
        r, st, f, y = d.race[j], d.stint[j], d.frac[j], d.y[j]
        μ = γ[r] + δ[r] * f + a[d.comp[j]] + b[d.mach[j]] + β[1] * st + β[2] * st^2 / 10
        v, p = value_grad4((m, s, q, l) -> paceloss_logpdf(y - m, s, q, l), μ, σ, π, λ)
        val += v
        gγ[r] += p[1]; gδ[r] += p[1] * f; ga[d.comp[j]] += p[1]; gb[d.mach[j]] += p[1]
        gβ[1] += p[1] * st; gβ[2] += p[1] * st^2 / 10
        gσ += p[2]; gπ += p[3]; gλ += p[4]
    end
    pullback(Δ) = (NoTangent(), Δ .* gγ, Δ .* gδ, Δ .* ga, Δ .* gb, Δ .* gβ, Δ * gσ, Δ * gπ, Δ * gλ, NoTangent())
    return val, pullback
end
ReverseDiff.@grad_from_chainrules lap_loglik(γ::ReverseDiff.TrackedArray, δ::ReverseDiff.TrackedArray,
                                             a::ReverseDiff.TrackedArray, b::ReverseDiff.TrackedArray,
                                             β::ReverseDiff.TrackedArray, σ::ReverseDiff.TrackedReal,
                                             π::ReverseDiff.TrackedReal, λ::ReverseDiff.TrackedReal, d::LapData)

"""
    lap_effects(d::LapData)

Lap time (% of the race's median clean lap) = race intercept + race slope·frac
+ driver + car-season + tyre(stint age) + pace noise + incident loss.

- Driver effects `σ_comp·z` sum to zero; car-season effects `σ_mach·z` sum to
  zero within each season (with race intercepts only within-season car
  differences are identified), as in `gap_effects`.
- Race intercepts γ ~ N(μ_γ, 2), with μ_γ learned; race slopes δ ~ N(μ_δ, τ_δ):
  the change in lap time from the start to the end of a race (fuel burn, track
  evolution), partially pooled.
- Tyres: β[1]·stint + β[2]·stint²/10, in % per lap of stint age.
- Noise: pace + loss (`paceloss_logpdf`) with one σ, incident probability π
  and mean loss λ: traffic, mistakes and damage are losses, as in the race model.
"""
@model function lap_effects(d::LapData, mach_counts::Vector{Int})
    σ_comp ~ truncated(Normal(0, 2); lower = 0)
    σ_mach ~ truncated(Normal(0, 2); lower = 0)
    σ ~ truncated(Normal(0, 1); lower = 0)
    z_comp ~ filldist(Normal(), length(d.competitors))
    z_mach ~ filldist(Normal(), length(d.machines))
    μ_γ ~ Normal(0, 2)
    γ ~ filldist(Normal(μ_γ, 2), length(d.races))
    μ_δ ~ Normal(0, 2)
    τ_δ ~ truncated(Normal(0, 2); lower = 0)
    δ ~ filldist(Normal(μ_δ, τ_δ), length(d.races))
    β ~ filldist(Normal(0, 1), 2)
    a_π ~ Normal(-1.5, 1)
    a_λ ~ Normal(0, 1)
    a = σ_comp .* sum_to_zero(z_comp)
    b = σ_mach .* center_by(z_mach, d.mach_season, mach_counts)
    @addlogprob! lap_loglik(γ, δ, a, b, β, σ, logistic(a_π), exp(a_λ), d)
end

lap_mach_counts(d::LapData) = [count(==(k), d.mach_season) for k in eachindex(d.seasons)]
lap_effects(d::LapData) = lap_effects(d, lap_mach_counts(d))

"""
    fit_laps(d::LapData; n_samples=1000, n_chains=1, checkpoint=nothing, seed=1, ...)

Sample `lap_effects`, starting near the prior centre with race intercepts and
slopes from a per-race least-squares line (jittered per chain). Options as in
`fit_paceloss`, including `checkpoint` (resumable blocks, one chain).
"""
function fit_laps(d::LapData; n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                  sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                  progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100,
                  checkpoint::Union{Nothing,AbstractString} = nothing, seed::Int = 1, kwargs...)
    checkpoint === nothing || n_chains == 1 || throw(ArgumentError("checkpointed fits run one chain per call"))
    model = lap_effects(d)
    # per-race line through the laps, as starting values
    γ0, δ0 = zeros(length(d.races)), zeros(length(d.races))
    for r in eachindex(d.races)
        sel = d.race .== r
        f, y = d.frac[sel], d.y[sel]
        δ0[r] = var(f) > 0 ? cov(f, y) / var(f) : 0.0
        γ0[r] = mean(y) - δ0[r] * mean(f)
    end
    init() = InitFromParams((; σ_comp = 0.3 + 0.4 * rand(rng), σ_mach = 0.3 + 0.4 * rand(rng),
                             σ = 0.3 + 0.4 * rand(rng),
                             z_comp = 0.1 .* randn(rng, length(d.competitors)),
                             z_mach = 0.1 .* randn(rng, length(d.machines)),
                             μ_γ = mean(γ0), γ = γ0 .+ 0.05 .* randn(rng, length(γ0)),
                             μ_δ = mean(δ0), τ_δ = std(δ0) * (0.9 + 0.2 * rand(rng)),
                             δ = δ0 .+ 0.05 .* randn(rng, length(δ0)), β = 0.01 .* randn(rng, 2),
                             a_π = -1.5 + 0.1 * randn(rng), a_λ = 0.1 * randn(rng)))
    checkpoint === nothing ||
        return run_nuts_checkpointed(model, init(), checkpoint; n_samples, seed, progress_log, log_every, kwargs...)
    return run_nuts(model, [init() for _ in 1:n_chains]; n_samples, n_chains, ensemble, sampler, rng,
                    progress, progress_log, log_every, kwargs...)
end
