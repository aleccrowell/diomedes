# Retirement (DNF) model: competing-risks hazards per start (#13).
#
# Each start is a survival record in laps: `n` laps completed, then either the
# race distance ran out (censored: finished, lapped, or running but not
# classified) or the car retired during lap n+1 with a cause. Causes:
#   - mechanical: the car broke (engine, gearbox, ...); hazard has a car-season
#     reliability effect,
#   - incident: the driver crashed alone (accident, spun off); hazard has a
#     driver incident-proneness effect,
#   - collision: contact with another car; the driver effect enters with a
#     learned loading β_col, since a collision is not one driver's alone,
#   - other (disqualified, out of fuel, puncture, driver unwell, "Retired", ...):
#     treated as censoring at the laps completed (non-informative).
# Non-starters (did not (pre)qualify, withdrew before the start) are dropped.
#
# Time is measured in fractions of the race distance L (the winner's laps), so
# hazards are per race, comparable between short and long laps. Every cause k
# has cumulative hazard
#     H_k(n) = exp(η_k) · (B_k·[n ≥ 1] + n/L),        B_k = exp(b1_k)
# i.e. a constant hazard over the race plus an extra first-lap jump B_k (start
# crashes, launch failures), both scaled by the start's linear predictor η_k.
# Within each lap the hazard is constant, so for a retirement from cause k in
# lap n+1, with ΔH_j = H_j(n+1) - H_j(n) and ΔH = Σ_j ΔH_j,
#     log p = -Σ_j H_j(n) + log(1 - exp(-ΔH)) + log(ΔH_k / ΔH),
# and a censored start contributes -Σ_j H_j(n).

const RETIRE_CAUSES = (:mech, :incident, :collision)

const MECH_STATUS = Set([
    "Engine", "Gearbox", "Suspension", "Transmission", "Electrical", "Brakes", "Clutch", "Fuel system",
    "Turbo", "Hydraulics", "Overheating", "Ignition", "Oil leak", "Throttle", "Halfshaft", "Wheel",
    "Oil pressure", "Fuel pump", "Differential", "Fuel leak", "Steering", "Radiator", "Wheel bearing",
    "Injection", "Fuel pressure", "Alternator", "Water leak", "Exhaust", "Chassis", "Power Unit",
    "Mechanical", "Magneto", "Driveshaft", "Axle", "Heat shield fire", "Battery", "Oil pump", "Distributor",
    "Power loss", "Oil pipe", "Vibrations", "Electronics", "Wheel nut", "Rear wing", "Front wing",
    "Water pump", "Water pressure", "Supercharger", "ERS", "Technical", "Pneumatics", "Fuel", "Wheel rim",
    "Fire", "Spark plugs", "Fuel pipe", "Water pipe", "Track rod", "Oil line", "Drivetrain", "Engine fire",
    "Crankshaft", "Launch control", "CV joint", "Engine misfire", "Brake duct", "Handling", "Driver Seat",
    "Seat", "Safety belt"])
const INCIDENT_STATUS = Set(["Accident", "Spun off", "Fatal accident"])
const COLLISION_STATUS = Set(["Collision", "Collision damage", "Broken wing", "Damage"])
const NONSTART_STATUS = Set(["Did not qualify", "Did not prequalify", "107% Rule", "Did not start",
                             "Not restarted"])

"""
    retirement_cause(status, laps) -> Symbol

Cause of a start's end, from the source status: `:finish` (ran to the end,
classified or not), one of `RETIRE_CAUSES`, `:other` (retired, cause not
attributable: censored), or `:nonstart`. A withdrawal with no laps is a non-start.
"""
function retirement_cause(status::AbstractString, laps)
    (status == "Finished" || is_lapped_status(status) || status == "Not classified") && return :finish
    status in NONSTART_STATUS && return :nonstart
    status == "Withdrew" && return coalesce(laps, 0) == 0 ? :nonstart : :other
    status in MECH_STATUS && return :mech
    status in INCIDENT_STATUS && return :incident
    status in COLLISION_STATUS && return :collision
    return :other
end

"""
    RetireData

Starts for the retirement model: per start the driver, machine-season and race
indices, laps completed `n`, race distance `L` (laps), and `cause` (0 =
censored, else the index into `RETIRE_CAUSES`).
"""
struct RetireData
    comp::Vector{Int}
    mach::Vector{Int}
    race::Vector{Int}
    n::Vector{Int}
    L::Vector{Int}
    cause::Vector{Int}
    season::Vector{Int}           # season index (1..n_seasons) of each start
    mach_season::Vector{Int}      # season index of each machine level
    race_season::Vector{Int}      # season index of each race
    seasons::Vector{Int}          # calendar years
    competitors::Vector{String}
    machines::Vector{String}
    races::Vector{String}
    rows::DataFrame               # starts used, with :cause (Symbol) and :L
end

function Base.show(io::IO, d::RetireData)
    k = [count(==(c), d.cause) for c in 1:length(RETIRE_CAUSES)]
    print(io, "RetireData($(length(d.n)) starts: ", join(("$(n) $(c)" for (c, n) in zip(RETIRE_CAUSES, k)), ", "),
          ", $(count(==(0), d.cause)) censored; $(length(d.competitors)) drivers, ",
          "$(length(d.machines)) machine-seasons, $(length(d.races)) races, $(length(d.seasons)) seasons)")
end

"""
    prepare_retirements(results; machine_key) -> RetireData

Starts from a circuit-racing results table (needs `laps` and `status`). The race
distance is the most laps completed by anyone in the race. Starts with unknown
laps and non-starters are dropped, as are races without a known distance.
"""
function prepare_retirements(results::AbstractDataFrame; machine_key = r -> string(r.machine_id, "_", r.season))
    df = DataFrame(results)
    df = df[.!ismissing.(df.laps), :]
    df.cause = [retirement_cause(r.status, r.laps) for r in eachrow(df)]
    df = df[df.cause .!= :nonstart, :]
    df.race_key = string.(df.series, "/", df.event_id, "/", df.stage_id)
    df.L = zeros(Int, nrow(df))
    for g in groupby(df, :race_key)
        g.L .= maximum(g.laps)
    end
    df = df[df.L .> 0, :]
    df.n = min.(df.laps, df.L)
    # a retirement on the last lap of the distance still happened during lap n+1 ≤ L
    df.n = [c in RETIRE_CAUSES && n == L ? L - 1 : n for (c, n, L) in zip(df.cause, df.n, df.L)]
    df.machine_key = [machine_key(r) for r in eachrow(df)]

    seasons = sort(unique(df.season))
    si = Dict(s => i for (i, s) in enumerate(seasons))
    comps, machs, races = sort(unique(df.competitor_id)), sort(unique(df.machine_key)), sort(unique(df.race_key))
    ci = Dict(l => i for (i, l) in enumerate(comps))
    mi = Dict(l => i for (i, l) in enumerate(machs))
    ri = Dict(l => i for (i, l) in enumerate(races))
    mach_season, race_season = zeros(Int, length(machs)), zeros(Int, length(races))
    for r in eachrow(df)
        mach_season[mi[r.machine_key]] = si[r.season]
        race_season[ri[r.race_key]] = si[r.season]
    end
    cause = [something(findfirst(==(c), RETIRE_CAUSES), 0) for c in df.cause]
    return RetireData([ci[k] for k in df.competitor_id], [mi[k] for k in df.machine_key],
                      [ri[k] for k in df.race_key], df.n, df.L, cause, [si[s] for s in df.season],
                      mach_season, race_season, seasons, comps, machs, races, df)
end

# Log-likelihood of one start given its three linear predictors and first-lap
# jumps B (exp of b1), see the file header.
@inline function retire_row(ηm, ηi, ηc, b1m, b1i, b1c, n::Int, L::Int, cause::Int)
    first = n >= 1
    Hm = exp(ηm) * ((first ? exp(b1m) : zero(b1m)) + n / L)
    Hi = exp(ηi) * ((first ? exp(b1i) : zero(b1i)) + n / L)
    Hc = exp(ηc) * ((first ? exp(b1c) : zero(b1c)) + n / L)
    ll = -(Hm + Hi + Hc)
    cause == 0 && return ll
    # increments over lap n+1 (the first-lap jump only if n == 0)
    dm = exp(ηm) * ((first ? zero(b1m) : exp(b1m)) + 1 / L)
    di = exp(ηi) * ((first ? zero(b1i) : exp(b1i)) + 1 / L)
    dc = exp(ηc) * ((first ? zero(b1c) : exp(b1c)) + 1 / L)
    dH = dm + di + dc
    dk = cause == 1 ? dm : cause == 2 ? di : dc
    return ll + log1mexp(-dH) + log(dk) - log(dH)
end

"""
    retire_loglik(αm, αi, αc, car, drv, β_col, rm, ri, rc, b1, d::RetireData)

Total log-likelihood of the starts in `d`. Per start j the linear predictors are

    η_mech      = αm[season] + car[mach] + rm[race]
    η_incident  = αi[season] + drv[comp] + ri[race]
    η_collision = αc[season] + β_col·drv[comp] + rc[race]

and `b1` holds the log first-lap jumps (one per cause). Has a fused reverse rule:
per-start derivatives from ForwardDiff duals, scattered into the parameter
vectors, so the sampler's tape never holds per-start arrays.
"""
function retire_loglik(αm, αi, αc, car, drv, β_col, rm, ri, rc, b1, d::RetireData)
    s = 0.0
    for j in eachindex(d.n)
        t, dv, r = d.season[j], drv[d.comp[j]], d.race[j]
        s += retire_row(αm[t] + car[d.mach[j]] + rm[r], αi[t] + dv + ri[r], αc[t] + β_col * dv + rc[r],
                        b1[1], b1[2], b1[3], d.n[j], d.L[j], d.cause[j])
    end
    return s
end

function ChainRulesCore.rrule(::typeof(retire_loglik), αm, αi, αc, car, drv, β_col, rm, ri, rc, b1, d::RetireData)
    gαm, gαi, gαc = zeros(length(αm)), zeros(length(αi)), zeros(length(αc))
    gcar, gdrv = zeros(length(car)), zeros(length(drv))
    grm, gri, grc = zeros(length(rm)), zeros(length(ri)), zeros(length(rc))
    gb, gβ, val = zeros(3), 0.0, 0.0
    for j in eachindex(d.n)
        t, c, m, r = d.season[j], d.comp[j], d.mach[j], d.race[j]
        n, L, k = d.n[j], d.L[j], d.cause[j]
        v, p = value_gradN((a, b, e, x, y, z) -> retire_row(a, b, e, x, y, z, n, L, k),
                           αm[t] + car[m] + rm[r], αi[t] + drv[c] + ri[r], αc[t] + β_col * drv[c] + rc[r],
                           b1[1], b1[2], b1[3])
        val += v
        gαm[t] += p[1]; gcar[m] += p[1]; grm[r] += p[1]
        gαi[t] += p[2]; gdrv[c] += p[2] + β_col * p[3]; gri[r] += p[2]
        gαc[t] += p[3]; gβ += p[3] * drv[c]; grc[r] += p[3]
        gb[1] += p[4]; gb[2] += p[5]; gb[3] += p[6]
    end
    pullback(Δ) = (NoTangent(), Δ .* gαm, Δ .* gαi, Δ .* gαc, Δ .* gcar, Δ .* gdrv, Δ * gβ,
                   Δ .* grm, Δ .* gri, Δ .* grc, Δ .* gb, NoTangent())
    return val, pullback
end
ReverseDiff.@grad_from_chainrules retire_loglik(αm::ReverseDiff.TrackedArray, αi::ReverseDiff.TrackedArray,
                                                αc::ReverseDiff.TrackedArray, car::ReverseDiff.TrackedArray,
                                                drv::ReverseDiff.TrackedArray, β_col::ReverseDiff.TrackedReal,
                                                rm::ReverseDiff.TrackedArray, ri::ReverseDiff.TrackedArray,
                                                rc::ReverseDiff.TrackedArray, b1::ReverseDiff.TrackedArray,
                                                d::RetireData)

"""
    retirement_effects(d::RetireData)

Competing-risks retirement model (see the file header). Linear predictors per start:

    η_mech      = α_mech[season] + car-season reliability + race effect
    η_incident  = α_inc[season]  + driver incident-proneness + race effect
    η_collision = α_col[season]  + β_col · driver incident-proneness + race effect

- `α_*[season]`: a level plus a random walk over seasons with Student-t(3)
  steps (as the era terms of the pace model, `rw_path`).
- Car-season effects `σ_car·z_car` are centred within season, race effects
  `σ_race_k·u_k` within season, and driver effects `σ_drv·z_drv` overall, so
  the season levels carry every average.
- `β_col` is the loading of driver incident-proneness on collisions: 0 would
  mean collisions are not driver-specific, 1 that they follow solo incidents.
- `b1`: log first-lap jump of each hazard, in race-distance units.

Effects are on the log-hazard scale: a car effect of +0.5 is a mechanical
failure rate 1.65× its season's average.
"""
@model function retirement_effects(d::RetireData, mach_counts::Vector{Int}, race_counts::Vector{Int})
    S = length(d.seasons)
    a ~ filldist(Normal(-1.5, 1.5), 3)                      # mean log hazard per race distance
    τ_rw ~ filldist(truncated(Normal(0, 0.25); lower = 0), 3)
    e_m ~ filldist(TDist(3), S - 1)
    e_i ~ filldist(TDist(3), S - 1)
    e_c ~ filldist(TDist(3), S - 1)
    b1 ~ filldist(Normal(-2, 2), 3)
    σ_car ~ truncated(Normal(0, 1); lower = 0)
    σ_drv ~ truncated(Normal(0, 1); lower = 0)
    β_col ~ Normal(0, 1)
    σ_race ~ filldist(truncated(Normal(0, 1); lower = 0), 3)
    z_car ~ filldist(Normal(), length(d.machines))
    z_drv ~ filldist(Normal(), length(d.competitors))
    u_m ~ filldist(Normal(), length(d.races))
    u_i ~ filldist(Normal(), length(d.races))
    u_c ~ filldist(Normal(), length(d.races))
    car = σ_car .* center_by(z_car, d.mach_season, mach_counts)
    drv = σ_drv .* sum_to_zero(z_drv)
    αm = a[1] .+ rw_path(τ_rw[1] .* e_m)
    αi = a[2] .+ rw_path(τ_rw[2] .* e_i)
    αc = a[3] .+ rw_path(τ_rw[3] .* e_c)
    rm = σ_race[1] .* center_by(u_m, d.race_season, race_counts)
    ri = σ_race[2] .* center_by(u_i, d.race_season, race_counts)
    rc = σ_race[3] .* center_by(u_c, d.race_season, race_counts)
    @addlogprob! retire_loglik(αm, αi, αc, car, drv, β_col, rm, ri, rc, b1, d)
end

retire_counts(d::RetireData) = ([count(==(k), d.mach_season) for k in eachindex(d.seasons)],
                                [count(==(k), d.race_season) for k in eachindex(d.seasons)])

function retirement_effects(d::RetireData)
    length(d.seasons) >= 2 || throw(ArgumentError("random walks over seasons need at least 2 seasons"))
    return retirement_effects(d, retire_counts(d)...)
end

"""
    fit_retirement(d::RetireData; n_samples=1000, n_chains=1, ...)

Sample `retirement_effects`, starting near the prior centre with each cause's
level at its observed rate (jittered per chain). Sampler, progress and ensemble
options as in `fit_gaps` (uncompiled ReverseDiff: the likelihood has a fused rule).
"""
function fit_retirement(d::RetireData; n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                        sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                        progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100, kwargs...)
    model = retirement_effects(d)
    S = length(d.seasons)
    # observed events per race distance of exposure, per cause
    expo = sum(d.n ./ d.L)
    rate = [log(max(count(==(k), d.cause), 1) / expo) for k in 1:3]
    init() = InitFromParams((; a = rate .+ 0.1 .* randn(rng, 3), τ_rw = 0.05 .+ 0.05 .* rand(rng, 3),
                             e_m = 0.1 .* randn(rng, S - 1), e_i = 0.1 .* randn(rng, S - 1),
                             e_c = 0.1 .* randn(rng, S - 1), b1 = -2 .+ 0.2 .* randn(rng, 3),
                             σ_car = 0.3 + 0.2 * rand(rng), σ_drv = 0.3 + 0.2 * rand(rng), β_col = 0.1 * randn(rng),
                             σ_race = 0.2 .+ 0.2 .* rand(rng, 3),
                             z_car = 0.1 .* randn(rng, length(d.machines)),
                             z_drv = 0.1 .* randn(rng, length(d.competitors)),
                             u_m = 0.1 .* randn(rng, length(d.races)), u_i = 0.1 .* randn(rng, length(d.races)),
                             u_c = 0.1 .* randn(rng, length(d.races))))
    return run_nuts(model, [init() for _ in 1:n_chains]; n_samples, n_chains, ensemble, sampler, rng,
                    progress, progress_log, log_every, kwargs...)
end
