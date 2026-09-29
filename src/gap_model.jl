# Gap-to-winner model with censored lapped finishers (#9).
#
# Outcome: y = 100·log(T / T_winner), the percentage gap to the stage winner.
# Lead-lap finishers have a time. A classified car that is k laps down has no
# time, but assuming constant pace its full-distance time is bounded by lap
# counts alone: a car that completed l of L laps finished its l-th lap at some
# t ∈ [T_w, T_w + one of its laps), so its pace p satisfies l·p ∈ [T_w, T_w + p),
# i.e. L·p ∈ [T_w·L/l, T_w·L/(l-1)). On the gap scale:
#     y ∈ [100·log(L/l), 100·log(L/(l-1)))
# Retirements (classified or not) are dropped as uninformative about pace.
# Times are racing times: official time minus any red-flag suspension
# (`suspended_ms`, see src/sources/suspensions.jl). Unknown suspensions
# (`missing`) count as 0, which is correct for aggregate-timed stopped races
# before 2002.
# Lapped cars are identified by status ("+N Lap(s)" in Ergast, "Lapped" in
# recent Jolpica data), not by position: Ergast gives some retirements a numeric
# position (e.g. Spain 2013: "Suspension" after 8 of 66 laps, position 22).
# Counting those as lapped put 599 retirements into the lapped set as huge
# "losses".

"Whether a status string means a lapped finisher."
is_lapped_status(s::AbstractString) = s == "Lapped" || occursin(r"^\+\d+ Laps?$", s)

"""
    GapData

Model inputs for `gap_effects`: timed rows (`y`) and interval-censored lapped
rows (`lo`, `hi`), each with competitor, machine-season and race indices.
"""
struct GapData
    y::Vector{Float64}
    t_comp::Vector{Int}
    t_mach::Vector{Int}
    t_race::Vector{Int}
    lo::Vector{Float64}
    hi::Vector{Float64}
    c_comp::Vector{Int}
    c_mach::Vector{Int}
    c_race::Vector{Int}
    mach_season::Vector{Int}      # season index of each machine level
    race_season::Vector{Int}      # calendar year of each race
    race_minutes::Vector{Float64} # winner's race time in minutes (race duration)
    t_age::Vector{Int}            # age bin of each timed row (0 = unknown)
    c_age::Vector{Int}            # age bin of each lapped row (0 = unknown)
    age_years::Vector{Int}        # age (whole years) of each age bin
    age_rows::Vector{Float64}     # number of rows in each age bin
    competitors::Vector{String}
    machines::Vector{String}
    races::Vector{String}
    rows::DataFrame               # rows used: :kind (:timed / :lapped), :y, :lo, :hi
end

function Base.show(io::IO, g::GapData)
    print(io, "GapData($(length(g.y)) timed + $(length(g.lo)) lapped obs, ",
          "$(length(g.competitors)) competitors, $(length(g.machines)) machine-seasons, ",
          "$(length(g.races)) races)")
end

"""
    prepare_gaps(results; include_lapped = true, machine_key)

Build `GapData` from a results table. Per stage, the winner is the fastest
timed row, and the race distance `L` is the winner's lap count. Timed rows get
their % gap to the winner. With `include_lapped`, untimed rows with a lapped
status (see `is_lapped_status`) that completed 2 ≤ l < L laps become intervals
(see file header); with
`max_laps_down = k`, only those at most k laps down. Everything else is dropped.
"""
function prepare_gaps(results::AbstractDataFrame; include_lapped::Bool = true,
                      max_laps_down::Union{Nothing,Int} = nothing,
                      machine_key = r -> string(r.machine_id, "_", r.season))
    df = DataFrame(results)
    df.kind = fill(:drop, nrow(df))
    df.winner_ms = Vector{Union{Missing,Float64}}(missing, nrow(df))
    df.y = Vector{Union{Missing,Float64}}(missing, nrow(df))
    df.lo = Vector{Union{Missing,Float64}}(missing, nrow(df))
    df.hi = Vector{Union{Missing,Float64}}(missing, nrow(df))
    # racing time: official time minus any red-flag suspension (same for every car)
    # (tables without the column, e.g. simulated ones, have no suspensions)
    df.race_ms = hasproperty(df, :suspended_ms) ? df.time_ms .- coalesce.(df.suspended_ms, 0.0) : df.time_ms
    for g in groupby(df, [:series, :event_id, :stage_id])
        timed = findall(!ismissing, g.race_ms)
        isempty(timed) && continue
        w = timed[argmin(g.race_ms[timed])]
        T_w = g.race_ms[w]
        g.winner_ms .= T_w
        for i in timed
            g.kind[i] = :timed
            g.y[i] = 100 * log(g.race_ms[i] / T_w)
        end
        L = g.laps[w]
        (include_lapped && !ismissing(L)) || continue
        for i in eachindex(g.kind)
            l = g.laps[i]
            if ismissing(g.time_ms[i]) && is_lapped_status(g.status[i]) && !ismissing(l) && 2 <= l < L &&
               (max_laps_down === nothing || L - l <= max_laps_down)
                g.kind[i] = :lapped
                g.lo[i] = 100 * log(L / l)
                g.hi[i] = 100 * log(L / (l - 1))
            end
        end
    end
    df = df[df.kind .!= :drop, :]
    df.machine_key = [machine_key(r) for r in eachrow(df)]
    df.race_key = string.(df.series, "/", df.event_id, "/", df.stage_id)

    comps = sort(unique(df.competitor_id))
    machs = sort(unique(df.machine_key))
    races = sort(unique(df.race_key))
    ci = Dict(l => i for (i, l) in enumerate(comps))
    mi = Dict(l => i for (i, l) in enumerate(machs))
    ri = Dict(l => i for (i, l) in enumerate(races))
    seasons = sort(unique(df.season))
    si = Dict(s => i for (i, s) in enumerate(seasons))
    mach_season = zeros(Int, length(machs))
    for r in eachrow(df)
        mach_season[mi[r.machine_key]] = si[r.season]
    end

    race_season = zeros(Int, length(races))
    race_minutes = zeros(length(races))
    for r in eachrow(df)
        k = ri[r.race_key]
        race_season[k] = r.season
        race_minutes[k] = r.winner_ms / 60_000
    end

    # driver age in whole years at the event, binned (0 = unknown date of birth or event date)
    hasage = hasproperty(df, :competitor_birth) && hasproperty(df, :event_date)
    ageyrs = [hasage && !ismissing(r.competitor_birth) && !ismissing(r.event_date) ?
              floor(Int, Dates.value(r.event_date - r.competitor_birth) / 365.2425) : missing
              for r in eachrow(df)]
    known = collect(skipmissing(ageyrs))
    age_years = isempty(known) ? Int[] : collect(minimum(known):maximum(known))
    df.age_bin = [ismissing(a) ? 0 : a - first(age_years) + 1 for a in ageyrs]
    age_rows = [Float64(count(==(k), df.age_bin)) for k in eachindex(age_years)]

    t = df[df.kind .== :timed, :]
    c = df[df.kind .== :lapped, :]
    idx(d, col, map) = [map[k] for k in d[!, col]]
    return GapData(Float64.(t.y), idx(t, :competitor_id, ci), idx(t, :machine_key, mi),
                   idx(t, :race_key, ri),
                   Float64.(c.lo), Float64.(c.hi), idx(c, :competitor_id, ci),
                   idx(c, :machine_key, mi), idx(c, :race_key, ri),
                   mach_season, race_season, race_minutes, t.age_bin, c.age_bin, age_years, age_rows,
                   comps, machs, races, df)
end

"""
    logdiffΦ(a, b)

`log(Φ(b) - Φ(a))` for a < b. Tails use the scaled complementary error function
so the difference never cancels catastrophically:
`erfc(x) - erfc(y) = exp(-x²)·(erfcx(x) - exp(x² - y²)·erfcx(y))`. Intervals that
straddle 0 use `erf` directly. About 5x faster than differencing two
`normlogcdf`s.
"""
function logdiffΦ(a, b)
    if a >= 0                       # upper tail: Φ(-a) - Φ(-b) = ½(erfc(a/√2) - erfc(b/√2))
        x, y = a / SQRT2, b / SQRT2
        return -x^2 + log(erfcx(x) - exp((x - y) * (x + y)) * erfcx(y)) - LOG2
    elseif b <= 0                   # lower tail, by symmetry
        return logdiffΦ(-b, -a)
    else                            # straddles 0
        return log((erf(b / SQRT2) - erf(a / SQRT2)) / 2)
    end
end
const SQRT2 = sqrt(2.0)
const LOG2 = log(2.0)

"Subtract the mean within each group; `group[i]` is element i's group index, `counts` the group sizes."
function sum_to_zero_by(z::AbstractVector, group::Vector{Int}, counts::Vector{Int})
    sums = [sum(z[group .== k]) for k in eachindex(counts)]
    return z .- (sums ./ counts)[group]
end

"""
    gap_effects(g::GapData; noise = StudentTNoise(4))

% gap to winner = race intercept + competitor effect + machine-season effect + noise,
with lapped finishers as interval-censored observations.

- Race intercepts `γ` (wide N(0, 5²) prior, in % units) absorb each race's
  reference point, so a dominant winner does not shift the whole field.
  `γ[race]` is the expected gap of that race's *average* entrant: row means
  are `γ[race] + c - mean(c over the race)` with `c = a[comp] + b[mach]` (see
  `gap_loglik`).
- Competitor effects sum to zero (relative to the average competitor).
- Machine-season effects sum to zero **within each season**: with race
  intercepts, shifting every car in a season and every race in that season in
  opposite directions leaves predictions unchanged, so only within-season car
  differences are identified. Competitors link seasons, so their effects stay
  comparable across careers.
- `σ_y` is learned. Scale priors are half-normal(2) in % units.
- `noise`: `StudentTNoise(4)` (default) or `NormalNoise()`; `σ_y` is the noise scale.
  Heavy tails matter here: with Gaussian noise, incident-hit results (a spin,
  a slow stop, a few laps down) inflated σ_y from 0.48% (timed rows only) to
  3.7% once lapped cars were included; Student-t(4) brings it to ~1.0%.
"""
@model function gap_effects(g::GapData, season_counts::Vector{Int}, noise::Noise)
    σ_comp ~ truncated(Normal(0, 2); lower = 0)
    σ_mach ~ truncated(Normal(0, 2); lower = 0)
    σ_y ~ truncated(Normal(0, 2); lower = 0)
    z_comp ~ filldist(Normal(), length(g.competitors))
    z_mach ~ filldist(Normal(), length(g.machines))
    γ ~ filldist(Normal(0, 5), length(g.races))
    @addlogprob! gap_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ_y, g, season_counts, noise)
end

season_counts(g::GapData) = [count(==(k), g.mach_season) for k in 1:maximum(g.mach_season)]
gap_effects(g::GapData; noise::Noise = StudentTNoise(4)) = gap_effects(g, season_counts(g), noise)

"""
    gap_sampler()

NUTS with **uncompiled** ReverseDiff. `gap_effects` relies on a hand-written
rule for its likelihood (`gap_loglik`), and a compiled ReverseDiff tape would
replay stale gradients from that rule.
"""
gap_sampler() = NUTS(0.8; adtype = AutoReverseDiff(; compile = false))

"""
    fit_gaps(g::GapData; noise=StudentTNoise(4), n_samples=1000, n_chains=1, ensemble=MCMCSerial(),
             sampler=gap_sampler(), rng=Random.default_rng(), kwargs...) -> Chains

Sample `gap_effects`. Chains start with race intercepts at each race's mean
timed gap and everything else near the prior centre (jittered per chain).
Progress and ensemble options as in `fit_effects`.
"""
function fit_gaps(g::GapData; noise::Noise = StudentTNoise(4), n_samples::Int = 1000, n_chains::Int = 1,
                  ensemble = MCMCSerial(),
                  sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                  progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100, kwargs...)
    adtype = hasproperty(sampler, :adtype) ? sampler.adtype : nothing
    adtype isa AutoReverseDiff && adtype.compile &&
        throw(ArgumentError("gap_effects needs uncompiled ReverseDiff (see gap_sampler); " *
                            "a compiled tape gives stale gradients for its custom rule"))
    race_mean = zeros(length(g.races))
    for r in eachindex(race_mean)
        ys = g.y[g.t_race .== r]
        race_mean[r] = isempty(ys) ? 0.0 : mean(ys)
    end
    scale() = 0.3 + 0.4 * rand(rng)
    inits = [InitFromParams((; σ_comp = scale(), σ_mach = scale(), σ_y = scale(),
                             z_comp = 0.1 .* randn(rng, length(g.competitors)),
                             z_mach = 0.1 .* randn(rng, length(g.machines)),
                             γ = race_mean .+ 0.1 .* randn(rng, length(race_mean))))
             for _ in 1:n_chains]
    return run_nuts(gap_effects(g; noise), inits; n_samples, n_chains, ensemble, sampler, rng,
                    progress, progress_log, log_every, kwargs...)
end

"""
    gap_effects_table(chain, g::GapData) -> (; competitors, machines)

Posterior summaries of competitor effects (sum to zero) and machine-season
effects (sum to zero within season), in % of race time; negative = faster.
With a pace scale (#15), competitor effects are in average-season units and
machine effects in their own season's units.
"""
function gap_effects_table(chain, g::GapData)
    draws(sym) = stack(vec(chain[sym]); dims = 1)
    centred_drivers = any(vn -> string(vn) == "a_comp", Turing.FlexiChains.parameters(chain))   # #28
    comp = centred_drivers ? mapslices(sum_to_zero, draws(:a_comp); dims = 2) :
        mapslices(sum_to_zero, draws(:z_comp); dims = 2) .* vec(chain[:σ_comp])
    counts = season_counts(g)
    centred = any(vn -> string(vn) == "b_mach", Turing.FlexiChains.parameters(chain))   # #15 models
    mach = centred ? mapslices(b -> sum_to_zero_by(b, g.mach_season, counts), draws(:b_mach); dims = 2) :
        mapslices(z -> sum_to_zero_by(z, g.mach_season, counts), draws(:z_mach); dims = 2) .* vec(chain[:σ_mach])
    names = Dict(zip(g.rows.competitor_id, g.rows.competitor_name))
    all_comp = vcat(g.t_comp, g.c_comp)
    all_mach = vcat(g.t_mach, g.c_mach)
    summarise(draws, labels, idx) = DataFrame(
        label = labels,
        mean = vec(mean(draws; dims = 1)),
        sd = vec(std(draws; dims = 1)),
        q05 = [quantile(c, 0.05) for c in eachcol(draws)],
        q95 = [quantile(c, 0.95) for c in eachcol(draws)],
        n_obs = [count(==(i), idx) for i in eachindex(labels)],
    )
    competitors = summarise(comp, g.competitors, all_comp)
    insertcols!(competitors, 2, :name => [names[c] for c in competitors.label])
    machines = summarise(mach, g.machines, all_mach)
    return (; competitors = sort!(competitors, :mean), machines = sort!(machines, :mean))
end
