# Partial-distance gaps of retired cars, for the selection check of #13.
#
# The gap models drop retirements. If a car's chance of retiring depends on how
# it was running in that race (not only on its driver and car effects), the
# finishers are a selected sample and the effects are biased. Lap times (Ergast,
# 1996 on) show how a retiree was running before it stopped.
#
# A retiree that completed n of L laps has a partial gap to the eventual winner
# at lap n: y(n) = 100·log(C_i(n) / C_w(n)), with C the cumulative lap time.
# Partial gaps are not on the scale of finishing gaps: gaps change over a race
# (start losses, safety cars bunching the field), and for F1 lead-lap finishers
# the gap at half distance is ~0.3 points larger than at the flag on average.
# Each partial gap is therefore calibrated in its own race: it is shifted by the
# median change, from lap n to the flag, of the timed finishers' gaps,
#     ŷ = y(n) + median_j (y_j(L) - y_j(n)),
# an estimate of the finishing gap the car would have had at its pace so far.

"""
    ergast_cumulative_laps(dir) -> Dict{Tuple{String,String},Vector{Float64}}

Cumulative lap times (ms) per (event_id, driverRef) from Ergast `lap_times.csv`;
element n is the time at the end of lap n. Ids as in `ErgastCSV`.
"""
function ergast_cumulative_laps(dir::AbstractString)
    rd(name, cols) = CSV.read(joinpath(dir, name), DataFrame; select = cols, missingstring = "\\N")
    lt = rd("lap_times.csv", [:raceId, :driverId, :lap, :milliseconds])
    races = rd("races.csv", [:raceId, :year, :round])
    drivers = rd("drivers.csv", [:driverId, :driverRef])
    ev = Dict(r.raceId => string(r.year, "-", lpad(r.round, 2, '0')) for r in eachrow(races))
    dref = Dict(r.driverId => String(r.driverRef) for r in eachrow(drivers))
    sort!(lt, [:raceId, :driverId, :lap])
    out = Dict{Tuple{String,String},Vector{Float64}}()
    for g in groupby(lt, [:raceId, :driverId])
        # a car's laps are 1..n without gaps; anything else is unusable
        g.lap == 1:nrow(g) || continue
        out[(ev[first(g.raceId)], dref[first(g.driverId)])] = cumsum(Float64.(g.milliseconds))
    end
    return out
end

"""
    partial_gaps(results, cum; min_frac = 0.25) -> DataFrame

Calibrated partial-distance gaps (see the file header) of the retirements in
`results` (an Ergast-sourced results table) with lap data in `cum` (see
`ergast_cumulative_laps`). A retirement counts if its cause is not a non-start
or a disqualification (see `retirement_cause`) and it completed at least
`min_frac` of the race distance (and 2 laps). Columns: `event_id`,
`competitor_id`, `n`, `L`, `y_n` (raw partial gap), `shift`, `y_hat`, `cause`,
`n_ref` (finishers the shift is the median over).
"""
function partial_gaps(results::AbstractDataFrame, cum; min_frac::Real = 0.25)
    out = DataFrame(event_id = String[], competitor_id = String[], n = Int[], L = Int[], y_n = Float64[],
                    shift = Float64[], y_hat = Float64[], cause = Symbol[], n_ref = Int[])
    for g in groupby(results, :event_id)
        timed = findall(!ismissing, g.time_ms)
        isempty(timed) && continue
        w = timed[argmin(g.time_ms[timed])]
        Cw = get(cum, (g.event_id[w], g.competitor_id[w]), nothing)
        L = g.laps[w]
        (Cw === nothing || ismissing(L) || length(Cw) < L) && continue
        # timed finishers with full lap data: gap at the flag and per lap
        refs = [cum[(g.event_id[j], g.competitor_id[j])] for j in timed
                if j != w && length(get(cum, (g.event_id[j], g.competitor_id[j]), Float64[])) >= L]
        isempty(refs) && continue
        for i in eachindex(g.status)
            cause = retirement_cause(g.status[i], g.laps[i])
            (cause in RETIRE_CAUSES || cause === :other) || continue
            n = coalesce(g.laps[i], 0)
            (n >= max(2, ceil(Int, min_frac * L)) && n < L) || continue
            Ci = get(cum, (g.event_id[i], g.competitor_id[i]), nothing)
            (Ci === nothing || length(Ci) < n) && continue
            y_n = 100 * log(Ci[n] / Cw[n])
            shift = median(100 * (log(C[L] / Cw[L]) - log(C[n] / Cw[n])) for C in refs)
            push!(out, (g.event_id[i], g.competitor_id[i], n, L, y_n, shift, y_n + shift, cause, length(refs)))
        end
    end
    return out
end

"""
    with_partial_rows(results, pg) -> DataFrame

`results` with the retirements in `pg` (see `partial_gaps`) turned into timed
rows at their calibrated gap `y_hat`: `time_ms` is set so that `prepare_gaps`
gives them y = y_hat (racing time = winner's racing time · exp(y_hat/100), plus
the race's suspension), and `status` to "Partial". A Bool column `partial`
marks them.
"""
function with_partial_rows(results::AbstractDataFrame, pg::AbstractDataFrame)
    df = copy(results)
    df.partial = falses(nrow(df))
    yhat = Dict((r.event_id, r.competitor_id) => r.y_hat for r in eachrow(pg))
    for g in groupby(df, :event_id)
        timed = findall(!ismissing, g.time_ms)
        isempty(timed) && continue
        susp = coalesce(first(g.suspended_ms), 0.0)
        Tw = minimum(g.time_ms[timed]) - susp            # winner's racing time
        for i in eachindex(g.status)
            y = get(yhat, (g.event_id[i], g.competitor_id[i]), nothing)
            y === nothing && continue
            g.time_ms[i] = Tw * exp(y / 100) + susp
            g.status[i] = "Partial"
            g.partial[i] = true
        end
    end
    return df
end

"Whether each GapData row (timed, then lapped) came from a partial gap (see `with_partial_rows`)."
partial_mask(g::GapData) = vcat(Bool.(g.rows.partial[g.rows.kind .== :timed]), falses(length(g.lo)))
