# Data check for #113: do teammates differ in their maximum pace, or in how
# consistently they reach a shared limit?
#
#   julia --project scripts/peak_check.jl [<first>-<last>]     (default 2011-2026)
#
# Clean laps (`prepare_laps`, recorded stops, no per-car size cut) minus a per-race
# fit of fuel/track evolution and stint age (shared by the field), pooled per
# driver and car-season over the races both teammates ran. For each teammate pair,
# Δq_p = q_p(driver) - q_p(teammate) at several quantiles p of their residuals.
# If the pair differs only by a shift of the whole distribution (max pace), every
# Δq_p equals Δq50; if only in a shortfall from a shared edge (consistency), the
# fast quantiles differ less. R(p) = the share of the median difference seen at
# quantile p, from a split-half regression (odd races give the regressor, even
# races the outcome) so noise in Δq50 doesn't shrink it. Reference curves come
# from simulating the current lap noise (probe 108f) with a shift, a higher
# shortfall frequency or a larger shortfall.
#
# Traffic: the gap on track to the car ahead and behind, at the line at the start
# of the lap, from cumulative lap times (all cars, lapped ones too). The check is
# run on all laps and on free-air laps only (gap ahead over FREE_AIR seconds).
#
# Writes output/peak_check_<first>-<last>_pairs.csv (per pair-season) and prints R(p) and
# the split-half reliability of the pairs' spread differences (consistency apart from pace).

using Diomedes, DataFrames, CSV, Statistics, Random, LinearAlgebra

const QS = [0.05, 0.10, 0.25, 0.50, 0.75]
const FREE_AIR = 2.0          # s
const MIN_LAPS = 15           # clean laps per driver per race for the race to count

span = get(ARGS, 1, "2011-2026")
first_s, last_s = parse.(Int, split(span, "-"))
seasons = first_s:last_s
jolpica = JolpicaF1()
res = fetch_results(jolpica, collect(seasons))
laps = fetch_laps(jolpica, collect(seasons))
pits = fetch_pit_stops(jolpica, collect(seasons))
d = prepare_laps(laps, res; pits = :recorded, pit_stops = pits, stoppage = false)
println(d); flush(stdout)

# gaps on track at the line: each car's crossing times, every car in the race
function track_gaps(laps)
    out = Dict{Tuple{String,String,Int},Tuple{Float64,Float64}}()   # (event, car, lap) => (ahead, behind) at its start
    for ev in groupby(laps, :event_id)
        cross = Tuple{Float64,String,Int}[]                          # (time, car, lap ending there)
        for car in groupby(ev, :competitor_id)
            car = sort(car, :lap)
            car.lap == 1:nrow(car) || continue
            t = cumsum(car.time_ms) ./ 1000
            append!(cross, zip(t, car.competitor_id, car.lap))
        end
        sort!(cross)
        for (i, (t, c, l)) in enumerate(cross)
            j = i - 1
            while j >= 1 && cross[j][2] == c; j -= 1; end
            k = i + 1
            while k <= length(cross) && cross[k][2] == c; k += 1; end
            ahead = j >= 1 ? t - cross[j][1] : Inf
            behind = k <= length(cross) ? cross[k][1] - t : Inf
            out[(first(ev.event_id), c, l + 1)] = (ahead, behind)     # the gaps at the start of lap l + 1
        end
    end
    return out
end
gaps = track_gaps(laps)
rows = copy(d.rows)
g = [get(gaps, (r.event_id, r.competitor_id, r.lap), (Inf, Inf)) for r in eachrow(rows)]
rows.ahead = first.(g)
rows.behind = last.(g)

# per-race fit of the shared terms: intercept, race fraction, stint age (quadratic),
# on the laps without a big loss (y < 3 %), then residuals for all
rows.res = zeros(nrow(rows))
for gr in groupby(rows, :event_id)
    s = gr.stint ./ 10
    X = hcat(ones(nrow(gr)), gr.frac, s, s .^ 2)
    ok = gr.y .< 3
    β = X[ok, :] \ gr.y[ok]
    gr.res .= gr.y .- X * β
end
println("free-air share of laps: ", round(mean(rows.ahead .> FREE_AIR); digits = 3))

# teammate pairs per car-season, over the races where both have MIN_LAPS clean laps
function pair_table(rows, keep)
    out = DataFrame([:machine => String[], :season => Int[], :drv => String[], :mate => String[], :half => Int[],
                     :n => Int[], :nm => Int[], (Symbol("dq", round(Int, 100p)) => Float64[] for p in QS)...])
    sub = rows[keep, :]
    for car in groupby(sub, :machine_key)
        drivers = unique(car.competitor_id)
        for i in eachindex(drivers), j in eachindex(drivers)
            i < j || continue
            a, b = drivers[i], drivers[j]
            races = String[]
            for ev in groupby(car, :event_id)
                count(==(a), ev.competitor_id) >= MIN_LAPS && count(==(b), ev.competitor_id) >= MIN_LAPS &&
                    push!(races, first(ev.event_id))
            end
            length(races) >= 4 || continue
            sort!(races)
            for h in 0:1
                rs = Set(races[(1 + h):2:end])
                ra = car.res[(car.competitor_id .== a) .& in.(car.event_id, Ref(rs))]
                rb = car.res[(car.competitor_id .== b) .& in.(car.event_id, Ref(rs))]
                push!(out, (first(car.machine_key), first(car.season), a, b, h, length(ra), length(rb),
                            (quantile(ra, p) - quantile(rb, p) for p in QS)...))
            end
        end
    end
    return out
end

# R(p): slope of Δq_p (even races) on Δq50 (odd races) over the slope of Δq50 (even) on it
function share_curve(pt)
    w = unstack(select(pt, :machine, :drv, :mate, :half, r"^dq"), [:machine, :drv, :mate], :half, :dq50;
                renamecols = h -> "m$h")
    w = dropmissing(w)
    x = Float64.(w.m0)
    x .-= mean(x)
    R = Float64[]
    den = sum(x .* (w.m1 .- mean(w.m1)))
    for p in QS
        k = Symbol("dq", round(Int, 100p))
        e = pt[pt.half .== 1, [:machine, :drv, :mate, k]]
        y = leftjoin(w[:, [:machine, :drv, :mate]], e; on = [:machine, :drv, :mate], order = :left)[!, k]
        push!(R, sum(x .* (y .- mean(y))) / den)
    end
    return R, nrow(w), sqrt(mean(abs2, w.m0))
end

# bootstrap over pair-seasons for an interval on R(p)
function share_boot(pt; B = 500, rng = MersenneTwister(113))
    keys_ = unique(pt[:, [:machine, :drv, :mate]])
    Rs = zeros(B, length(QS))
    for b in 1:B
        pick = keys_[rand(rng, 1:nrow(keys_), nrow(keys_)), :]
        pick.rep = 1:nrow(pick)
        bt = innerjoin(pick, pt; on = [:machine, :drv, :mate])
        bt.machine = string.(bt.machine, "#", bt.rep)          # resampled pairs stay distinct
        Rs[b, :] = share_curve(bt)[1]
    end
    return [quantile(Rs[:, k], 0.05) for k in eachindex(QS)], [quantile(Rs[:, k], 0.95) for k in eachindex(QS)]
end

# reference: the current lap noise (probe 108f: σ 0.48 %, π 0.36, λ 0.85 %) with a
# driver who is slower by Δ through a shift, a higher shortfall frequency or a larger shortfall
function reference_curves(; σ = 0.48, π = 0.36, λ = 0.85, Δ = 0.1, n = 2_000_000, rng = MersenneTwister(1))
    draw(π_, λ_, sh) = sh .+ σ .* randn(rng, n) .+ (rand(rng, n) .< π_) .* (-λ_ .* log.(rand(rng, n)))
    base = draw(π, λ, 0.0)
    qb = quantile(base, QS)
    π2 = π + Δ / λ                      # same mean loss through frequency
    λ2 = λ + Δ / π                      # or through size
    cur(x) = (quantile(x, QS) .- qb) ./ (quantile(x, 0.5) - qb[4])
    return (shift = cur(draw(π, λ, Δ)), frequency = cur(draw(π2, λ, 0.0)), size = cur(draw(π, λ2, 0.0)))
end

# Consistency apart from pace: does a pair's difference in spread repeat across the two
# halves of their races? Split-half correlation of Δ(q75 - q10), its upper (q75 - q50) and
# lower (q50 - q10) parts, against the same for the median and the edge (q10).
function spread_reliability(pt)
    stats = ["spread q75-q10" => (r -> r.dq75 - r.dq10), "upper q75-q50" => (r -> r.dq75 - r.dq50),
             "lower q50-q10" => (r -> r.dq50 - r.dq10), "median" => (r -> r.dq50), "edge q10" => (r -> r.dq10)]
    h0 = pt[pt.half .== 0, :]
    h1 = leftjoin(h0[:, [:machine, :drv, :mate]], pt[pt.half .== 1, :]; on = [:machine, :drv, :mate], order = :left)
    for (name, f) in stats
        a, b = f.(eachrow(h0)), f.(eachrow(h1))
        println("  ", rpad(name, 16), " split-half r = ", round(cor(a, b); digits = 2),
                ", sd ", round(std(vcat(a, b)); digits = 3), " %")
    end
    println("  cor(spread, median), across halves: ",
            round(cor((r -> r.dq75 - r.dq10).(eachrow(h0)), h1.dq50); digits = 2))
end

fmt(v) = join((lpad(string(round(x; digits = 2)), 6) for x in v), " ")
println("\nquantiles p:         ", fmt(QS))
ref = reference_curves()
println("reference  shift:     ", fmt(ref.shift))
println("reference  frequency: ", fmt(ref.frequency))
println("reference  size:      ", fmt(ref.size))
mkpath("output")
for (name, keep) in [("all laps", trues(nrow(rows))), ("free air", rows.ahead .> FREE_AIR)]
    pt = pair_table(rows, keep)
    R, np, sd50 = share_curve(pt)
    lo, hi = share_boot(pt)
    println("\n$name: $np pair-seasons, sd of Δq50 $(round(sd50; digits = 3)) %")
    println("R(p):                 ", fmt(R))
    println("  90% interval        ", fmt(lo))
    println("                      ", fmt(hi))
    spread_reliability(pt)
    name == "all laps" && CSV.write("output/peak_check_$(span)_pairs.csv", pt)
    flush(stdout)
end
