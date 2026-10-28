# Per-lap context for the lap model (#113): traffic and field pace, kept out of
# `LapData` (which chain checkpoints embed, so new fields would make old fits
# unreadable) and built from it, as `RaceWalk` is.

"""
    LapTraffic(d::LapData, laps)

Traffic around each of `d`'s laps (#113), at the line at the start of the lap (the
end of the car's previous lap), from cumulative lap times in `laps` (`LAP_SCHEMA`;
every car of the race, including the laps `prepare_laps` left out):

- on track (dirty air, blue flags): `ahead` and `behind`, seconds to the last car to
  cross the line before this one and the next after it, whatever their lap; and
  `ahead_laps`, `behind_laps`, that car's completed laps minus this car's (`ahead_laps
  < 0`: catching a lapped car; `behind_laps > 0`: being lapped by it);
- in the race (pace management): `front` and `back`, seconds to the cars just ahead
  and behind in race position, i.e. the previous and next to complete the same number
  of laps (a lapped car behind is reached once it completes that lap).

`Inf` when there is no such car (the leader's `front`, the first car to cross). The
line crossings miss whatever happens on the lap (a pass, a car pitting), and the
crossing times include the grid offset of lap 1. A car with gaps in its lap record
is left out of the crossings, as `prepare_laps` leaves out its laps.
"""
struct LapTraffic
    ahead::Vector{Float64}
    behind::Vector{Float64}
    ahead_laps::Vector{Int}
    behind_laps::Vector{Int}
    front::Vector{Float64}
    back::Vector{Float64}
end

function LapTraffic(d::LapData, laps::AbstractDataFrame)
    # per race: line crossings (time s, car, laps completed), in time order
    at = Dict{Tuple{String,String,Int},NTuple{6,Float64}}()      # (event, car, laps done) => gaps
    for ev in groupby(laps, :event_id)
        cross = Tuple{Float64,String,Int}[]
        for car in groupby(ev, :competitor_id)
            car = sort(car, :lap)
            car.lap == 1:nrow(car) || continue
            append!(cross, zip(cumsum(car.time_ms) ./ 1000, car.competitor_id, car.lap))
        end
        sort!(cross)
        bylap = Dict{Int,Vector{Int}}()                              # laps done => crossings, in time order
        for (i, (_, _, l)) in enumerate(cross); push!(get!(bylap, l, Int[]), i); end
        e = first(ev.event_id)
        for (l, idx) in bylap, (p, i) in enumerate(idx)
            t, c, _ = cross[i]
            j = i - 1
            while j >= 1 && cross[j][2] == c; j -= 1; end
            k = i + 1
            while k <= length(cross) && cross[k][2] == c; k += 1; end
            at[(e, c, l)] = (j >= 1 ? t - cross[j][1] : Inf, k <= length(cross) ? cross[k][1] - t : Inf,
                             j >= 1 ? cross[j][3] - l : 0, k <= length(cross) ? cross[k][3] - l : 0,
                             p > 1 ? t - cross[idx[p - 1]][1] : Inf, p < length(idx) ? cross[idx[p + 1]][1] - t : Inf)
        end
    end
    g = [get(at, (r.event_id, r.competitor_id, r.lap - 1), (Inf, Inf, 0, 0, Inf, Inf)) for r in eachrow(d.rows)]
    return LapTraffic([x[1] for x in g], [x[2] for x in g], [Int(x[3]) for x in g], [Int(x[4]) for x in g],
                      [x[5] for x in g], [x[6] for x in g])
end

"""
    field_pace(d::LapData; window = 2, min_cars = 5) -> Vector{Float64}

Per-lap conditions index (#113): the field's median clean lap on each lap of a race,
in % of the race's median clean lap (`d.y`'s scale), averaged over the laps within
±`window` that have at least `min_cars` clean laps, then with its linear trend over
the race distance taken out within each race (the model's race slope δ takes that).
What is left is the race's pace on that lap beyond its mean and trend: a drying or
wetting track, a cooling evening, a red-flag restart on fresh tyres. It also holds
the field's tyre-age mix (stop windows) and each car's own lap, one of the field.
Race-wide slow laps (`SLOW_RACE`) are not in `d`, so the wettest laps are missing.
A lap with no valid neighbour gets 0.
"""
function field_pace(d::LapData; window::Int = 2, min_cars::Int = 5)
    fp = zeros(length(d.y))
    for r in eachindex(d.races)
        js = findall(==(r), d.race)
        laps = d.rows.lap[js]
        ys = d.y[js]
        med = Dict(l => median(ys[laps .== l]) for l in unique(laps) if count(==(l), laps) >= min_cars)
        sm = Dict{Int,Float64}()
        for l in unique(laps)
            v = [med[k] for k in (l - window):(l + window) if haskey(med, k)]
            isempty(v) || (sm[l] = mean(v))
        end
        x = [get(sm, l, NaN) for l in laps]
        ok = .!isnan.(x)
        count(ok) >= 2 || continue
        X = hcat(ones(count(ok)), d.frac[js][ok])
        β = X \ x[ok]
        fp[js[ok]] .= x[ok] .- X * β
    end
    return fp
end
