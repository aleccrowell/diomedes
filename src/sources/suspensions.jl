# Red-flag suspensions (#17).
#
# Since 2005, a red-flagged race is resumed behind the safety car and the
# official race time includes the suspension. Canada 2011's winner's time is
# 244.7 min, of which ~123 min was the stoppage. Until 2001, stopped races were
# scored on aggregate times of their parts, so the stoppage is excluded, and no
# suspended-and-resumed races occur in 2002–2004. Checked against the official
# times: for every suspension detected from lap times (2007–2017 in the Ergast
# data), the winner's official total minus laps × typical lap matches the
# lap-time suspension to within ~1–2 min.
#
# Detection is per car: every car running at the time carries the stoppage in
# exactly one of its laps, whatever its lap number. For each car, laps over 3×
# that car's median lap count as stoppage, by their excess over the median.
# The suspension is the median of these per-car totals over cars that completed
# ≥ 90% of the laps (all on track at the stoppage; the median ignores one-off
# long pit stops). Summing handles two stoppages in one race (Brazil 2016).
#
# A per-lap median across cars (tried first) misses stoppages when the field is
# spread over several lap numbers: at Monaco 2011, cars were on laps 68–72 when
# the red flag came. Validation on the Ergast lap times (1996–2019): the per-car
# method flags exactly the 12 races listed as red-flagged and resumed in
# 2005–2019 (List of red-flagged Formula One races, Wikipedia), and no others.
# Durations match the excess in the official times to within ~1–2 min.
#
# Explicit red-flag timestamps exist only from 2018, in F1's live-timing
# archive (TrackStatus, code 5 = red). Ergast and Jolpica carry no flag data.

const SUSPENSION_FACTOR = 3.0

"""
    suspension_ms(car_laps) -> Float64

Total suspended time in a race, from `car_laps`: a collection of per-car
vectors of lap times (ms). Returns 0.0 if no stoppage is found.
"""
function suspension_ms(car_laps)
    cars = [v for v in car_laps if !isempty(v)]
    isempty(cars) && return 0.0
    L = maximum(length, cars)
    excess = [sum((t - m for t in v if t > SUSPENSION_FACTOR * m); init = 0.0)
              for v in cars if length(v) >= 0.9L for m in (median(v),)]
    return isempty(excess) ? 0.0 : median(excess)
end

"""
    suspect_suspension(winner_ms, laps, fastest_lap_ms) -> Bool

Cheap screen for sources without lap times in the results: flags a race whose
winner's average lap is more than 1.25× the fastest lap. Normal races run at
~1.05–1.25× (safety cars included), so this only picks candidates, whose lap
times are then fetched and checked with `suspension_ms`.
"""
suspect_suspension(winner_ms, laps, fastest_lap_ms) =
    !ismissing(winner_ms) && !ismissing(fastest_lap_ms) && laps > 0 &&
    winner_ms / laps > 1.25 * fastest_lap_ms

"Parse a lap time string like \"1:16.956\" or \"16.956\" into milliseconds."
function laptime_ms(s::AbstractString)
    parts = split(s, ":")
    secs = parse(Float64, last(parts))
    mins = length(parts) > 1 ? parse(Int, parts[end - 1]) : 0
    hours = length(parts) > 2 ? parse(Int, parts[end - 2]) : 0
    return 1000 * (3600hours + 60mins + secs)
end
