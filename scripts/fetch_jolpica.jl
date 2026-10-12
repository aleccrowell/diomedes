# Download (and cache) F1 data from Jolpica for the 2025 refits (#55).
#
#   julia --project scripts/fetch_jolpica.jl [laps_from=2020] [laps_to=2025]
#
# One process, one rate limiter at 7.5 s per request (~480/hour, under Jolpica's
# 500/hour), so concurrent downloads can't exceed the limit. Fetches results for
# 1950-<laps_to> (mostly cached), then lap times for <laps_from>-<laps_to>.
# Everything goes to the response cache (data/cache/jolpica); progress is printed
# per season.

using Diomedes, Dates

from = parse(Int, get(ARGS, 1, "2020"))
to = parse(Int, get(ARGS, 2, "2025"))
src = JolpicaF1(limiter = Diomedes.RateLimiter(7.5))
for s in 1950:to
    n = size(fetch_results(src, [s]), 1)
    println(now(), "  results $s: $n rows"); flush(stdout)
end
for s in from:to
    t = @elapsed n = size(fetch_laps(src, [s]), 1)
    println(now(), "  laps $s: $n laps in $(round(t / 60; digits = 1)) min"); flush(stdout)
end
println(now(), "  done")
