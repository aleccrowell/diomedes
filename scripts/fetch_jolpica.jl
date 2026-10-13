# Download (and cache) F1 data from Jolpica for the 2025 refits (#55).
#
#   julia --project scripts/fetch_jolpica.jl [laps_from=2020] [laps_to=2025]
#
# One process, one rate limiter at 9 s per request (~400/hour, under Jolpica's
# 500/hour; don't query the API from elsewhere while it runs). A request that keeps
# getting HTTP 429 after the cache layer's retries makes the season wait 10 minutes
# and resume (responses already fetched are cached, so nothing is repeated). Fetches results for
# 1950-<laps_to> (mostly cached), then lap times for <laps_from>-<laps_to>.
# Everything goes to the response cache (data/cache/jolpica); progress is printed
# per season.

using Diomedes, Dates

from = parse(Int, get(ARGS, 1, "2020"))
to = parse(Int, get(ARGS, 2, "2025"))
src = JolpicaF1(limiter = Diomedes.RateLimiter(9.0))
function patiently(f, what)
    for attempt in 1:12
        try
            return f()
        catch err
            println(now(), "  $what failed ($(first(sprint(showerror, err), 120))); waiting 10 min, attempt $attempt")
            flush(stdout)
            sleep(600)
        end
    end
    error("$what: giving up after 12 attempts")
end
for s in 1950:to
    n = patiently(() -> size(fetch_results(src, [s]), 1), "results $s")
    println(now(), "  results $s: $n rows"); flush(stdout)
end
for s in from:to
    t = @elapsed n = patiently(() -> size(fetch_laps(src, [s]), 1), "laps $s")
    println(now(), "  laps $s: $n laps in $(round(t / 60; digits = 1)) min"); flush(stdout)
end
println(now(), "  done")
