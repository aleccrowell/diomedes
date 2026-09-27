# On-disk cache for raw API responses, plus polite rate limiting.
#
# Responses are stored verbatim so a source's parsing can change without
# re-fetching. Location defaults to `<repo>/data/cache` and can be overridden
# with the DIOMEDES_CACHE environment variable.

default_cache_dir() = get(ENV, "DIOMEDES_CACHE", joinpath(dirname(@__DIR__), "data", "cache"))

"""
    RateLimiter(min_interval)

Ensures at least `min_interval` seconds between successive requests.
"""
mutable struct RateLimiter
    min_interval::Float64
    last::Float64
end
RateLimiter(min_interval::Real) = RateLimiter(Float64(min_interval), 0.0)

function wait!(rl::RateLimiter)
    dt = time() - rl.last
    dt < rl.min_interval && sleep(rl.min_interval - dt)
    rl.last = time()
    return nothing
end

# SHA-1 rather than `hash`, which is not stable across Julia versions.
cache_path(cache_dir, key) = joinpath(cache_dir, bytes2hex(sha1(key)) * ".json")

"""
    cached_get(url; cache_dir, limiter, refresh=false, max_retries=5)

GET `url` and return the body as a `String`, reading from / writing to the cache.
On HTTP 429 or 5xx it backs off exponentially, honouring `Retry-After` if given.
"""
function cached_get(url::AbstractString; cache_dir::AbstractString = default_cache_dir(),
                    limiter::RateLimiter, refresh::Bool = false, max_retries::Int = 5)
    path = cache_path(cache_dir, url)
    if !refresh && isfile(path)
        return read(path, String)
    end
    backoff = 2.0
    for attempt in 1:max_retries
        wait!(limiter)
        resp = HTTP.get(url; status_exception = false, read_idle_timeout = 60,
                        headers = ["User-Agent" => "Diomedes.jl"])
        if resp.status == 200
            body = String(resp.body)
            mkpath(cache_dir)
            write(path, body)
            return body
        elseif resp.status == 429 || resp.status >= 500
            ra = HTTP.header(resp, "Retry-After", "")
            delay = max(something(tryparse(Float64, ra), 0.0), backoff)   # Retry-After can be 0
            @warn "HTTP $(resp.status) from $url; retrying in $(delay)s" attempt
            sleep(delay)
            backoff *= 2
        else
            error("HTTP $(resp.status) fetching $url")
        end
    end
    error("giving up on $url after $max_retries attempts")
end

cached_json(url; kw...) = JSON3.read(cached_get(url; kw...))
