# Match Jolpica race results to F1DB race entries (#95), for each entry's chassis
# constructor and engine manufacturer, which Jolpica doesn't have.
#
#   julia --project scripts/cars/match_f1db.jl <f1db csv dir> [out=reference/cars/entries.csv] [to=2026]
#
# <f1db csv dir>: the unzipped `f1db-csv.zip` of an F1DB release
# (github.com/f1db/f1db). Within each race, a Jolpica row matches the F1DB entries
# whose driver surname is in its name; ties (two drivers of one surname, a driver
# in two cars) are broken by first name, then laps completed. Rows without a
# unique match are reported, not written.

using Diomedes, DataFrames, CSV, Unicode

dir = ARGS[1]
opts = Dict(split(a, "="; limit = 2) for a in ARGS[2:end] if occursin("=", a))
out = get(opts, "out", "reference/cars/entries.csv")
to = parse(Int, get(opts, "to", "2026"))

jol = fetch_results(JolpicaF1(), 1950:to)
f1 = CSV.read(joinpath(dir, "f1db-races-race-results.csv"), DataFrame)
drv = CSV.read(joinpath(dir, "f1db-drivers.csv"), DataFrame)
lastname = Dict(r.id => r.lastName for r in eachrow(drv))

fold(s) = lowercase(filter(isletter, Unicode.normalize(String(s); stripmark = true)))
first_name = Dict(r.id => fold(r.firstName) for r in eachrow(drv))
# names Jolpica gives differently: the F1DB driver id => a folded part of the Jolpica name
const ALIAS = Dict("birabongse-bhanudej" => "bira")
# the F1DB surname, or any part of a double-barrelled one (Giraud-Cabantous, Schwelm Cruz, Sainz Jr.)
function surname_in(id, name)
    haskey(ALIAS, id) && return occursin(ALIAS[id], name)
    occursin(fold(lastname[id]), name) && return true
    parts = filter(p -> length(p) >= 4, fold.(split(lastname[id], r"[ -]")))
    return any(p -> occursin(p, name), parts)
end
byrace = Dict{Tuple{Int,Int},Vector{Int}}()
for (i, r) in enumerate(eachrow(f1)); push!(get!(byrace, (r.year, r.round), Int[]), i); end

rows = NamedTuple[]
bad = String[]
used = Set{Int}()
for r in eachrow(jol)
    name = fold(r.competitor_name)
    cand = filter(i -> surname_in(f1.driverId[i], name), get(byrace, (r.season, r.round), Int[]))
    # same surname in one race (Hill, Schumacher) or a driver in two cars: first name, then laps
    length(cand) > 1 && (c = filter(i -> startswith(name, first_name[f1.driverId[i]]), cand); isempty(c) || (cand = c))
    length(cand) > 1 && !ismissing(r.laps) && (c = filter(i -> coalesce(f1.laps[i], -1) == r.laps, cand); isempty(c) || (cand = c))
    length(cand) > 1 && (c = filter(i -> !(i in used), cand); isempty(c) || (cand = c))
    # a driver in two cars of different makes (Moss 1961 Britain, Lotus and Ferguson): the make
    mk = replace(r.machine_id, "_" => "-")
    length(cand) > 1 && (c = filter(i -> startswith(mk, f1.constructorId[i]), cand); isempty(c) || (cand = c))
    # shared cars: several F1DB rows of the same driver, car and engine are equivalent here
    length(cand) > 1 && allequal((f1.constructorId[i], f1.engineManufacturerId[i]) for i in cand) && (cand = cand[1:1])
    if length(cand) != 1
        push!(bad, "$(length(cand)) F1DB rows: $(r.season)-$(r.round) $(r.competitor_name) ($(r.machine_id), laps $(r.laps))")
        continue
    end
    i = only(cand); push!(used, i)
    e = f1[i, :]
    push!(rows, (season = r.season, round = r.round, driver_id = r.competitor_id, machine_id = r.machine_id,
                 laps = r.laps, f1db_driver = e.driverId, f1db_constructor = e.constructorId,
                 f1db_engine = e.engineManufacturerId))
end
df = DataFrame(rows)
println("Jolpica rows $(nrow(jol)), matched $(nrow(df)), unmatched $(length(bad))")
# entries that never started (0 laps) aren't in F1DB's race results, and no model uses them
foreach(println, filter(b -> !occursin(r"laps (0|missing)\)", b), bad))
mkpath(dirname(out)); CSV.write(out, df)
