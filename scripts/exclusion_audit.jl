# Which rows the race model leaves out, and whether that leans towards the conditions
# or the drivers and cars it measures (#133).
#
#   julia --project scripts/exclusion_audit.jl [data=jolpica1950-2026+partial]
#
# Every result row of the data mode (as in scripts/gap_data.jl) is put in one class:
# used (timed, lapped interval, partial gap), or dropped (retired before a quarter of the
# distance or without lap times, disqualified, never started a lap, lapped but its laps
# don't match its status, in a race dropped whole). Then:
#   1. the classes by race condition (dry / mixed / wet, from the circuit table) and decade;
#   2. the races dropped whole;
#   3. per driver and per car-season, the dropped share against the race model's
#      effects (the published fit's competitor and machine tables, if in output/).
# The lap model's slow-lap filter is covered by #116 (keepwet); this audit is the race model.
include("gap_data.jl")
using CSV, Statistics, Printf

mode = (i = findfirst(a -> startswith(a, "data="), ARGS); i === nothing ? "jolpica1950-2026+partial" : String(split(ARGS[i], "=")[2]))
res, g = gap_data(mode)
used = g.rows
key(r) = (r.event_id, r.stage_id, r.competitor_id)
ukind = Dict(key(r) => (hasproperty(r, :partial) && r.partial ? :partial : r.kind) for r in eachrow(used))
races_used = Set(used.event_id)
finished(s) = s == "Finished" || occursin(r"^\+\d+ Laps?$", s) || startswith(s, "Lapped")
function class(r)
    k = key(r)
    haskey(ukind, k) && return ukind[k]
    hasproperty(r, :partial) && r.partial && return :partial_unused
    r.event_id in races_used || return :race_dropped
    s = r.status
    occursin(r"^Disqualified|^Excluded", s) && return :disqualified
    (ismissing(r.laps) || r.laps == 0) && return :no_laps
    finished(s) && return :lapped_unmatched
    return :retired
end
df = DataFrame(res)        # one row per start: a retirement used as a partial gap is marked in place
df.class = [class(r) for r in eachrow(df)]
ct = circuit_table()
cond = Dict(ct.event_id .=> ct.condition)
df.condition = [get(cond, e, "unknown") for e in df.event_id]
df.decade = 10 .* (df.season .÷ 10)

classes = [:timed, :lapped, :partial, :retired, :no_laps, :disqualified, :lapped_unmatched, :race_dropped, :partial_unused]
function table(by)
    println("\n", rpad(string(by), 10), lpad("rows", 7), join(lpad.(string.(classes), 17)))
    for gdf in groupby(sort(df, by), by)
        n = nrow(gdf)
        @printf("%-10s%7d", string(gdf[1, by]), n)
        for c in classes; @printf("%16.1f%%", 100count(==(c), gdf.class) / n); end
        println()
    end
end
println("$(nrow(df)) starts in $(length(unique(df.event_id))) races; used $(count(in((:timed, :lapped, :partial)), df.class))")
table(:condition); table(:decade)
# within each decade, dropped share in wet / mixed races against dry ones
println("\ndropped share (not timed, lapped or partial) by decade and condition:")
for gdf in groupby(sort(df, :decade), :decade)
    f(c) = (x = gdf[gdf.condition .== c, :]; nrow(x) == 0 ? NaN : 100 * mean(.!in.(x.class, Ref((:timed, :lapped, :partial)))))
    @printf("  %d  dry %5.1f %%  mixed %5.1f %%  wet %5.1f %%\n", gdf.decade[1], f("dry"), f("mixed"), f("wet"))
end
dropped = unique(df[df.class .== :race_dropped, [:event_id, :event_name]])
println("\nraces dropped whole ($(nrow(dropped))): ", join(["$(r.event_id) $(r.event_name)" for r in eachrow(dropped)], "; "))

# per driver and car-season: dropped share against the published effects
dropped_share(gdf) = mean(.!in.(gdf.class, Ref((:timed, :lapped, :partial))))
for (col, file, lab) in ((:competitor_id, "competitor_effects", "drivers"), (:machine_key, "machine_effects", "car-seasons"))
    col === :machine_key && (df.machine_key = [string(r.machine_id, "_", r.season) for r in eachrow(df)])
    f = "output/gap_$(mode)_pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc_$(file).csv"
    isfile(f) || (println("\n(no $f: skipping $lab)"); continue)
    eff = CSV.read(f, DataFrame; stringtype = String)
    s = combine(groupby(df, col), nrow => :starts, AsTable(:) => (t -> dropped_share(DataFrame(t))) => :dropped)
    j = innerjoin(s, select(eff, :label => col, :mean); on = col)
    j = j[j.starts .>= 20, :]
    @printf("\n%s with ≥ 20 starts (%d): corr(dropped share, effect) %.3f (positive: slower ones lose more rows)\n",
            lab, nrow(j), cor(j.dropped, j.mean))
    for (lo, hi) in ((-Inf, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, Inf))
        x = j[(j.mean .>= lo) .& (j.mean .< hi), :]
        nrow(x) > 0 && @printf("  effect in [%5.1f, %5.1f): %4d, mean dropped share %.1f %%\n", lo, hi, nrow(x), 100mean(x.dropped))
    end
end

# the retirement model (prepare_retirements): starts with unknown laps, non-starters and
# disqualifications are dropped; retirements of unknown cause stay in (cause -1)
println("\nretirement model, by condition: dropped (unknown laps, non-start, excluded) and kept with unknown cause")
rr = fetch_results(JolpicaF1(), parse(Int, match(r"(\d{4})-(\d{4})", mode)[1]):parse(Int, match(r"(\d{4})-(\d{4})", mode)[2]))
rr.cause = [ismissing(r.laps) ? :laps_unknown : Diomedes.retirement_cause(r.status, r.laps) for r in eachrow(rr)]
rr.condition = [get(cond, e, "unknown") for e in rr.event_id]
for gdf in groupby(sort(rr, :condition), :condition)
    n = nrow(gdf)
    @printf("  %-8s %6d starts: unknown laps %4.1f %%, non-start %4.1f %%, excluded %4.1f %%; kept with unknown cause %4.1f %%\n",
            gdf.condition[1], n, 100count(==(:laps_unknown), gdf.cause) / n, 100count(==(:nonstart), gdf.cause) / n,
            100count(==(:excluded), gdf.cause) / n, 100count(==(:unknown), gdf.cause) / n)
end
