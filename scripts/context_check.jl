# Data check for #113 step 2: how much traffic and field pace move lap times, to choose
# their form in the model.
#
#   julia --project scripts/context_check.jl [<first>-<last>]     (default 2011-2026)
#
# Clean laps (`prepare_laps`, recorded stops, no per-car size cut) minus a per-race fit
# of fuel/track evolution and stint age shared by the field (as peak_check.jl), then
# minus each car's mean over its clean laps in that race, so a bin's mean is the change
# against the same driver's other laps that race. Laps over 3 % (big losses) are left
# out of the means. Prints the mean and count per bin of:
# - the gap on track to the car ahead, by whether it is on the same lap, a lapped car
#   or a car laps up;
# - the gap on track to the car behind when it is lapping this car (blue flags);
# - the gaps to the cars ahead and behind in race position (pace management);
# and the spread of `field_pace` per race by track condition (dry, mixed, wet).

using Diomedes, DataFrames, Statistics, Printf

span = get(ARGS, 1, "2011-2026")
first_s, last_s = parse.(Int, split(span, "-"))
seasons = first_s:last_s
jolpica = JolpicaF1()
res = fetch_results(jolpica, collect(seasons))
laps = fetch_laps(jolpica, collect(seasons))
pits = fetch_pit_stops(jolpica, collect(seasons))
circ = circuit_table()
d = prepare_laps(laps, res; pits = :recorded, pit_stops = pits, stoppage = false, circuits = circ)
println(d); flush(stdout)
tr = LapTraffic(d, laps)
fp = field_pace(d)

rows = copy(d.rows)
rows.res = zeros(nrow(rows))
for gr in groupby(rows, :event_id)
    s = gr.stint ./ 10
    X = hcat(ones(nrow(gr)), gr.frac, s, s .^ 2)
    ok = gr.y .< 3
    gr.res .= gr.y .- X * (X[ok, :] \ gr.y[ok])
end
rows.big = rows.res .> 3
for gr in groupby(rows, [:event_id, :competitor_id])
    ok = .!gr.big
    any(ok) && (gr.res .-= mean(gr.res[ok]))
end

const EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, Inf]
function bin_table(title, x, sel)
    println("\n$title")
    println("  gap (s)        mean %    laps")
    for k in 1:(length(EDGES) - 1)
        m = sel .& (EDGES[k] .<= x .< EDGES[k + 1]) .& .!rows.big
        n = count(m)
        n == 0 && continue
        @printf("  %4.1f – %-4.1f  %+7.3f  %7d\n", EDGES[k], EDGES[k + 1], mean(rows.res[m]), n)
    end
    flush(stdout)
end

bin_table("car ahead on track, same lap", tr.ahead, tr.ahead_laps .== 0)
bin_table("car ahead on track, a lapped car", tr.ahead, tr.ahead_laps .< 0)
bin_table("car ahead on track, laps up", tr.ahead, tr.ahead_laps .> 0)
bin_table("car behind on track, lapping this car (blue flags)", tr.behind, tr.behind_laps .> 0)
bin_table("car behind on track, same lap", tr.behind, tr.behind_laps .== 0)
bin_table("car ahead in race position (all)", tr.front, trues(nrow(rows)))
bin_table("car behind in race position, with the car ahead over 5 s away", tr.back, tr.front .> 5)
bin_table("car behind in race position, leading", tr.back, isinf.(tr.front))

println("\nfield pace per race (sd over its laps, %), by condition")
cond = d.race_track_names
k_mixed, k_wet = findfirst(==(:mixed), cond), findfirst(==(:wet), cond)
rsd = [std(fp[d.race .== r]) for r in eachindex(d.races)]
for (name, sel) in [("dry", (d.race_track[:, k_mixed] .== 0) .& (d.race_track[:, k_wet] .== 0)),
                    ("mixed", d.race_track[:, k_mixed] .== 1), ("wet", d.race_track[:, k_wet] .== 1)]
    v = rsd[sel]
    @printf("  %-6s races %3d  median %.3f  90%% %.3f  max %.3f\n", name, length(v), median(v), quantile(v, 0.9),
            maximum(v))
end
top = sortperm(rsd; rev = true)[1:10]
println("  largest: ", join(("$(d.races[r]) $(round(rsd[r]; digits = 2))" for r in top), ", "))
