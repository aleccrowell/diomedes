# Summary of the per-row PSIS-LOO tables of scripts/pareto_k_rows.jl (#91): where the
# rows with Pareto k > 0.7 are, and the elpd gain of one fit over another split by row
# reliability, decade, race condition and residual size.
#
#   julia --project scripts/pareto_k_summary.jl [model_a model_b] [data=<mode>]
#
# Reads output/pareto_k_<data>_<model>.csv for two models (default: the published
# fit and the #81 fit); the gain is model_b − model_a.

using CSV, DataFrames, Statistics
dat = something(findfirst(x -> startswith(x, "data="), ARGS), 0)
data = dat == 0 ? "jolpica1950-2025+partial" : split(ARGS[dat], "=")[2]
ms = filter(x -> !startswith(x, "data="), ARGS)
ma, mb = isempty(ms) ? ("pl_dur_rw_kappa_age_big_slope_dev", "pl_dur_rw_kappa_age_big_slope_dev_circ_track") : (ms[1], ms[2])
a = CSV.read("output/pareto_k_$(data)_$(ma).csv", DataFrame); b = CSV.read("output/pareto_k_$(data)_$(mb).csv", DataFrame)
println("a = $ma\nb = $mb")
q(v) = round.(quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]); digits = 2)
function tab(d, col)
    t = combine(groupby(d, col), :hi => length => :rows, :hi => sum => :highk)
    t.rate = round.(100 .* t.highk ./ t.rows; digits = 1); t.share = round.(100 .* t.highk ./ sum(t.highk); digits = 1)
    show(stdout, MIME"text/plain"(), sort(t, col); allrows = true, summary = false, eltypes = false); println()
end
for (name, d) in (("a", a), ("b", b))
    d.hi = d.k .> 0.7
    d.decade = 10 .* (d.season .÷ 10)
    d.wallsbin = [w < 0.5 ? "0-0.5" : w < 1.0 ? "0.5-1" : w < 1.5 ? "1-1.5" : "1.5-2" for w in d.walls]
    d.rbin = [r < -1 ? "a <-1" : r < 1 ? "b -1..1" : r < 3 ? "c 1..3" : r < 10 ? "d 3..10" : "e >10" for r in d.resid]
    println("\n===== $name: $(sum(d.hi)) high-k of $(nrow(d))")
    for col in (:kind, :condition, :decade, :wallsbin, :rbin); tab(d, col); end
    h = d[d.hi, :]
    println("resid quantiles high-k ", q(h.resid), "  all ", q(d.resid))
end
# rows that became high-k in b
new = (b.k .> 0.7) .& (a.k .<= 0.7); gone = (a.k .> 0.7) .& (b.k .<= 0.7)
println("\nnew high-k: $(sum(new)), no longer high-k: $(sum(gone))")
n = b[new, :]
for col in (:kind, :condition, :decade, :rbin)
    t = combine(groupby(n, col), nrow => :n); show(stdout, MIME"text/plain"(), sort(t, col); allrows = true, summary = false, eltypes = false); println()
end
# per car-season / driver concentration
for col in (:car, :driver, :race)
    t = sort(combine(groupby(b[b.k .> 0.7, :], col), nrow => :n), :n; rev = true)
    println("top $col: ", join(["$(r[col]) $(r.n)" for r in eachrow(first(t, 10))], ", "))
end
# rows per car-season vs high-k rate (sparse cars?)
cs = combine(groupby(b, :car), nrow => :rows, :hi => sum => :hk)
for (lo, hi) in ((1, 5), (6, 10), (11, 20), (21, 40), (41, 1000))
    s = cs[(cs.rows .>= lo) .& (cs.rows .<= hi), :]
    println("car-seasons with $lo-$hi rows: $(nrow(s)) cars, rows $(sum(s.rows)), high-k rate $(round(100sum(s.hk)/max(sum(s.rows),1); digits=1))%")
end

# the gain of b over a, split
d = b.elpd .- a.elpd
show_gain(lbl, m) = println(rpad(lbl, 44), "rows $(lpad(sum(m), 5))  Δelpd $(lpad(round(sum(d[m]); digits = 1), 7))  se $(round(sqrt(sum(m)) * std(d[m]); digits = 1))")
ok = (a.k .<= 0.7) .& (b.k .<= 0.7)
show_gain("all rows", trues(nrow(a)))
show_gain("k ≤ 0.7 in both fits", ok)
show_gain("k > 0.7 in either fit", .!ok)
for dec in 1950:10:2020
    m = 10 .* (b.season .÷ 10) .== dec
    show_gain("decade $dec", m); show_gain("decade $dec, k ≤ 0.7 both", m .& ok)
end
for c in ("dry", "mixed", "wet"); show_gain("condition $c, k ≤ 0.7 both", (b.condition .== c) .& ok); end
# residual size: which rows gain? (incident rows vs clean)
show_gain("resid > 3 (incident rows), k ≤ 0.7 both", (b.resid .> 3) .& ok)
show_gain("|resid| ≤ 1, k ≤ 0.7 both", (abs.(b.resid) .<= 1) .& ok)
