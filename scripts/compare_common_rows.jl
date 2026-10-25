# PSIS-LOO comparison of two race fits on the rows they share, when they were fitted
# to different data (e.g. 1950–2025 and 1950–2026, #99).
#
#   julia --project scripts/compare_common_rows.jl <table_a.csv> <table_b.csv>
#
# Reads two per-row tables of scripts/pareto_k_rows.jl and joins them on season,
# race, driver, car and row kind, plus an occurrence count for the rare rows that
# still repeat (1950s shared drives). Prints Δelpd (b − a) on the shared rows and on
# those with Pareto k ≤ 0.7 in both, by decade, and the high-k counts on the shared
# rows. Each fit's own extra rows also inform its parameters, so the comparison is
# approximate.

using CSV, DataFrames, Statistics, Printf

a, b = CSV.read(ARGS[1], DataFrame), CSV.read(ARGS[2], DataFrame)
base = [:season, :race, :driver, :car, :kind]
for d in (a, b)
    d.occ = zeros(Int, nrow(d))
    for grp in groupby(d, base); grp.occ .= 1:nrow(grp); end
end
key = [base; :occ]
j = innerjoin(select(a, key..., :k => :k_a, :elpd => :elpd_a), select(b, key..., :k => :k_b, :elpd => :elpd_b); on = key)
@printf("a: %d rows, b: %d rows, shared: %d\n", nrow(a), nrow(b), nrow(j))
d = j.elpd_b .- j.elpd_a
show_gain(lbl, m) = @printf("%-34s rows %5d  Δelpd %7.1f  se %.1f\n", lbl, sum(m), sum(d[m]), sqrt(sum(m)) * std(d[m]))
ok = (j.k_a .<= 0.7) .& (j.k_b .<= 0.7)
show_gain("shared rows", trues(nrow(j)))
show_gain("k ≤ 0.7 in both", ok)
show_gain("k > 0.7 in either", .!ok)
for dec in 1950:10:2020
    m = 10 .* (j.season .÷ 10) .== dec
    any(m .& ok) && show_gain("decade $dec, k ≤ 0.7 both", m .& ok)
end
@printf("high-k (> 0.7) on shared rows: a %d, b %d\n", count(>(0.7), j.k_a), count(>(0.7), j.k_b))
