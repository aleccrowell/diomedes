# Per-driver career slopes from a pace + loss fit with `_slope` (#15 stage 2b).
#
#   julia --project scripts/career_slopes.jl [model=pl_dur_rw_kappa_age_slope] [min_rows=40]
#
# Prints the drivers whose pace changed most over their careers relative to
# the average driver, in % of race time per decade of age (average-season
# units; negative = got faster with age than the average driver did). The
# slopes are centred across drivers (a common slope is not identifiable; see
# `CareerRows`), so they rank trajectories, not absolute improvement.

using Diomedes, CSV, DataFrames, Serialization, Statistics

model = get(ARGS, 1, "pl_dur_rw_kappa_age_slope")
min_rows = parse(Int, get(ARGS, 2, "40"))
g = prepare_gaps(fetch_results(ErgastCSV(joinpath(@__DIR__, "..", "data")), 1950:2100))
chain = reduce(hcat, rehash_chain!.(deserialize.(filter(isfile, ["output/gap_all_$(model)_chain$(k).jls" for k in 1:8]))))
ni, nc = size(chain[:τ_slope])
S = reduce(vcat, [career_slopes(chain[:τ_slope][i, c], chain[:u_slope][i, c])' for c in 1:nc for i in 1:ni])

comps = vcat(g.t_comp, g.c_comp)
ages = [k == 0 ? missing : g.age_years[k] for k in vcat(g.t_age, g.c_age)]
names = Dict(r.competitor_id => r.competitor_name for r in eachrow(g.rows))
span(d) = (a = collect(skipmissing(ages[comps .== d])); isempty(a) ? (missing, missing) : extrema(a))
df = DataFrame(id = g.competitors, name = [names[c] for c in g.competitors],
               rows = [count(==(d), comps) for d in eachindex(g.competitors)],
               first_age = [span(d)[1] for d in eachindex(g.competitors)],
               last_age = [span(d)[2] for d in eachindex(g.competitors)],
               mean = vec(mean(S; dims = 1)), sd = vec(std(S; dims = 1)),
               q05 = [quantile(c, 0.05) for c in eachcol(S)], q95 = [quantile(c, 0.95) for c in eachcol(S)])
mkpath("output")
CSV.write("output/career_slopes_$(model).csv", df)

println("τ_slope = $(round(mean(chain[:τ_slope]); digits = 3)) ± $(round(std(chain[:τ_slope]); digits = 3)) % per decade")
big = filter(r -> r.rows >= min_rows, df)
println("drivers with ≥ $min_rows rows: $(nrow(big)); 90% interval excludes 0 for ",
        count(r -> r.q05 > 0 || r.q95 < 0, eachrow(big)))
show_rows(t) = for r in eachrow(t)
    println(rpad(r.name, 24), lpad(r.rows, 5), "   ages ", r.first_age, "–", r.last_age, "   ",
            rpad(round(r.mean; digits = 2), 6), "± ", rpad(round(r.sd; digits = 2), 5),
            "[", round(r.q05; digits = 2), ", ", round(r.q95; digits = 2), "]")
end
println("\nImproved most with age (relative to the average driver):")
show_rows(first(sort(big, :mean), 12))
println("\nDeclined most with age:")
show_rows(first(sort(big, :mean; rev = true), 12))
