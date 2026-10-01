# Population career curve from a pace + loss fit with `_age` (#15 stage 2).
#
#   julia --project scripts/age_curve.jl [model=pl_dur_rw_kappa_age]
#
# Prints the curve per age (% of race time, in average-season units; negative
# = faster) with 90% intervals and rows per age. Only the shape up to a linear
# tilt is identifiable (age–period–cohort; see `AgeCurveBasis`): the curve has
# no constant or linear part, so read its bends (e.g. rise then fall), not its
# overall slope or the absolute age of its minimum.

using Diomedes, CSV, DataFrames, Serialization, Statistics

model = get(ARGS, 1, "pl_dur_rw_kappa_age")
g = prepare_gaps(fetch_results(ErgastCSV(joinpath(@__DIR__, "..", "data")), 1950:2100))
chain = reduce(hcat, rehash_chain!.(deserialize.(filter(isfile, ["output/gap_all_$(model)_chain$(k).jls" for k in 1:8]))))
B = AgeCurveBasis(g)
ni, nc = size(chain[:τ_age])
F = reduce(vcat, [age_curve(chain[:τ_age][i, c], chain[:e_age][i, c], B)' for c in 1:nc for i in 1:ni])
df = DataFrame(age = g.age_years, rows = Int.(g.age_rows), mean = vec(mean(F; dims = 1)),
               q05 = [quantile(c, 0.05) for c in eachcol(F)], q95 = [quantile(c, 0.95) for c in eachcol(F)])
mkpath("output")
CSV.write("output/age_curve_$(model).csv", df)
println("τ_age = $(round(mean(chain[:τ_age]); digits = 3)) ± $(round(std(chain[:τ_age]); digits = 3))")
println("age  rows   curve (% of race time) [90%]")
for r in eachrow(df)
    println(lpad(r.age, 3), lpad(r.rows, 6), "   ", rpad(round(r.mean; digits = 3), 7), "[",
            round(r.q05; digits = 3), ", ", round(r.q95; digits = 3), "]")
end
