# Fit the gap-to-winner model (#9) on F1 data and write effect tables.
#
#   julia --project scripts/gap_fit.jl <timed|all> [ergast_csv_dir]
#
# `timed`: lead-lap finishers only; `all`: plus lapped cars as intervals.
# Student-t(4) noise, 4 chains x 1000 draws, run one after another. Progress
# lines with an ETA are printed every 100 iterations per chain.

using Diomedes, CSV, DataFrames, Statistics

mode = get(ARGS, 1, "all")
dir = get(ARGS, 2, joinpath(@__DIR__, "..", "data"))
res = fetch_results(ErgastCSV(dir), 1950:2100)
g = prepare_gaps(res; include_lapped = mode == "all")
println(g); flush(stdout)

t = @elapsed chain = fit_gaps(g; n_chains = 4, progress = false, progress_log = stdout)
println("fit: $(round(t / 60; digits = 1)) min")
for k in (:σ_comp, :σ_mach, :σ_y)
    println("$k: $(round(mean(chain[k]); digits = 3)) ± $(round(std(chain[k]); digits = 3))")
end
println("divergences: ", count(identity, chain[:numerical_error]),
        ", mean tree depth: ", round(mean(chain[:tree_depth]); digits = 2))
println("convergence: ", convergence_summary(chain))

eff = gap_effects_table(chain, g)
mkpath("output")
CSV.write("output/gap_$(mode)_competitor_effects.csv", eff.competitors)
CSV.write("output/gap_$(mode)_machine_effects.csv", eff.machines)
println("\nFastest 10 drivers (% of race time vs the average driver; negative = faster):")
show(first(eff.competitors, 10); allcols = true)
println()
