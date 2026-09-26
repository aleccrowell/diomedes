# Fit the gap-to-winner model (#9) on F1 data and write effect tables.
#
#   julia --project scripts/gap_fit.jl <timed|all>                  # 4 chains, one after another
#   julia --project scripts/gap_fit.jl <timed|all> chain <k>        # one chain (seed k), saved to output/
#   julia --project scripts/gap_fit.jl <timed|all> combine [n=4]    # combine saved chains 1..n
#
# `timed`: lead-lap finishers only; `all`: plus lapped cars as intervals.
# Student-t(4) noise, 1000 draws per chain after 500 warm-up iterations.
#
# On a Raspberry Pi 5, running the chains as separate processes (`chain k`, one
# per core, then `combine`) gives more total throughput than threads or serial.
# Progress lines (including warm-up) with ETAs are printed every 100 iterations.

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics
using Turing: FlexiChains

mode = get(ARGS, 1, "all")
action = get(ARGS, 2, "serial")
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
chain_path(k) = "output/gap_$(mode)_chain$(k).jls"

res = fetch_results(ErgastCSV(data_dir), 1950:2100)
g = prepare_gaps(res; include_lapped = mode == "all")
println(g); flush(stdout)
mkpath("output")

if action == "chain"
    k = parse(Int, ARGS[3])
    t = @elapsed chain = fit_gaps(g; rng = Xoshiro(k), progress = false, progress_log = stdout)
    serialize(chain_path(k), chain)
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", step size ", round(mean(chain[:step_size]); digits = 4))
    exit()
elseif action == "combine"
    n = parse(Int, get(ARGS, 3, "4"))
    chain = reduce(hcat, [deserialize(chain_path(k)) for k in 1:n])
else
    t = @elapsed chain = fit_gaps(g; n_chains = 4, progress = false, progress_log = stdout)
    println("fit: $(round(t / 60; digits = 1)) min")
end

println("chains: $(FlexiChains.nchains(chain)) × $(FlexiChains.niters(chain)) draws")
for k in (:σ_comp, :σ_mach, :σ_y)
    println("$k: $(round(mean(chain[k]); digits = 3)) ± $(round(std(chain[k]); digits = 3))")
end
println("divergences: ", count(identity, chain[:numerical_error]),
        ", mean tree depth: ", round(mean(chain[:tree_depth]); digits = 2))
println("convergence: ", convergence_summary(chain))

eff = gap_effects_table(chain, g)
CSV.write("output/gap_$(mode)_competitor_effects.csv", eff.competitors)
CSV.write("output/gap_$(mode)_machine_effects.csv", eff.machines)
println("\nFastest 10 drivers (% of race time vs the average driver; negative = faster):")
show(first(eff.competitors, 10); allcols = true)
println()
