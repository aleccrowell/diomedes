# Fit the gap-to-winner model (#9) on F1 data and write effect tables.
#
#   julia --project scripts/gap_fit.jl <timed|all> <model>                  # 4 chains, one after another
#   julia --project scripts/gap_fit.jl <timed|all> <model> chain <k>        # one chain (seed k), saved to output/
#   julia --project scripts/gap_fit.jl <timed|all> <model> combine [n=4]    # combine saved chains 1..n
#
# `timed`: lead-lap finishers only; `all`: plus lapped cars as intervals.
# <model>: `t4` (Student-t(4) noise) or pace + loss noise `pl[_dur][_<era>]`:
# `_dur` adds race duration to the loss size, `<era>` (decade, regime, rw) adds
# era effects to incident probability and loss size, `_kappa` a pace scale by
# season (#15) (see `model_spec`).
# 1000 draws per chain after 500 warm-up iterations.
#
# On a Raspberry Pi 5, running the chains as separate processes (`chain k`, one
# per core, then `combine`) gives more total throughput than threads or serial.
# Progress lines (including warm-up) with ETAs are printed every 100 iterations.

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics
using Turing: FlexiChains

mode = get(ARGS, 1, "all")
model = get(ARGS, 2, "t4")
action = get(ARGS, 3, "serial")
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
chain_path(k) = "output/gap_$(mode)_$(model)_chain$(k).jls"
spec = model_spec(model)
fit(; kw...) = spec.family === :t4 ? fit_gaps(g; kw...) :
    fit_paceloss(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                 age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev, kw...)

res = fetch_results(ErgastCSV(data_dir), 1950:2100)
g = prepare_gaps(res; include_lapped = mode == "all")
println(g); flush(stdout)
mkpath("output")

if action == "chain"
    k = parse(Int, ARGS[4])
    t = @elapsed chain = fit(; rng = Xoshiro(k), progress = false, progress_log = stdout)
    serialize(chain_path(k), chain)
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", step size ", round(mean(chain[:step_size]); digits = 4))
    exit()
elseif action == "combine"
    n = parse(Int, get(ARGS, 4, "4"))
    chain = reduce(hcat, [deserialize(chain_path(k)) for k in 1:n])
else
    t = @elapsed chain = fit(; n_chains = 4, progress = false, progress_log = stdout)
    println("fit: $(round(t / 60; digits = 1)) min")
end

println("chains: $(FlexiChains.nchains(chain)) × $(FlexiChains.niters(chain)) draws")
for k in (spec.family === :t4 ? (:σ_comp, :σ_mach, :σ_y) : (:σ_comp, :σ_mach, :σ, :a_π, :a_λ))
    println("$k: $(round(mean(chain[k]); digits = 3)) ± $(round(std(chain[k]); digits = 3))")
end
if spec.family === :pl
    println("baseline incident probability logistic(a_π) ≈ $(round(mean(1 ./ (1 .+ exp.(-vec(chain[:a_π])))); digits = 3)), ",
            "baseline mean loss exp(a_λ) ≈ $(round(mean(exp.(vec(chain[:a_λ]))); digits = 2))%")
    for k in (Diomedes.loss_param_names(spec.loss_duration, spec.era)[3:end]...,
              Diomedes.pace_param_names(spec.pace_scale)..., Diomedes.age_param_names(spec.age)...,
              Diomedes.big_param_names(spec.big_loss)..., Diomedes.slope_param_names(spec.slopes)..., Diomedes.dev_param_names(spec.dev)...)
        k in (:z_π_era, :z_λ_era, :e_π_rw, :e_λ_rw, :e_κ, :b_mach, :e_age, :u_slope, :u_dev) && continue
        println("$k: $(round(mean(chain[k]); digits = 3)) ± $(round(std(chain[k]); digits = 3))")
    end
end
println("divergences: ", count(identity, chain[:numerical_error]),
        ", mean tree depth: ", round(mean(chain[:tree_depth]); digits = 2))
println("convergence (raw sampled coordinates): ", convergence_summary(chain))
# convergence of what the results use: centred effects, per-race loss/pace, scalars (#25)
ic = identified_convergence(chain, g, model)
println("convergence (identified quantities): max R-hat $(round(ic.overall.max_rhat; digits = 3)), ",
        "min ESS $(round(ic.overall.min_ess; digits = 1))")
for (k, v) in pairs(ic)
    k === :overall && continue
    println("  ", rpad(k, 8), " max R-hat $(round(v.max_rhat; digits = 3)), min ESS $(round(v.min_ess; digits = 1)) ($(v.worst))")
end

eff = gap_effects_table(chain, g)
CSV.write("output/gap_$(mode)_$(model)_competitor_effects.csv", eff.competitors)
CSV.write("output/gap_$(mode)_$(model)_machine_effects.csv", eff.machines)
println("\nFastest 10 drivers (% of race time vs the average driver; negative = faster):")
show(first(eff.competitors, 10); allcols = true)
println()
