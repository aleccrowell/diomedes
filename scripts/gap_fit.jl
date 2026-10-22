# Fit the gap-to-winner model (#9) on F1 data and write effect tables.
#
#   julia --project scripts/gap_fit.jl <data> <model>                  # 4 chains, one after another
#   julia --project scripts/gap_fit.jl <data> <model> chain <k>        # one chain (seed k), saved to output/
#   julia --project scripts/gap_fit.jl <data> <model> combine [n=4]    # combine saved chains 1..n
#
# Data modes (built by scripts/gap_data.jl):
# `jolpica<first>-<last>` (e.g. jolpica1950-2025): all rows from the Jolpica API
# (#55); `jolpica<first>-<last>+partial` adds retirements as calibrated partial gaps
# (1996 on, where lap times exist; the published fit's data); `+timed` keeps only
# lead-lap finishers.
# `wrc<first>-<last>` (e.g. wrc2023-2025): WRC stage times in the top two tiers
# (Rally1 incl. WRC, Rally2 incl. R5; see `wrc_top_tiers`), each special stage
# a "race", cars grouped by manufacturer × tier × season (#10).
# <model>: `t4` (Student-t(4) noise) or pace + loss noise `pl[_dur][_<era>]`:
# `_dur` adds race duration to the loss size, `<era>` (decade, regime, rw) adds
# era effects to incident probability and loss size, `_kappa` a pace scale by
# season (#15), `_circ` car-season circuit slopes and `_track` tight/mixed/wet on
# incidents (#81). The common race level is learned by default (#36); `_fgam` uses
# the fixed race-intercept prior, `_hgam` hierarchical intercepts (see `model_spec`).
# 1000 draws per chain after 500 warm-up iterations. `delta=<x>` sets the target
# acceptance rate (default 0.8, #62).
#
# On a Raspberry Pi 5, running the chains as separate processes (`chain k`, one
# per core, then `combine`) gives more total throughput than threads or serial.
# Progress lines (including warm-up) with ETAs are printed every 100 iterations.

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics
using Turing: FlexiChains, NUTS

# `delta=<x>` (anywhere) sets NUTS's target acceptance rate (default 0.8, #62)
δarg = findfirst(a -> startswith(a, "delta="), ARGS)
δ = δarg === nothing ? 0.8 : parse(Float64, split(ARGS[δarg], "=")[2])
args = filter(a -> !startswith(a, "delta="), ARGS)
mode = get(args, 1, "jolpica1950-2025+partial")
model = get(args, 2, "t4")
action = get(args, 3, "serial")
chain_path(k) = "output/gap_$(mode)_$(model)_chain$(k).jls"
spec = model_spec(model)
sampler = NUTS(δ; adtype = Diomedes.gap_sampler().adtype)
fit(; kw...) = spec.family === :t4 ? fit_gaps(g; sampler, kw...) :
    fit_paceloss(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                 age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                 centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν,
                 race_hier = spec.race_hier, race_mean = spec.race_mean, circ = spec.circ, track = spec.track, walk = spec.walk, sampler, kw...)

include("gap_data.jl")
res, g = gap_data(mode)
println(g); flush(stdout)
mkpath("output")

if action == "chain"
    k = parse(Int, args[4])
    t = @elapsed chain = fit(; rng = Xoshiro(k), progress = false, progress_log = stdout)
    serialize(chain_path(k), chain)
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", step size ", round(mean(chain[:step_size]); digits = 4))
    exit()
elseif action == "combine"
    n = parse(Int, get(args, 4, "4"))
    chain = reduce(hcat, [rehash_chain!(deserialize(chain_path(k))) for k in 1:n])
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
    for k in (Diomedes.loss_param_names(spec.loss_duration, spec.era, spec.track)[3:end]...,
              Diomedes.pace_param_names(spec.pace_scale)..., Diomedes.age_param_names(spec.age)...,
              Diomedes.big_param_names(spec.big_loss)..., Diomedes.slope_param_names(spec.slopes)..., Diomedes.dev_param_names(spec.dev)...,
              Diomedes.circ_param_names(spec.circ)..., Diomedes.walk_param_names(spec.walk)...)
        k in (:z_π_era, :z_λ_era, :e_π_rw, :e_λ_rw, :e_κ, :b_mach, :e_age, :u_slope, :u_dev, :u_cx, :z_ch, :z_en) && continue
        if k in (:τ_cx, :β_track_π, :β_track_λ)     # one per circuit covariate (#81)
            labels = k === :τ_cx ? ("speed", "rough", "temp") : ("tight", "mixed", "wet")
            v = stack(vec(chain[k]))
            for (j, l) in enumerate(labels)
                println("$k[$l]: $(round(mean(v[j, :]); digits = 3)) ± $(round(std(v[j, :]); digits = 3))")
            end
            continue
        end
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
