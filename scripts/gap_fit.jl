# Fit the gap-to-winner model (#9) on F1 data and write effect tables.
#
#   julia --project scripts/gap_fit.jl <data> <model>                  # 4 chains, one after another
#   julia --project scripts/gap_fit.jl <data> <model> chain <k>        # one chain (seed k), saved to output/
#   julia --project scripts/gap_fit.jl <data> <model> chain <k> block=25 [step=<x>] [warm=<tag>] [probe]
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

# key=value options, anywhere (#131):
#   delta=<x>    NUTS's target acceptance rate (default 0.8, #62)
#   block=<n>    `chain`: sample in saved, resumable blocks of n draws, with a per-iteration
#                trace in output/gap_<data>_<model>[_probe]_chain<k>_trace.tsv. Rerun the same
#                command to resume after a kill. Combine reads blocks or whole-chain files.
#   step=<x>     with block=: a fixed step size after block 1 (some chains adapt one that
#                puts them at tree depth 9-10, several times slower)
#   draws=<n>, warmup=<n>  draws per chain (default 1000) and warm-up iterations (500; 300 warm)
#   warm=<tag>   with block=: start from chain k's adapted metric and step size, read from the
#                last block of output/<tag>_chain<k> (a fit of the same model and data, e.g. its
#                probe; the parameter count must match)
# The flag `probe` adds _probe to the output names, so a short test run never mixes with a fit.
KEYS = ("delta", "block", "step", "draws", "warmup", "warm")
opt(key) = (i = findfirst(a -> startswith(a, key * "="), ARGS); i === nothing ? nothing : split(ARGS[i], "="; limit = 2)[2])
for a in ARGS
    occursin('=', a) && split(a, "=")[1] ∉ KEYS && error("unknown option $a (known: $(join(KEYS, ", ")))")
end
δ = something(tryparse(Float64, something(opt("delta"), "")), 0.8)
args = filter(a -> !occursin('=', a) && a != "probe", ARGS)
probe = "probe" in ARGS
mode = get(args, 1, "jolpica1950-2025+partial")
model = get(args, 2, "t4")
action = get(args, 3, "serial")
tag = "gap_$(mode)_$(model)" * (probe ? "_probe" : "")
chain_path(k) = "output/$(tag)_chain$(k).jls"
block_files(k) = sort(filter(f -> occursin(Regex("^\\Q$(tag)_chain$(k)_block\\E\\d+\\.jls\$"), f), readdir("output")),
                      by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
load_chain(k) = (fs = block_files(k); isempty(fs) ? rehash_chain!(deserialize(chain_path(k))) :
                 reduce(vcat, [Diomedes.block_chain(joinpath("output", f)) for f in fs]))
spec = model_spec(model)
sampler = opt("warmup") === nothing ? NUTS(δ; adtype = Diomedes.gap_sampler().adtype) :
          NUTS(parse(Int, opt("warmup")), δ; adtype = Diomedes.gap_sampler().adtype)
fit(; kw...) = spec.family === :t4 ? fit_gaps(g; sampler, kw...) :
    fit_paceloss(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                 age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                 centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν,
                 race_hier = spec.race_hier, race_mean = spec.race_mean, circ = spec.circ, track = spec.track, walk = spec.walk, walk1 = spec.walk1, works = spec.works, dinc = spec.dinc, ncres = spec.ncres, wlev = spec.wlev, kc = spec.kc, kst = spec.kst, wst = spec.wst, wstc = spec.wstc, wstl = spec.wstl, wlevn = spec.wlevn, crel = get(spec, :crel, false), sampler, kw...)

include("gap_data.jl")
res, g = gap_data(mode)
println(g); flush(stdout)
mkpath("output")

n_samples = parse(Int, something(opt("draws"), "1000"))
if action == "chain"
    k = parse(Int, args[4])
    t = if opt("block") === nothing
        opt("step") === nothing && opt("warm") === nothing || error("step= and warm= need block=")
        @elapsed chain = fit(; rng = Xoshiro(k), progress = false, progress_log = stdout, n_samples)
    else
        kw = (; checkpoint = "output/$(tag)_chain$(k)", seed = k, block = parse(Int, opt("block")), n_samples, δ,
              fixed_step = opt("step") === nothing ? nothing : parse(Float64, opt("step")))
        if opt("warm") === nothing
            kw = (; kw..., n_adapts = parse(Int, something(opt("warmup"), "500")))
        else
            wa = Diomedes.checkpoint_adapted("output/$(opt("warm"))_chain$(k)")
            println("warm start from output/$(opt("warm"))_chain$(k): step size $(round(wa.ϵ; digits = 4))"); flush(stdout)
            kw = (; kw..., n_adapts = parse(Int, something(opt("warmup"), "300")), external = lap_warm_sampler(wa.Minv, wa.ϵ; δ))
        end
        @elapsed chain = fit(; progress_log = stdout, kw...)
    end
    opt("block") === nothing && serialize(chain_path(k), chain)
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", step size ", round(mean(chain[:step_size]); digits = 4))
    exit()
elseif action == "combine"
    n = parse(Int, get(args, 4, "4"))
    chs = [load_chain(k) for k in 1:n]
    m = minimum(size(c, 1) for c in chs)          # chains still sampling contribute their saved blocks
    chain = reduce(hcat, [c[iter = 1:m] for c in chs])
else
    t = @elapsed chain = fit(; n_chains = 4, progress = false, progress_log = stdout, n_samples)
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
              Diomedes.pace_param_names(spec.pace_scale, spec.ncres, spec.kc, spec.kst)..., Diomedes.age_param_names(spec.age)...,
              Diomedes.big_param_names(spec.big_loss)..., Diomedes.slope_param_names(spec.slopes)..., Diomedes.dev_param_names(spec.dev)...,
              Diomedes.circ_param_names(spec.circ)..., Diomedes.walk_param_names(spec.walk, spec.walk1, spec.wlev, spec.wst, spec.wstc, spec.wstl)...,
              Diomedes.works_param_names(spec.works)..., Diomedes.dinc_param_names(spec.dinc)...)
        k in (:z_π_era, :z_λ_era, :e_π_rw, :e_λ_rw, :e_κ, :s_κ, :l_κ, :b_mach, :z_res, :e_age, :u_slope, :u_dev, :u_cx, :z_ch, :z_en, :zl_ch, :zl_en, :xs_ch, :xs_en, :u_dinc) && continue
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

eff = gap_effects_table(chain, g; cov = get(spec, :crel, false) ? LossCovariates(g, spec) : nothing)
CSV.write("output/$(tag)_competitor_effects.csv", eff.competitors)
CSV.write("output/$(tag)_machine_effects.csv", eff.machines)
println("\nFastest 10 drivers (% of race time vs the average driver; negative = faster):")
show(first(eff.competitors, 10); allcols = true)
println()
