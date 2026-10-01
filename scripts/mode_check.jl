# Does a fit depend on where its chains start? (#32)
#
#   julia --project scripts/mode_check.jl <model> <A|B> <seed>
#
# Starts one chain of <model> on F1 data (all rows) from the last draw of a
# saved fit of `pl_dur_rw_kappa_age_big_slope_dev`, then samples 500 warm-up +
# 1000 draws, keeping the warm-up so the path away from the start is visible:
#   A: output/pre_priors/ (the old-priors fit, σ ≈ 0.36, a_π ≈ 0.9)
#   B: output/            (the scaled-priors fit, σ ≈ 0.41, a_π ≈ 0.27)
# Parameters the start fit lacks are set from it: μ_γ, τ_γ (`_hgam`) from the
# mean and sd of its race intercepts. Saves the full chain to
# output/modecheck_<model>_<start><seed>_full.jls and the kept draws to
# output/gap_all_<model>_chain<seed>.jls (for `loo_compare.jl`); prints σ and a_π
# by block.

using Diomedes, Random, Serialization, Statistics, Turing

model, start, seed = ARGS[1], ARGS[2], parse(Int, ARGS[3])
base = "pl_dur_rw_kappa_age_big_slope_dev"
src = start == "A" ? "output/pre_priors/gap_all_$(base)_chain1.jls" : "output/gap_all_$(base)_chain1.jls"
# a deserialized FlexiChain can carry key hashes from the process that wrote it
ch0 = deserialize(src)
let d = getfield(ch0, :_data); parentmodule(typeof(d)).rehash!(d); end
θ0 = Diomedes.draw(ch0, Diomedes.model_param_names(model_spec(base)), size(ch0[:σ], 1), 1)
spec = model_spec(model)
spec.race_hier && (θ0 = (; θ0..., μ_γ = mean(θ0.γ), τ_γ = std(θ0.γ)))
println("$model from $start (seed $seed): σ = $(θ0.σ), a_π = $(θ0.a_π)"); flush(stdout)

g = prepare_gaps(fetch_results(ErgastCSV("data"), 1950:2100))
m = paceloss_effects(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                     age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                     centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν, race_hier = spec.race_hier)
n_warm, n_keep = 500, 1000
t = @elapsed ch = Diomedes.run_nuts(m, [InitFromParams(θ0)]; n_samples = n_warm + n_keep, n_chains = 1,
                                    ensemble = MCMCSerial(), sampler = Diomedes.gap_sampler(), rng = Xoshiro(seed),
                                    progress = false, progress_log = stdout, log_every = 50,
                                    nadapts = n_warm, discard_adapt = false, discard_initial = 0)
mkpath("output")
serialize("output/modecheck_$(model)_$(start)$(seed)_full.jls", ch)
serialize("output/gap_all_$(model)_chain$(seed).jls", ch[iter = (n_warm + 1):(n_warm + n_keep)])
σ, aπ = vec(ch[:σ]), vec(ch[:a_π])
for r in (1:25, 26:100, 101:250, 251:500, 501:750, 751:1000, 1001:1250, 1251:1500)
    println("  iter $(first(r))-$(last(r)): σ $(round(mean(σ[r]); digits = 3)), a_π $(round(mean(aπ[r]); digits = 2))")
end
println("done $(round(t / 60; digits = 1)) min")
