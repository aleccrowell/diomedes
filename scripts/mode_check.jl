# Does a fit depend on where its chains start? (#32)
#
#   julia --project scripts/mode_check.jl <model> <A|B> <seed> [base=pl_dur_rw_kappa_age_big_slope_dev] [data=<mode>]
#
# Starts one chain of <model> on F1 data (`data`: a mode of scripts/gap_data.jl,
# default the published fit's) from the last draw of a saved fit of <base> on the
# same data, then samples 500 warm-up + 1000 draws, keeping the warm-up so the path
# away from the start is visible:
#   A: output/pre_priors/ (the old-priors fit, σ ≈ 0.36, a_π ≈ 0.9)
#   B: output/            (the scaled-priors fit, σ ≈ 0.41, a_π ≈ 0.27)
# Parameters the start fit lacks are set from it: μ_γ, τ_γ (`_hgam`) from the
# mean and sd of its race intercepts (μ_γ only for the default learned level). Saves the full chain to
# output/modecheck_<data>_<model>_<start><seed>_full.jls and the kept draws to
# output/gap_<data>_<model>_chain<seed>.jls (for `loo_compare.jl`); prints σ and a_π
# by block.

using Diomedes, Random, Serialization, Statistics, Turing
include("gap_data.jl")

mode, args = data_arg(ARGS)
model, start, seed = args[1], args[2], parse(Int, args[3])
base = get(args, 4, "pl_dur_rw_kappa_age_big_slope_dev")
src = (start == "A" ? "output/pre_priors/" : "output/") * "gap_$(mode)_$(base)_chain1.jls"
ch0 = rehash_chain!(deserialize(src))
θ0 = Diomedes.draw(ch0, Diomedes.model_param_names(model_spec(base)), size(ch0[:σ], 1), 1)
spec = model_spec(model)
spec.race_hier && (θ0 = (; θ0..., μ_γ = mean(θ0.γ), τ_γ = std(θ0.γ)))
spec.race_mean && (θ0 = (; θ0..., μ_γ = mean(θ0.γ)))
println("$model from $start (seed $seed): σ = $(θ0.σ), a_π = $(θ0.a_π)"); flush(stdout)

_, g = gap_data(mode)
m = paceloss_effects(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                     age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                     centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν, race_hier = spec.race_hier,
                     race_mean = spec.race_mean)
n_warm, n_keep = 500, 1000
t = @elapsed ch = Diomedes.run_nuts(m, [InitFromParams(θ0)]; n_samples = n_warm + n_keep, n_chains = 1,
                                    ensemble = MCMCSerial(), sampler = Diomedes.gap_sampler(), rng = Xoshiro(seed),
                                    progress = false, progress_log = stdout, log_every = 50,
                                    nadapts = n_warm, discard_adapt = false, discard_initial = 0)
mkpath("output")
serialize("output/modecheck_$(mode)_$(model)_$(start)$(seed)_full.jls", ch)
serialize("output/gap_$(mode)_$(model)_chain$(seed).jls", ch[iter = (n_warm + 1):(n_warm + n_keep)])
σ, aπ = vec(ch[:σ]), vec(ch[:a_π])
for r in (1:25, 26:100, 101:250, 251:500, 501:750, 751:1000, 1001:1250, 1251:1500)
    println("  iter $(first(r))-$(last(r)): σ $(round(mean(σ[r]); digits = 3)), a_π $(round(mean(aπ[r]); digits = 2))")
end
println("done $(round(t / 60; digits = 1)) min")
