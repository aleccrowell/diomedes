# Adapted step size and tree depth of the race model against NUTS's target acceptance (#62).
#
#   julia --project scripts/bench/race_delta.jl <δ> [n_adapts=150] [keep=20]
#
# Starts one chain of pl_dur_rw_kappa_age_big_slope_dev on jolpica1950-2025+partial
# from the last draw of chain 2 of the earlier fit of that model (its race
# intercepts matched by race), adapts for n_adapts iterations at target δ, then
# samples `keep` draws and prints their step size, tree depth, leapfrog steps and
# time per draw. One copy per core, one δ each.

using Diomedes, Serialization, Statistics, Random, Turing, Printf
include(joinpath(@__DIR__, "..", "gap_data.jl"))

δ = parse(Float64, ARGS[1])
n_adapts = parse(Int, get(ARGS, 2, "150"))
keep = parse(Int, get(ARGS, 3, "20"))
model_name, mode = "pl_dur_rw_kappa_age_big_slope_dev", "jolpica1950-2025+partial"
spec = model_spec(model_name)
res, g = gap_data(mode; io = devnull)
ch0 = rehash_chain!(deserialize("output/gap_$(mode)_$(model_name)_chain2.jls"))
θ0 = Diomedes.draw(ch0, Diomedes.model_param_names(spec), size(ch0[:σ], 1), 1)
# the earlier fit had one more race (2021 Belgian GP, now dropped): match intercepts by race
if length(θ0.γ) != length(g.races)
    old = sort(vcat(g.races, ["f1/2021-12/race"]))
    length(old) == length(θ0.γ) || error("race count mismatch: $(length(θ0.γ)) vs $(length(g.races))")
    θ0 = merge(θ0, (; γ = θ0.γ[[findfirst(==(r), old) for r in g.races]]))
end
m = paceloss_effects(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                     age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                     centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν, race_hier = spec.race_hier,
                     race_mean = spec.race_mean)
t0 = time()
ch = Diomedes.run_nuts(m, [InitFromParams(θ0)]; n_samples = keep, n_chains = 1, ensemble = MCMCSerial(),
                       sampler = NUTS(n_adapts, δ; adtype = Diomedes.gap_sampler().adtype), rng = Xoshiro(1),
                       progress = false, progress_log = stdout, log_every = 50, nadapts = n_adapts,
                       discard_adapt = true)
t = time() - t0
@printf("δ %.2f: step size %.4f, tree depth %.2f, leapfrog steps %.0f, acceptance %.3f, divergences %d; %.0f s total\n",
        δ, mean(ch[:step_size]), mean(ch[:tree_depth]), mean(ch[:n_steps]), mean(ch[:acceptance_rate]),
        count(ch[:numerical_error] .> 0), t)
