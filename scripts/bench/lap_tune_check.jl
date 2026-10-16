# Step-size tuning on a real lap fit (#62): from the last saved block of a chain of
# laps_1996-2025_inferred_notyres_slopes_dev, run tune_step_size, then 100 draws at
# the chosen step size; print the probe and the draws' speed, acceptance, depth.
#
#   julia --project scripts/bench/lap_tune_check.jl [chain=1] [draws=100]
using Diomedes, Turing, Serialization, Statistics, Random, DataFrames, Printf
k = parse(Int, get(ARGS, 1, "1")); n = parse(Int, get(ARGS, 2, "100"))
tag = "laps_1996-2025_inferred_notyres_slopes_dev"
src(f) = f(JolpicaF1(), collect(1996:2025))
d = prepare_laps(src(fetch_laps), src(fetch_results); pits = :inferred)
model = lap_effects(d; tyres = false, slopes = true, dev = true)
blocks = filter(f -> startswith(f, "$(tag)_chain$(k)_block"), readdir("output"))
last_block = argmax(f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]), blocks)
ch0 = rehash_chain!(deserialize(joinpath("output", last_block)))
state = only(Turing.loadstate(ch0))
ad = Diomedes.lap_adapted(state)
sampler = Diomedes.lap_warm_sampler(ad.Minv, ad.ϵ)
@printf("chain %d, %s: adapted step size %.4f, its block: %.0f steps per draw, acceptance %.3f\n",
        k, last_block, ad.ϵ, mean(ch0[:n_steps]), mean(ch0[:acceptance_rate]))
t = @elapsed st, ϵ, probe = Diomedes.tune_step_size(model, state, sampler, 0.8; external = true)
for r in probe; println("  probe: factor $(r[1]), step $(round(r[2]; digits = 4)), acceptance $(r[3]), depth $(r[4]), divergences $(r[5])"); end
@printf("chosen step size %.4f (probe %.1f min)\n", ϵ, t / 60)
t = @elapsed ch = Diomedes.run_nuts(model, nothing; n_samples = n, n_chains = 1, ensemble = MCMCSerial(), sampler,
                                    rng = Xoshiro(99), progress = false, progress_log = nothing, log_every = 100,
                                    initial_state = st, n_adapts = 0)
@printf("%d draws at %.4f: %.1f s per draw, %.0f steps per draw, acceptance %.3f, depth %.2f, divergences %d\n",
        n, ϵ, t / n, mean(ch[:n_steps]), mean(ch[:acceptance_rate]), mean(ch[:tree_depth]), count(ch[:numerical_error] .> 0))
