# Short fit of a race model to compare tree depth and step size after adaptation
# (#102): a few hundred warm-up iterations, then a few draws.
#
#   julia --project scripts/bench/race_depth_probe.jl <model> <chain> [data=<mode>] [warmup=200] [draws=50]
#
# Saves output/probe_<data>_<model>_chain<k>.jls and prints the kept draws' mean
# tree depth, leapfrog steps and step size. Run one copy per core.

using Diomedes, Random, Serialization, Statistics, Printf, Turing
include(joinpath(@__DIR__, "..", "gap_data.jl"))

mode, args = data_arg(ARGS)
opt(k, d) = (i = findfirst(a -> startswith(a, "$k="), args); i === nothing ? d : parse(Int, split(args[i], "=")[2]))
pos = filter(a -> !occursin("=", a), args)
model_name, k = pos[1], parse(Int, pos[2])
nw, nd = opt("warmup", 200), opt("draws", 50)
spec = model_spec(model_name)
_, g = gap_data(mode)
println(g); flush(stdout)
sampler = NUTS(nw, 0.8; adtype = Diomedes.gap_sampler().adtype)
t = @elapsed chain = fit_paceloss(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                                  age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                                  centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν,
                                  race_hier = spec.race_hier, race_mean = spec.race_mean, circ = spec.circ,
                                  track = spec.track, walk = spec.walk, walk1 = spec.walk1, works = spec.works,
                                  dinc = spec.dinc, ncres = spec.ncres, wlev = spec.wlev, kc = spec.kc, kst = spec.kst, wst = spec.wst, sampler, n_samples = nd, rng = Xoshiro(k),
                                  progress = false, progress_log = stdout, log_every = 25)
serialize("output/probe_$(mode)_$(model_name)_chain$(k).jls", chain)
td, ns, ss = vec(chain[:tree_depth]), vec(chain[:n_steps]), vec(chain[:step_size])
@printf("probe %s chain %d: %.1f min; %d warm-up + %d draws; tree depth mean %.2f (max %d), leapfrog steps %.0f, step size %.4f, divergences %d\n",
        model_name, k, t / 60, nw, nd, mean(td), maximum(td), mean(ns), mean(ss), count(identity, coalesce.(chain[:numerical_error], false)))
