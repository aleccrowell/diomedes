# Cost of one log density and one gradient of a race model, for hard stops and
# cost estimates (#62, #81).
#
#   julia --project scripts/bench/race_gradient.jl [model=pl_dur_rw_kappa_age_big_slope_dev] [data=<mode>] [n=30]
#
# Evaluates at a draw from the prior-centred start used by `fit_paceloss`, with
# the sampler's AD backend (uncompiled ReverseDiff), and prints the mean of `n`
# calls after warm-up: the median leaves out garbage collection pauses, which a
# sampler pays (#92; `race_step_profile.jl` breaks them down). Run one copy per
# core to time it under the load of a 4-chain fit.

using Diomedes, Random, Statistics, Printf, Turing
include(joinpath(@__DIR__, "..", "gap_data.jl"))

mode, args = data_arg(ARGS)
model_name = get(filter(a -> !startswith(a, "n="), args), 1, "pl_dur_rw_kappa_age_big_slope_dev")
narg = findfirst(a -> startswith(a, "n="), args)
n = narg === nothing ? 30 : parse(Int, split(args[narg], "=")[2])
spec = model_spec(model_name)
_, g = gap_data(mode; io = devnull)
m = paceloss_effects(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                     age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                     centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν, race_hier = spec.race_hier,
                     race_mean = spec.race_mean, circ = spec.circ, track = spec.track, walk = spec.walk, walk1 = spec.walk1, works = spec.works, dinc = spec.dinc, ncres = spec.ncres, wlev = spec.wlev, kc = spec.kc, kst = spec.kst, wst = spec.wst, wstc = spec.wstc, wstl = spec.wstl, wlevn = spec.wlevn)
LDF = Turing.DynamicPPL.LogDensityFunction
LDP = Turing.DynamicPPL.LogDensityProblems
# the sampler's coordinates: linked (unconstrained); constrained ones can hide slow transforms (#102)
f = LDF(m, Turing.DynamicPPL.getlogjoint_internal, Turing.DynamicPPL.LinkAll(); adtype = Diomedes.gap_sampler().adtype)
x = 0.1 .* randn(Xoshiro(1), LDP.dimension(f))            # unconstrained: any vector is valid
for _ in 1:3; LDP.logdensity_and_gradient(f, x); end
tl = mean([@elapsed(LDP.logdensity(f, x)) for _ in 1:n])
tg = mean([@elapsed(LDP.logdensity_and_gradient(f, x)) for _ in 1:n])
@printf("%s: %d parameters, log density %.1f ms, gradient %.1f ms (mean of %d)\n",
        model_name, LDP.dimension(f), 1e3tl, 1e3tg, n)
