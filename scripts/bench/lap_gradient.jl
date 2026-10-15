# Time one gradient of the lap model's log density (#62).
#
#   julia --project scripts/bench/lap_gradient.jl <first>-<last> [reps=30] [terms...]
#
# `terms` are lap_fit.jl's: notyres, slopes, dev, lead, tyremgmt, starts (with none
# given: notyres slopes dev, the model these timings were first taken for).
# Prints the median time of a log density and of a log density + gradient at a
# point near the prior centre, with the sampler's AD (uncompiled ReverseDiff).
# Run several copies at once (one per core) to see how much they slow each other
# (shared memory bandwidth) against one copy alone.

using Diomedes, Turing, Statistics, Printf, Random, DataFrames
const DynamicPPL = Turing.DynamicPPL
const LogDensityProblems = Turing.LogDensityProblems

terms = ("notyres", "slopes", "dev", "lead", "tyremgmt", "starts")
has = any(in(terms), ARGS) ? Set(filter(in(terms), ARGS)) : Set(["notyres", "slopes", "dev"])
args = filter(∉(terms), ARGS)
m = match(r"^(\d{4})-(\d{4})$", args[1])
seasons = parse(Int, m[1]):parse(Int, m[2])
reps = parse(Int, get(args, 2, "30"))
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "..", "data"))
ergast, jolpica = ErgastCSV(data_dir), JolpicaF1()
part(f, ss) = filter(p -> nrow(p) > 0, [f(ergast, filter(<=(2019), collect(ss))), f(jolpica, filter(>(2019), collect(ss)))])
src(f) = reduce(vcat, part(f, seasons))
d = prepare_laps(src(fetch_laps), src(fetch_results); pits = :inferred)
println(d)
model = lap_effects(d; tyres = "notyres" ∉ has, slopes = "slopes" ∈ has, dev = "dev" ∈ has, lead = "lead" ∈ has,
                    tyre_mgmt = "tyremgmt" ∈ has, starts = "starts" ∈ has)
println("terms: ", join(sort(collect(has)), " "))
ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
                                    adtype = Diomedes.gap_sampler().adtype)
x = 0.05 .* randn(Xoshiro(1), LogDensityProblems.dimension(ldf))
LogDensityProblems.logdensity_and_gradient(ldf, x)
tl = [(@elapsed LogDensityProblems.logdensity(ldf, x)) for _ in 1:reps]
tg = [(@elapsed LogDensityProblems.logdensity_and_gradient(ldf, x)) for _ in 1:reps]
@printf("pid %d: dim %d, log density %.1f ms, gradient %.1f ms (median of %d)\n",
        getpid(), length(x), 1000median(tl), 1000median(tg), reps)
