# Acceptance rate and tree depth of the 1996–2025 lap fit at a fixed step size (#62).
#
#   julia --project scripts/bench/lap_stepsize.jl <ϵ> [iters=20] [chain=1]
#
# Starts NUTS from the last saved position and the adapted metric of chain <chain>
# of laps_1996-2025_inferred_notyres_slopes_dev, runs <iters> iterations at step
# size ϵ without adaptation, and prints the mean acceptance rate, tree depth and
# time per iteration. NUTS's dual averaging picks the ϵ whose mean acceptance is
# the target δ, so running a few ϵ at once shows which δ keeps the tree depth
# below a doubling. One copy per core.

using Diomedes, Turing, AdvancedHMC, Serialization, Statistics, Printf, Random
const DynamicPPL = Turing.DynamicPPL

ϵ = parse(Float64, ARGS[1])
iters = parse(Int, get(ARGS, 2, "20"))
k = parse(Int, get(ARGS, 3, "1"))
tag = "laps_1996-2025_inferred_notyres_slopes_dev"
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "..", "data"))
ergast, jolpica = ErgastCSV(data_dir), JolpicaF1()
src(f) = reduce(vcat, filter(p -> nrow(p) > 0, [f(ergast, collect(1996:2019)), f(jolpica, collect(2020:2025))]))
d = prepare_laps(src(fetch_laps), src(fetch_results); pits = :inferred)
model = lap_effects(d; tyres = false, slopes = true, dev = true)
ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
                                    adtype = Diomedes.gap_sampler().adtype)
ws = deserialize("output/$(tag)_chain$(k)_warmstate.jls")                       # (; Minv, ϵ)
blocks = filter(f -> startswith(f, "$(tag)_chain$(k)_block"), readdir("output"))
last_block = argmax(f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]), blocks)
st = only(Turing.loadstate(rehash_chain!(deserialize(joinpath("output", last_block)))))
θ0 = Vector{Float64}((hasproperty(st, :state) ? st.state : st).transition.z.θ)
length(θ0) == length(ws.Minv) || error("position and metric have different lengths")
metric = DiagEuclideanMetric(Vector{Float64}(ws.Minv))
h = Hamiltonian(metric, ldf)
κ = HMCKernel(Trajectory{MultinomialTS}(Leapfrog(ϵ), GeneralisedNoUTurn()))
t = @elapsed samples, stats = AdvancedHMC.sample(Xoshiro(1), h, κ, θ0, iters; progress = false, verbose = false)
@printf("ϵ %.4f (adapted %.4f): acceptance %.3f, tree depth %.2f, leapfrog steps %.0f, %.1f s per iteration, divergences %d\n",
        ϵ, ws.ϵ, mean(s.acceptance_rate for s in stats), mean(s.tree_depth for s in stats),
        mean(s.n_steps for s in stats), t / iters, count(s.numerical_error for s in stats))
