# Mixing of the hard-to-mix quantities in short probe fits (#102).
#
#   julia --project scripts/bench/probe_mixing.jl <model> [data=<mode>] [prefix=probe]
#
# Reads output/<prefix>_<data>_<model>_chain<k>.jls (scripts/bench/race_depth_probe.jl)
# and prints, per chain, tree depth and step size, then per-chain means and ESS and
# the combined R-hat and ESS of the quantities that mixed worst in the
# directly-sampled-states fit: σ0_en, τ_en, Ferrari's 1966 engine and 1968 chassis
# states, β_works.

using Diomedes, Serialization, Statistics, Printf, MCMCDiagnosticTools
include(joinpath(@__DIR__, "..", "gap_data.jl"))

mode, args = data_arg(ARGS)
pi_ = findfirst(a -> startswith(a, "prefix="), args)
prefix = pi_ === nothing ? "probe" : String(split(args[pi_], "=")[2])
model = first(filter(a -> !occursin("=", a), args))
_, g = gap_data(mode; io = devnull)
spec = model_spec(model); cov = LossCovariates(g, spec); cw = cov.cars
files = filter(isfile, ["output/$(prefix)_$(mode)_$(model)_chain$(k).jls" for k in 1:8])
chs = [rehash_chain!(deserialize(f)) for f in files]
names = Diomedes.model_param_names(spec)
for (k, c) in enumerate(chs)
    @printf("chain %d: tree depth %.2f, steps %.0f, step size %.4f, divergences %d\n", k, mean(c[:tree_depth]),
            mean(c[:n_steps]), mean(c[:step_size]), count(identity, coalesce.(c[:numerical_error], false)))
end
ie = findfirst(==("ferrari 1966"), cw.engine.names); ic = findfirst(==("ferrari 1968"), cw.chassis.names)
qs = [("σ0_en", θ -> θ.σ0_en), ("τ_en", θ -> θ.τ_en), ("engine ferrari 1966", θ -> car_walk_states(θ, cw)[2][ie]),
      ("chassis ferrari 1968", θ -> car_walk_states(θ, cw)[1][ic])]
haskey(Diomedes.draw(chs[1], names, 1, 1), :β_works) && push!(qs, ("β_works", θ -> θ.β_works))
ni = size(chs[1][:σ_comp], 1)
for (lbl, f) in qs
    M = [f(Diomedes.draw(c, names, i, 1)) for i in 1:ni, c in chs]        # draws × chains
    A = reshape(M, size(M, 1), size(M, 2), 1)
    @printf("%-22s means %s  ESS per chain %s  | R-hat %.3f, ESS %.0f\n", lbl,
            join([@sprintf("%.3f", mean(M[:, k])) for k in axes(M, 2)], " "),
            join([@sprintf("%.0f", ess(reshape(M[:, k], :, 1, 1))[1]) for k in axes(M, 2)], " "), rhat(A)[1], ess(A)[1])
end
