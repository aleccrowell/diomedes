# Compare fitted gap models with PSIS-LOO (#9).
#
#   julia --project scripts/loo_compare.jl [models...] [data=<mode>]
#
# Uses the chains saved by `scripts/gap_fit.jl <data> <model> chain <k>`
# (output/gap_<data>_<model>_chain<k>.jls; data mode default the published fit's,
# see scripts/gap_data.jl). Defaults to every model in ALL with saved chains. All models are scored on the
# same rows (timed + lapped), so their elpd values are directly comparable.

using Diomedes, PosteriorStats, Serialization, Statistics

# `_fgam`: the fixed race-intercept prior used before #36; bare names learn the common race level
const ALL = ("t4", "pl_fgam", "pl_dur_fgam", "pl_decade_fgam", "pl_regime_fgam", "pl_rw_fgam",
             "pl_dur_regime_fgam", "pl_dur_rw_fgam", "pl_rw_kappa_fgam", "pl_dur_rw_kappa_fgam",
             "pl_dur_rw_kappa_age_fgam", "pl_dur_rw_kappa_big_fgam", "pl_dur_rw_kappa_age_big_fgam",
             "pl_dur_rw_kappa_age_slope_fgam", "pl_dur_rw_kappa_age_big_slope_fgam",
             "pl_dur_rw_kappa_age_big_slope_dev_fgam", "pl_dur_rw_kappa_age_big_slope_dev",
             "pl_dur_rw_kappa_age_big_slope_dev_circ_track", "pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk",
             "pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc",
             "pl_dur_rw_kappa_age_big_slope_dev_hgam")
include("gap_data.jl")
mode, args = data_arg(ARGS)
models = isempty(args) ? [m for m in ALL if isfile("output/gap_$(mode)_$(m)_chain1.jls")] : args

_, g = gap_data(mode)
println(g)

results = Dict{Symbol,Any}()
for m in models
    chain = gap_chains(mode, m)
    t = @elapsed ll = pointwise_loglik(chain, g, m)
    r = loo(ll)
    k = r.psis_result.pareto_shape
    println("\n== $m: $(Diomedes.Turing.FlexiChains.nchains(chain)) chains, pointwise log-lik in $(round(t; digits = 1)) s")
    show(stdout, MIME"text/plain"(), r.estimates)
    println("\n   Pareto k > 0.7: $(count(>(0.7), k)) of $(length(k)) rows",
            " (timed $(count(>(0.7), k[1:length(g.y)])), lapped $(count(>(0.7), k[(length(g.y) + 1):end])))")
    results[Symbol(m)] = r
end

if length(results) > 1
    println("\n== comparison")
    show(IOContext(stdout, :displaysize => (100, 250)), MIME"text/plain"(), compare(NamedTuple(results)))   # all columns
    println()
end
