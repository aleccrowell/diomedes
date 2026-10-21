# Per-row PSIS-LOO diagnostics of saved race fits (#91): where the high Pareto k rows are.
#
#   julia --project scripts/pareto_k_rows.jl [models...] [data=<mode>]
#
# For each model (default: the published fit and the #81 fit), writes
# output/pareto_k_<data>_<model>.csv: one row per likelihood row (timed, then
# lapped, as in `pointwise_loglik`) with its Pareto k, pointwise elpd, the posterior
# mean residual (y − μ for timed rows, the interval midpoint − μ for lapped ones)
# and the row's metadata: season, race, driver, car, kind (timed, lapped, partial
# retiree), the race's condition (dry, mixed, wet) and walls score.

using Diomedes, DataFrames, CSV, PosteriorStats, Serialization, Statistics

include("gap_data.jl")
mode, args = data_arg(ARGS)
models = isempty(args) ? ["pl_dur_rw_kappa_age_big_slope_dev", "pl_dur_rw_kappa_age_big_slope_dev_circ_track"] : args

_, g = gap_data(mode)
println(g); flush(stdout)

# row metadata, in likelihood order (timed rows, then lapped rows)
function row_meta(g::GapData)
    rc = RaceCircuits(g, circuit_table())
    t, c = g.rows[g.rows.kind .== :timed, :], g.rows[g.rows.kind .== :lapped, :]
    partial = hasproperty(g.rows, :partial) ? coalesce.(t.partial, false) : falses(nrow(t))
    kind = vcat([p ? "partial" : "timed" for p in partial],
                [e ? "lapped_exact" : "lapped" for e in c.lapped_exact])
    race = vcat(g.t_race, g.c_race)
    cond = [rc.track[r, 3] == 1 ? "wet" : rc.track[r, 2] == 1 ? "mixed" : "dry" for r in race]
    return DataFrame(row = eachindex(race), kind = kind, season = g.race_season[race], race = g.races[race],
                     driver = g.competitors[vcat(g.t_comp, g.c_comp)], car = g.machines[vcat(g.t_mach, g.c_mach)],
                     condition = cond, walls = 2 .* rc.track[race, 1], speed = rc.x_raw[race, 1],
                     y = vcat(g.y, (g.lo .+ min.(g.hi, g.lo .+ 5)) ./ 2))   # open-ended intervals: lo + 2.5
end

meta = row_meta(g)
for m in models
    chain = gap_chains(mode, m)
    t = @elapsed ll = pointwise_loglik(chain, g, m)
    r = loo(ll)
    # posterior mean of each row's mean μ, over every 10th draw
    spec, sc = model_spec(m), Diomedes.season_counts(g)
    cov, names = LossCovariates(g, spec), Diomedes.model_param_names(spec)
    ni, nc = size(chain[:σ_comp])
    μ, n = zeros(nrow(meta)), 0
    for c in 1:nc, i in 1:10:ni
        μt, μc = Diomedes._row_means(Diomedes.draw(chain, names, i, c), g, sc, cov)
        μ .+= vcat(μt, μc); n += 1
    end
    df = copy(meta)
    df.k = r.psis_result.pareto_shape
    df.elpd = r.pointwise.elpd
    df.mu = μ ./ n
    df.resid = df.y .- df.mu
    out = "output/pareto_k_$(mode)_$(m).csv"
    CSV.write(out, df)
    println("$m: pointwise log-lik in $(round(t; digits = 1)) s; k > 0.7 in $(count(>(0.7), df.k)) rows -> $out")
    flush(stdout)
end
