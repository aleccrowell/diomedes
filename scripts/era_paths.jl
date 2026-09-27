# Inferred era paths from a random-walk pace + loss fit (#9).
#
#   julia --project scripts/era_paths.jl [model=pl_rw]
#
# For each season: posterior mean and 90% interval of the incident probability
# π and the mean incident loss λ (both at average race duration). Then the
# largest year-to-year steps in logit π and log λ (the inferred era
# boundaries), marking those that fall on an a-priori regulation boundary
# (REGIME_STARTS). Writes output/era_paths_<model>.csv.

using Diomedes, CSV, DataFrames, Serialization, Statistics
using Diomedes: rw_path, draw, loss_param_names
using LogExpFunctions: logistic

model = get(ARGS, 1, "pl_rw")
spec = model_spec(model)
spec.era === :rw || error("era_paths needs a random-walk model (pl[_dur]_rw)")
g = prepare_gaps(fetch_results(ErgastCSV(joinpath(@__DIR__, "..", "data")), 1950:2100))
cov = LossCovariates(g)
chain = reduce(hcat, deserialize.(filter(isfile, ["output/gap_all_$(model)_chain$(k).jls" for k in 1:8])))
names = loss_param_names(spec.loss_duration, spec.era)

function paths(chain, names, cov)
    ni, nc = size(chain[:a_π])
    S = cov.n_seasons
    lp, ll = zeros(ni * nc, S), zeros(ni * nc, S)
    k = 0
    for c in 1:nc, i in 1:ni
        θ = draw(chain, names, i, c)
        k += 1
        lp[k, :] = θ.a_π .+ rw_path(θ.τ_π_rw .* θ.e_π_rw)
        ll[k, :] = θ.a_λ .+ rw_path(θ.τ_λ_rw .* θ.e_λ_rw)
    end
    return lp, ll
end
lp, ll = paths(chain, names, cov)
seasons = minimum(g.race_season) .+ (0:(cov.n_seasons - 1))
q(x, p) = [quantile(c, p) for c in eachcol(x)]
df = DataFrame(season = seasons,
               π_mean = vec(mean(logistic.(lp); dims = 1)), π_q05 = logistic.(q(lp, 0.05)), π_q95 = logistic.(q(lp, 0.95)),
               λ_mean = vec(mean(exp.(ll); dims = 1)), λ_q05 = exp.(q(ll, 0.05)), λ_q95 = exp.(q(ll, 0.95)))
mkpath("output")
CSV.write("output/era_paths_$(model).csv", df)

println("season   incident prob π [90%]        mean loss λ % [90%]")
for r in eachrow(df)
    mark = r.season in REGIME_STARTS ? "  ◀ regime start" : ""
    println(r.season, "   ", rpad(round(r.π_mean; digits = 3), 6), "[", round(r.π_q05; digits = 3), ", ",
            round(r.π_q95; digits = 3), "]      ", rpad(round(r.λ_mean; digits = 2), 5), "[",
            round(r.λ_q05; digits = 2), ", ", round(r.λ_q95; digits = 2), "]", mark)
end

# Year-to-year steps: posterior mean step and P(step has a consistent sign)
for (label, x) in (("logit π", lp), ("log λ", ll))
    steps = diff(x; dims = 2)
    m = vec(mean(steps; dims = 1))
    psign = vec(max.(mean(steps .> 0; dims = 1), mean(steps .< 0; dims = 1)))
    order = sortperm(abs.(m); rev = true)
    println("\nLargest steps in $label (into season):")
    for j in order[1:8]
        s = seasons[j + 1]
        println("  $s  mean step $(rpad(round(m[j]; digits = 3), 7))  P(sign) $(round(psign[j]; digits = 2))",
                s in REGIME_STARTS ? "  ◀ regime start" : "")
    end
end
