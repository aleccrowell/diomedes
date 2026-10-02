# Per-season incident rate and loss size implied by a fitted pace + loss model (#9).
#
#   julia --project scripts/incident_diagnostic.jl [model=pl] [n_draws=200]
#
# For each result and posterior draw:
#   r    = P(incident | data), the posterior probability that the result
#          included an incident loss;
#   E[L] = expected loss given an incident.
# Timed rows (residual x): given an incident, L ~ Normal(x - σ²/λ, σ) truncated
# to L ≥ 0 (closed form). Lapped rows (interval (lo, hi) on x): r is closed form,
# P_EMG = (P - (1-π)·D)/π, from the model's own interval probabilities. E[L] is
# approximated as the mean of Exponential(λ) truncated to (max(lo, 0), hi),
# which ignores the pace-noise smoothing (σ ≈ 0.6% vs intervals ≈ 1.7% wide),
# fine for reading trends. (Gauss–Laguerre quadrature was tried first and was
# off by up to 300% for intervals far in the tail.)
# Averages by season show where incident frequency and size change, as a guide
# to era structure. Writes output/incidents_by_season_<model>.csv.

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics
using Diomedes: _row_means, season_counts, LossCovariates, draw, logdiffΦ, race_loss,
                loss_param_names, paceloss_logpdf, paceloss_loginterval
using StatsFuns: normlogpdf, normlogcdf

model = get(ARGS, 1, "pl")
n_draws = parse(Int, get(ARGS, 2, "200"))
g = prepare_gaps(fetch_results(ErgastCSV(joinpath(@__DIR__, "..", "data")), 1950:2100))
chain = reduce(hcat, rehash_chain!.(deserialize.(filter(isfile, ["output/gap_all_$(model)_chain$(k).jls" for k in 1:8]))))
sc, cov = season_counts(g), LossCovariates(g)
spec = model_spec(model)
names = Diomedes.model_param_names(spec)

# Mean of Exponential(λ) truncated to (a, b), 0 ≤ a < b ≤ Inf.
function trunc_exp_mean(a, b, λ)
    w = b - a
    return isinf(w) ? a + λ : a + λ - w / expm1(w / λ)
end

# All heavy loops live in a function: at top level they would run on untyped
# globals, which is orders of magnitude slower.
function incident_sums(g, chain, names, picks, sc, cov, spec)
    nt, nc = length(g.y), length(g.lo)
    r_sum, rl_sum = zeros(nt + nc), zeros(nt + nc)   # Σ over draws of r and r·E[L|incident]
    for (i, c) in picks
        θ = draw(chain, names, i, c)
        μt, μc = _row_means(θ, g, sc, cov)
        πr, λr = race_loss(θ, cov, spec.loss_duration, spec.era)
        σ = θ.σ
        for k in 1:nt
            π, λ = πr[g.t_race[k]], λr[g.t_race[k]]; x = g.y[k] - μt[k]
            lemg = log(π) - log(λ) + σ^2 / (2λ^2) - x / λ + normlogcdf(x / σ - σ / λ)
            r = exp(lemg - paceloss_logpdf(x, σ, π, λ))
            m = x - σ^2 / λ                            # truncated-normal location
            EL = m + σ * exp(normlogpdf(m / σ) - normlogcdf(m / σ))
            r_sum[k] += r; rl_sum[k] += r * EL
        end
        for k in 1:nc
            π, λ = πr[g.c_race[k]], λr[g.c_race[k]]; lo, hi = g.lo[k] - μc[k], g.hi[k] - μc[k]
            logP = paceloss_loginterval(lo, hi, σ, π, λ)
            logD = logdiffΦ(lo / σ, hi / σ)
            # r = π·P_EMG/P = 1 - (1-π)·D/P, computed in log space (P_EMG can be tiny)
            r = clamp(-expm1(log1p(-π) + logD - logP), 0.0, 1.0)
            EL = hi <= 0 ? 0.0 : trunc_exp_mean(max(lo, 0.0), hi, λ)
            r_sum[nt + k] += r; rl_sum[nt + k] += r * EL
        end
    end
    return r_sum, rl_sum
end

nt, nc = length(g.y), length(g.lo)
ni, nch = size(chain[:σ])
picks = [(i, c) for c in 1:nch for i in round.(Int, range(1, ni; length = n_draws ÷ nch))]
t = @elapsed r_sum, rl_sum = incident_sums(g, chain, names, picks, sc, cov, spec)
n = length(picks)
println("draws used: $n in $(round(t; digits = 1)) s")

season = vcat(g.race_season[g.t_race], g.race_season[g.c_race])
lapped = vcat(falses(nt), trues(nc))
df = DataFrame(season = season, r = r_sum ./ n, rl = rl_sum ./ n, lapped = lapped)
by = combine(groupby(df, :season), nrow => :results, :lapped => mean => :lapped_share,
             :r => mean => :incident_rate,
             [:r, :rl] => ((r, rl) -> sum(rl) / sum(r)) => :loss_if_incident)
sort!(by, :season)
mkpath("output")
CSV.write("output/incidents_by_season_$(model).csv", by)
for row in eachrow(by)
    bar = repeat("█", round(Int, 40 * row.incident_rate))
    println(row.season, "  n=", lpad(row.results, 3), "  lapped ", lpad(round(Int, 100 * row.lapped_share), 3),
            "%  incident rate ", rpad(round(row.incident_rate; digits = 2), 4), "  loss|incident ",
            lpad(round(row.loss_if_incident; digits = 1), 5), "%  ", bar)
end
