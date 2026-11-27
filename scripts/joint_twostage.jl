# #128, stage 1: a two-stage prototype of shared driver ability between the race and lap models.
#
#   julia --project scripts/joint_twostage.jl [race=<drivers.csv>] [laps=<drivers.csv>] [minlaps=0]
#
# Each model's driver posterior (mean, sd) is treated as data on the drivers in both:
#
#     race_i ~ N(a_i + δr_i, sr_i),   lap_i ~ N(c + λ·a_i + δl_i, sl_i),
#     a_i ~ N(0, σ_a),  δr_i ~ N(0, τ_r),  δl_i ~ N(0, τ_l).
#
# - a is the shared ability, on the race model's scale (% of race time).
# - λ maps it to lap time. It should be well under 1, as clean laps leave out much of
#   what separates drivers in races.
# - c absorbs the different centring: the race model centres over all 1950–2026 drivers,
#   the lap model over its own seasons.
# This is a meta-analysis: the posteriors' own shrinkage is counted again, and neither
# model's other terms adjust. It's meant to check the design and get starting values
# (λ, the deviation spreads, and which drivers move) before the full joint fit.
using CSV, DataFrames, Turing, Statistics, Printf, Random, LinearAlgebra
import MCMCDiagnosticTools as MDT

opt(k, d) = (i = findfirst(a -> startswith(a, k * "="), ARGS); i === nothing ? d : String(split(ARGS[i], "="; limit = 2)[2]))
race = CSV.read(opt("race", "docs/src/assets/data/race/drivers.csv"), DataFrame)
laps = CSV.read(opt("laps", "docs/src/assets/data/laps/drivers.csv"), DataFrame)
minlaps = parse(Int, opt("minlaps", "0"))
j = innerjoin(select(race, :id, :name, :mean => :r, :sd => :sr), select(laps, :id, :mean => :l, :sd => :sl, :laps); on = :id)
filter!(row -> row.laps >= minlaps, j)
n = nrow(j)
println("drivers in both: $n (laps ≥ $minlaps); raw corr(race, lap) = $(round(cor(j.r, j.l); digits = 3)), ",
        "OLS slope of lap on race = $(round(cov(j.r, j.l) / var(j.r); digits = 3))")

@model function shared(r, sr, l, sl)
    n = length(r)
    σ_a ~ truncated(Normal(0, 1); lower = 0)
    τ_r ~ truncated(Normal(0, 0.5); lower = 0)
    τ_l ~ truncated(Normal(0, 0.5); lower = 0)
    λ ~ Normal(0.5, 0.5)
    c ~ Normal(0, 1)
    z ~ filldist(Normal(), n)
    zr ~ filldist(Normal(), n)
    zl ~ filldist(Normal(), n)
    a = σ_a .* z
    r ~ MvNormal(a .+ τ_r .* zr, Diagonal(sr .^ 2))
    l ~ MvNormal(c .+ λ .* a .+ τ_l .* zl, Diagonal(sl .^ 2))
    return a
end

m = shared(j.r, j.sr, j.l, j.sl)
ch = sample(Xoshiro(128), m, NUTS(1000, 0.9), MCMCThreads(), 1000, 4; progress = false)
for p in (:σ_a, :τ_r, :τ_l, :λ, :c)
    x = Array(ch[p])
    @printf("%-4s %.3f ± %.3f  (90%% %.3f to %.3f; R-hat %.3f, ESS %.0f)\n", p, mean(x), std(x),
            quantile(vec(x), 0.05), quantile(vec(x), 0.95), MDT.rhat(reshape(x, size(x)..., 1))[1], MDT.ess(reshape(x, size(x)..., 1))[1])
end
divs = sum(Array(ch[:numerical_error]))
println("divergences: $divs")

# each driver's shared ability against the race model's own estimate (race scale)
Z = permutedims(stack(vec(ch[:z])))           # draws × drivers (FlexiChains stores z as one vector)
a = Z .* vec(Array(ch[:σ_a]))
j.a_mean = vec(mean(a; dims = 1)); j.a_sd = vec(std(a; dims = 1))
j.shift = j.a_mean .- j.r
sort!(j, :shift)
println("\nlargest moves of shared ability against the race model's effect (race scale, negative = faster):")
for row in eachrow(vcat(first(j, 5), last(j, 5)))
    @printf("  %-22s race %+.2f ± %.2f, lap %+.2f ± %.2f (%5d laps) → shared %+.2f ± %.2f (shift %+.2f)\n",
            row.name, row.r, row.sr, row.l, row.sl, row.laps, row.a_mean, row.a_sd, row.shift)
end
@printf("\nsd shrink: race sd median %.3f → shared %.3f\n", median(j.sr), median(j.a_sd))
