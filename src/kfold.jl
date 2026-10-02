# Grouped K-fold cross-validation for the gap models (#18).
#
# PSIS-LOO is unreliable for rows with Pareto k > 0.7 (~3% of rows, mostly
# extreme results). K-fold CV refits the model without each fold and scores
# the held-out rows exactly. Folds are stratified within races, so each race
# keeps most of its rows in training and its intercept γ stays determined.

"""
    kfold_folds(g::GapData, K; rng) -> Vector{Int}

Fold (1..K) of each row, rows as in GapData (timed, then lapped). Within each
race the rows are shuffled and dealt round-robin from a random start, so every
fold holds out at most ⌈n/K⌉ of a race's n rows.
"""
function kfold_folds(g::GapData, K::Int; rng = Random.default_rng())
    races = vcat(g.t_race, g.c_race)
    folds = zeros(Int, length(races))
    byrace = [Int[] for _ in eachindex(g.races)]
    for (i, r) in enumerate(races); push!(byrace[r], i); end
    for rows in byrace
        isempty(rows) && continue
        start = rand(rng, 0:(K - 1))
        for (j, i) in enumerate(shuffle(rng, rows))
            folds[i] = mod1(start + j, K)
        end
    end
    return folds
end

"""
    subset_gaps(g::GapData, keep) -> GapData

The rows of `g` with `keep` true (a Bool per row: timed, then lapped). Competitor,
machine and race indices, and the age bins, are unchanged, so parameters of a
model fitted to the subset line up with `g`; levels with no rows left are
determined by their priors. Rows per age bin are recounted.
"""
function subset_gaps(g::GapData, keep::AbstractVector{Bool})
    nt = length(g.y)
    length(keep) == nt + length(g.lo) || throw(DimensionMismatch("keep needs one entry per row"))
    kt, kc = keep[1:nt], keep[(nt + 1):end]
    t_age, c_age = g.t_age[kt], g.c_age[kc]
    age_rows = [Float64(count(==(k), t_age) + count(==(k), c_age)) for k in eachindex(g.age_years)]
    # g.rows holds timed and lapped rows in source order; keep the matching ones
    tpos, cpos = findall(==(:timed), g.rows.kind), findall(==(:lapped), g.rows.kind)
    rows = g.rows[sort(vcat(tpos[kt], cpos[kc])), :]
    return GapData(g.y[kt], g.t_comp[kt], g.t_mach[kt], g.t_race[kt],
                   g.lo[kc], g.hi[kc], g.c_comp[kc], g.c_mach[kc], g.c_race[kc],
                   g.mach_season, g.race_season, g.race_minutes, t_age, c_age, g.age_years, age_rows,
                   g.competitors, g.machines, g.races, rows)
end

"""
    heldout_loglik(chain, g::GapData, train, model) -> Array{Float64,3}

Log-likelihood of each held-out row (`train` false) for every draw of `chain`,
a fit of `model` (see `model_spec`) to `subset_gaps(g, train)`. Shaped
(draws, chains, held-out rows), rows in GapData order.
"""
function heldout_loglik(chain, g::GapData, train::AbstractVector{Bool}, model::AbstractString)
    spec = model_spec(model)
    sc, cov = season_counts(g), LossCovariates(g)
    names = model_param_names(spec)
    age_basis = spec.age ? AgeCurveBasis(subset_gaps(g, train)) : nothing
    test = findall(!, train)
    rows(θ) = spec.family === :t4 ? gap_rows(θ, g; sc, train) :
        paceloss_rows(θ, g; loss_duration = spec.loss_duration, era = spec.era, sc, cov, train, age_basis)
    ni, nc = size(chain[:σ_comp])
    out = Array{Float64,3}(undef, ni, nc, length(test))
    for c in 1:nc, i in 1:ni
        out[i, c, :] = rows(draw(chain, names, i, c))[test]
    end
    return out
end

"""
    elpd_rows(ll) -> Vector{Float64}

Expected log predictive density of each row from its log-likelihood draws
(draws, chains, rows): log of the mean likelihood over draws.
"""
elpd_rows(ll::AbstractArray{<:Real,3}) =
    [logsumexp(vec(ll[:, :, j])) - log(size(ll, 1) * size(ll, 2)) for j in axes(ll, 3)]
