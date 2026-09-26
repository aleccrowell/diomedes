# Per-row log-likelihoods for model comparison with PSIS-LOO (#9).
#
# The gap models add their likelihood in one `@addlogprob!` call, so Turing
# cannot produce pointwise log-likelihoods itself. These functions recompute
# them per row (timed rows first, then lapped rows, as in GapData) from the
# posterior draws. They reuse the same per-row functions as the likelihoods,
# and the tests check that the rows sum to the fused likelihood.

# Centred effects and per-row means for one parameter draw.
function _row_means(θ, g::GapData, sc)
    a = θ.σ_comp .* sum_to_zero(θ.z_comp)
    b = θ.σ_mach .* sum_to_zero_by(θ.z_mach, g.mach_season, sc)
    ct = a[g.t_comp] .+ b[g.t_mach]
    cc = a[g.c_comp] .+ b[g.c_mach]
    csum, n = zeros(length(θ.γ)), zeros(Int, length(θ.γ))
    for (c, r) in zip(ct, g.t_race); csum[r] += c; n[r] += 1; end
    for (c, r) in zip(cc, g.c_race); csum[r] += c; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    return θ.γ[g.t_race] .+ ct .- cbar[g.t_race], θ.γ[g.c_race] .+ cc .- cbar[g.c_race]
end

"Per-row log-likelihoods of `gap_effects` for parameter values `θ` (a NamedTuple)."
function gap_rows(θ, g::GapData; noise::Noise = StudentTNoise(4), sc = season_counts(g))
    μt, μc = _row_means(θ, g, sc)
    σ = θ.σ_y
    timed = [logpdf_std(noise, (g.y[i] - μt[i]) / σ) - log(σ) for i in eachindex(μt)]
    lapped = [logdiffcdf_std(noise, (g.lo[i] - μc[i]) / σ, (g.hi[i] - μc[i]) / σ) for i in eachindex(μc)]
    return vcat(timed, lapped)
end

"Per-row log-likelihoods of `paceloss_effects` for parameter values `θ` (a NamedTuple)."
function paceloss_rows(θ, g::GapData; loss_duration::Bool = false, loss_era::Bool = false,
                       sc = season_counts(g), cov = LossCovariates(g))
    μt, μc = _row_means(θ, g, sc)
    logλ = fill(θ.a_λ, length(g.races))
    loss_duration && (logλ = logλ .+ θ.β_dur .* cov.log_duration)
    loss_era && (logλ = logλ .+ θ.τ_era .* sum_to_zero(θ.z_era)[cov.decade])
    λ, π = exp.(logλ), logistic(θ.a_π)
    timed = [paceloss_logpdf(g.y[i] - μt[i], θ.σ, π, λ[g.t_race[i]]) for i in eachindex(μt)]
    lapped = [paceloss_loginterval(g.lo[i] - μc[i], g.hi[i] - μc[i], θ.σ, π, λ[g.c_race[i]])
              for i in eachindex(μc)]
    return vcat(timed, lapped)
end

# The parameter values of draw (i, c) of a chain, as a NamedTuple.
draw(chain, names, i, c) = NamedTuple{names}(Tuple(chain[n][i, c] for n in names))

"""
    pointwise_loglik(chain, g::GapData, model) -> Array{Float64,3}

Log-likelihood of every row for every draw, shaped (draws, chains, rows) as
PosteriorStats' `loo` expects. `model` is `:t4` (Student-t(4) `gap_effects`),
`:pl`, `:pl_dur`, `:pl_era` or `:pl_both` (`paceloss_effects` with the
corresponding loss covariates).
"""
function pointwise_loglik(chain, g::GapData, model::Symbol)
    sc, cov = season_counts(g), LossCovariates(g)
    dur, era = model in (:pl_dur, :pl_both), model in (:pl_era, :pl_both)
    names = model === :t4 ? (:σ_comp, :σ_mach, :σ_y, :z_comp, :z_mach, :γ) :
        (:σ_comp, :σ_mach, :σ, :z_comp, :z_mach, :γ, :a_π, :a_λ,
         (dur ? (:β_dur,) : ())..., (era ? (:τ_era, :z_era) : ())...)
    rows(θ) = model === :t4 ? gap_rows(θ, g; sc) :
        paceloss_rows(θ, g; loss_duration = dur, loss_era = era, sc, cov)
    ni, nc = size(chain[:σ_comp])
    out = Array{Float64,3}(undef, ni, nc, length(g.y) + length(g.lo))
    for c in 1:nc, i in 1:ni
        out[i, c, :] = rows(draw(chain, names, i, c))
    end
    return out
end
