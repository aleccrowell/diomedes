# Per-row log-likelihoods for model comparison with PSIS-LOO (#9).
#
# The gap models add their likelihood in one `@addlogprob!` call, so Turing
# cannot produce pointwise log-likelihoods itself. These functions recompute
# them per row (timed rows first, then lapped rows, as in GapData) from the
# posterior draws. They reuse the same per-row functions as the likelihoods,
# and the tests check that the rows sum to the fused likelihood.

# Centred effects and per-row means for one parameter draw (with the pace scale
# κ if θ has one, see `race_pace_scale`).
function _row_means(θ, g::GapData, sc, cov = LossCovariates(g))
    a = θ.σ_comp .* sum_to_zero(θ.z_comp)
    b = θ.σ_mach .* sum_to_zero_by(θ.z_mach, g.mach_season, sc)
    ct = a[g.t_comp] .+ b[g.t_mach]
    cc = a[g.c_comp] .+ b[g.c_mach]
    csum, n = zeros(length(θ.γ)), zeros(Int, length(θ.γ))
    for (c, r) in zip(ct, g.t_race); csum[r] += c; n[r] += 1; end
    for (c, r) in zip(cc, g.c_race); csum[r] += c; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    κ = haskey(θ, :τ_κ) ? race_pace_scale(θ, cov) : ones(length(θ.γ))
    return θ.γ[g.t_race] .+ κ[g.t_race] .* (ct .- cbar[g.t_race]),
           θ.γ[g.c_race] .+ κ[g.c_race] .* (cc .- cbar[g.c_race])
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
function paceloss_rows(θ, g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                       sc = season_counts(g), cov = LossCovariates(g))    # pace scale if θ has τ_κ
    μt, μc = _row_means(θ, g, sc, cov)
    πr, λr = race_loss(θ, cov, loss_duration, era)
    timed = [paceloss_logpdf(g.y[i] - μt[i], θ.σ, πr[g.t_race[i]], λr[g.t_race[i]]) for i in eachindex(μt)]
    lapped = [paceloss_loginterval(g.lo[i] - μc[i], g.hi[i] - μc[i], θ.σ, πr[g.c_race[i]], λr[g.c_race[i]])
              for i in eachindex(μc)]
    return vcat(timed, lapped)
end

# The parameter values of draw (i, c) of a chain, as a NamedTuple.
draw(chain, names, i, c) = NamedTuple{names}(Tuple(chain[n][i, c] for n in names))

"""
    model_spec(name) -> (; family, loss_duration, era)

Parse a model name used by the scripts: `t4` (Student-t(4) `gap_effects`) or
`pl[_dur][_<era>][_kappa]` (`paceloss_effects`), e.g. `pl`, `pl_dur`,
`pl_regime`, `pl_dur_rw_kappa`, with `<era>` one of `decade`, `regime`, `rw`
and `_kappa` adding the pace scale (#15).
"""
function model_spec(name::AbstractString)
    name == "t4" && return (; family = :t4, loss_duration = false, era = :none, pace_scale = false)
    parts = split(name, "_")
    first(parts) == "pl" || throw(ArgumentError("unknown model $name"))
    dur, kappa = "dur" in parts, "kappa" in parts
    eras = [Symbol(p) for p in parts[2:end] if p ∉ ("dur", "kappa")]
    length(eras) <= 1 && all(in(ERA_TERMS), eras) || throw(ArgumentError("unknown model $name"))
    return (; family = :pl, loss_duration = dur, era = isempty(eras) ? :none : only(eras),
            pace_scale = kappa)
end

"""
    pointwise_loglik(chain, g::GapData, model) -> Array{Float64,3}

Log-likelihood of every row for every draw, shaped (draws, chains, rows) as
PosteriorStats' `loo` expects. `model` is a model name (see `model_spec`).
"""
function pointwise_loglik(chain, g::GapData, model::AbstractString)
    spec = model_spec(model)
    sc, cov = season_counts(g), LossCovariates(g)
    names = spec.family === :t4 ? (:σ_comp, :σ_mach, :σ_y, :z_comp, :z_mach, :γ) :
        (:σ_comp, :σ_mach, :σ, :z_comp, :z_mach, :γ, loss_param_names(spec.loss_duration, spec.era)...,
         pace_param_names(spec.pace_scale)...)
    rows(θ) = spec.family === :t4 ? gap_rows(θ, g; sc) :
        paceloss_rows(θ, g; loss_duration = spec.loss_duration, era = spec.era, sc, cov)
    ni, nc = size(chain[:σ_comp])
    out = Array{Float64,3}(undef, ni, nc, length(g.y) + length(g.lo))
    for c in 1:nc, i in 1:ni
        out[i, c, :] = rows(draw(chain, names, i, c))
    end
    return out
end
