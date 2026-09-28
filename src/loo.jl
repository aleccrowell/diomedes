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
    centred = haskey(θ, :b_mach)          # pace-scale model (#15): centred cars, κ on drivers
    b = centred ? sum_to_zero_by(θ.b_mach, g.mach_season, sc) : θ.σ_mach .* sum_to_zero_by(θ.z_mach, g.mach_season, sc)
    κ = haskey(θ, :τ_κ) ? race_pace_scale(θ, cov) : ones(length(θ.γ))
    kb = centred ? ones(length(θ.γ)) : κ
    at, ac = a[g.t_comp], a[g.c_comp]
    if haskey(θ, :τ_age)                  # career curve (#15 stage 2)
        fage = age_curve(θ.τ_age, θ.e_age, AgeCurveBasis(g))
        at = at .+ [k == 0 ? 0.0 : fage[k] for k in g.t_age]
        ac = ac .+ [k == 0 ? 0.0 : fage[k] for k in g.c_age]
    end
    ct = κ[g.t_race] .* at .+ kb[g.t_race] .* b[g.t_mach]
    cc = κ[g.c_race] .* ac .+ kb[g.c_race] .* b[g.c_mach]
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
function paceloss_rows(θ, g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                       sc = season_counts(g), cov = LossCovariates(g))    # pace scale if θ has τ_κ
    μt, μc = _row_means(θ, g, sc, cov)
    πr, λr = race_loss(θ, cov, loss_duration, era)
    if haskey(θ, :a_ρ)                    # two-component incident loss (#16)
        ρ, λ2 = logistic(θ.a_ρ), exp(θ.a_λ + θ.δ_λ2)
        timed = [pacebig_logpdf(g.y[i] - μt[i], θ.σ, πr[g.t_race[i]], λr[g.t_race[i]], ρ, λ2) for i in eachindex(μt)]
        lapped = [pacebig_loginterval(g.lo[i] - μc[i], g.hi[i] - μc[i], θ.σ, πr[g.c_race[i]], λr[g.c_race[i]], ρ, λ2)
                  for i in eachindex(μc)]
        return vcat(timed, lapped)
    end
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
`pl[_dur][_<era>][_kappa][_age]` (`paceloss_effects`), e.g. `pl`, `pl_dur`,
`pl_regime`, `pl_dur_rw_kappa_age`, with `<era>` one of `decade`, `regime`,
`rw`, `_kappa` adding the pace scale and `_age` the career curve (#15).
"""
function model_spec(name::AbstractString)
    name == "t4" && return (; family = :t4, loss_duration = false, era = :none, pace_scale = false, age = false,
                            big_loss = false)
    parts = split(name, "_")
    first(parts) == "pl" || throw(ArgumentError("unknown model $name"))
    dur, kappa, age, big = "dur" in parts, "kappa" in parts, "age" in parts, "big" in parts
    eras = [Symbol(p) for p in parts[2:end] if p ∉ ("dur", "kappa", "age", "big")]
    length(eras) <= 1 && all(in(ERA_TERMS), eras) || throw(ArgumentError("unknown model $name"))
    return (; family = :pl, loss_duration = dur, era = isempty(eras) ? :none : only(eras),
            pace_scale = kappa, age, big_loss = big)
end

"""
    pointwise_loglik(chain, g::GapData, model) -> Array{Float64,3}

Log-likelihood of every row for every draw, shaped (draws, chains, rows) as
PosteriorStats' `loo` expects. `model` is a model name (see `model_spec`).
"""
function pointwise_loglik(chain, g::GapData, model::AbstractString)
    spec = model_spec(model)
    sc, cov = season_counts(g), LossCovariates(g)
    names = model_param_names(spec)
    rows(θ) = spec.family === :t4 ? gap_rows(θ, g; sc) :
        paceloss_rows(θ, g; loss_duration = spec.loss_duration, era = spec.era, sc, cov)
    ni, nc = size(chain[:σ_comp])
    out = Array{Float64,3}(undef, ni, nc, length(g.y) + length(g.lo))
    for c in 1:nc, i in 1:ni
        out[i, c, :] = rows(draw(chain, names, i, c))
    end
    return out
end

"Names of the parameters of a model variant (see `model_spec`) needed to evaluate it."
function model_param_names(spec)
    spec.family === :t4 && return (:σ_comp, :σ_mach, :σ_y, :z_comp, :z_mach, :γ)
    cars = spec.pace_scale ? () : (:z_mach,)          # pace-scale models use centred b_mach
    return (:σ_comp, :σ_mach, :σ, :z_comp, cars..., :γ, loss_param_names(spec.loss_duration, spec.era)...,
            pace_param_names(spec.pace_scale)..., age_param_names(spec.age)..., big_param_names(spec.big_loss)...)
end
