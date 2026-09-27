# Pace + loss noise (#9): two independent sources of variation around the
# expected gap μ.
#
#     y = μ + ε + L,   ε ~ Normal(0, σ)          small symmetric pace noise
#                      L = 0 with prob 1 - π     clean race
#                      L ~ Exponential(mean λ)   with prob π: time lost to an
#                                                incident (slower only)
#
# L can only make a result slower. That matches incidents (spins, damage, slow
# stops), which the symmetric Student-t could only approximate. π and λ may
# vary by race, e.g. with race duration or era (see `paceloss_effects`).
#
# With residual x = y - μ:
#   density  f(x) = (1-π)·φ_σ(x) + π·EMG(x),
#            EMG(x) = (1/λ)·exp(σ²/(2λ²) - x/λ)·Φ(x/σ - σ/λ)   (exponentially modified Gaussian)
#   CDF      F(x) = Φ(x/σ) - π·G(x),   G(x) = exp(σ²/(2λ²) - x/λ)·Φ(x/σ - σ/λ)
# so an interval has probability  F(hi) - F(lo) = D + π·(G(lo) - G(hi)),
# with D = Φ(hi/σ) - Φ(lo/σ). Lapped cars sit in the right tail, where D
# underflows and G must be differenced in log space.

"log G(x), with G as above."
paceloss_logG(x, σ, λ) = σ^2 / (2λ^2) - x / λ + normlogcdf(x / σ - σ / λ)

"""
    paceloss_logpdf(x, σ, π, λ)

Log-density of the residual `x = y - μ` under pace noise + mixture loss.
"""
function paceloss_logpdf(x, σ, π, λ)
    lnorm = normlogpdf(x / σ) - log(σ)
    lemg = -log(λ) + σ^2 / (2λ^2) - x / λ + normlogcdf(x / σ - σ / λ)
    return logaddexp(log1p(-π) + lnorm, log(π) + lemg)
end

"""
    paceloss_loginterval(lo, hi, σ, π, λ)

`log P(lo < x < hi)` for the residual under pace noise + mixture loss, `lo < hi`.

Computed as the mixture of the two components' interval probabilities,
`(1-π)·D + π·P_EMG`, with `D = Φ(hi/σ) - Φ(lo/σ)` and `P_EMG = D + (G(lo) - G(hi))`.
- Right of G's peak, G(lo) > G(hi): both terms of P_EMG are positive, so it is
  summed in log space. This is where lapped cars usually sit.
- Left of the peak, G(lo) ≤ G(hi): P_EMG = D - |G(lo) - G(hi)|. In the far left
  tail both are nearly equal and the difference cancels catastrophically
  (rounding once made it negative, which crashed the step-size search). There
  P_EMG is set to 0 when the cancellation leaves nothing. That is accurate: a
  positive loss shifts mass right, so in the far left P_EMG ≪ D and the
  interval probability is dominated by (1-π)·D.
"""
function paceloss_loginterval(lo, hi, σ, π, λ)
    logD = logdiffΦ(lo / σ, hi / σ)
    lGlo, lGhi = paceloss_logG(lo, σ, λ), paceloss_logG(hi, σ, λ)
    if lGlo > lGhi
        logPemg = logaddexp(logD, lGlo + log1mexp(lGhi - lGlo))
    else
        r = lGhi + log1mexp(lGlo - lGhi) - logD          # log(|G(lo) - G(hi)| / D)
        logPemg = r < 0 ? logD + log1mexp(r) : oftype(logD, -Inf)
    end
    return logaddexp(log1p(-π) + logD, log(π) + logPemg)
end

"""
    paceloss_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr, g::GapData, season_counts)

Log-likelihood of `paceloss_effects`: per-row means as in `gap_loglik`
(field-centred race intercepts, sum-to-zero effects), pace noise `σ`, and
per-race incident probability `πr[race]` and mean loss `λr[race]`.
"""
function paceloss_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr, g, season_counts)
    a = σ_comp .* sum_to_zero(z_comp)
    b = σ_mach .* sum_to_zero_by(z_mach, g.mach_season, season_counts)
    ct = a[g.t_comp] .+ b[g.t_mach]
    cc = a[g.c_comp] .+ b[g.c_mach]
    T = promote_type(eltype(ct), eltype(γ), typeof(σ), eltype(πr), eltype(λr))
    csum, n = zeros(T, length(γ)), zeros(Int, length(γ))
    for (c, r) in zip(ct, g.t_race); csum[r] += c; n[r] += 1; end
    for (c, r) in zip(cc, g.c_race); csum[r] += c; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    ll = zero(T)
    for i in eachindex(g.y)
        r = g.t_race[i]
        ll += paceloss_logpdf(g.y[i] - (γ[r] + ct[i] - cbar[r]), σ, πr[r], λr[r])
    end
    for i in eachindex(g.lo)
        r = g.c_race[i]
        μ = γ[r] + cc[i] - cbar[r]
        ll += paceloss_loginterval(g.lo[i] - μ, g.hi[i] - μ, σ, πr[r], λr[r])
    end
    return ll
end

# Value and gradient of a scalar function of 4 arguments via ForwardDiff duals.
const FD = ReverseDiff.ForwardDiff
@inline function value_grad4(f, a, b, c, d)
    D = FD.Dual{Nothing}
    r = f(D(a, 1.0, 0.0, 0.0, 0.0), D(b, 0.0, 1.0, 0.0, 0.0), D(c, 0.0, 0.0, 1.0, 0.0),
          D(d, 0.0, 0.0, 0.0, 1.0))
    return FD.value(r), FD.partials(r)
end

# Fused reverse rule: per-row value and (μ, σ, π, λ) derivatives from 4-component
# duals, scattered into races and effects by the shared gap-model pullback.
function ChainRulesCore.rrule(::typeof(paceloss_loglik), γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr,
                              g, season_counts)
    st = gap_row_means(γ, z_comp, σ_comp, z_mach, σ_mach, g, season_counts)
    dμt, dμc = zeros(length(st.μt)), zeros(length(st.μc))
    dπ, dλ = zeros(length(πr)), zeros(length(λr))
    val, dσ = 0.0, 0.0
    for i in eachindex(st.μt)
        r, y = g.t_race[i], g.y[i]
        v, p = value_grad4((μ, s, q, l) -> paceloss_logpdf(y - μ, s, q, l), st.μt[i], σ, πr[r], λr[r])
        val += v; dμt[i] = p[1]; dσ += p[2]; dπ[r] += p[3]; dλ[r] += p[4]
    end
    for i in eachindex(st.μc)
        r, lo, hi = g.c_race[i], g.lo[i], g.hi[i]
        v, p = value_grad4((μ, s, q, l) -> paceloss_loginterval(lo - μ, hi - μ, s, q, l),
                           st.μc[i], σ, πr[r], λr[r])
        val += v; dμc[i] = p[1]; dσ += p[2]; dπ[r] += p[3]; dλ[r] += p[4]
    end
    d = gap_effects_pullback(dμt, dμc, st, σ_comp, σ_mach, g, season_counts)
    pullback(Δ) = (NoTangent(), Δ .* d.dγ, Δ .* d.dz_comp, Δ * d.dσ_comp, Δ .* d.dz_mach,
                   Δ * d.dσ_mach, Δ * dσ, Δ .* dπ, Δ .* dλ, NoTangent(), NoTangent())
    return val, pullback
end

# Uncompiled tapes only (see gap_loglik).
ReverseDiff.@grad_from_chainrules paceloss_loglik(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                  σ_comp::ReverseDiff.TrackedReal, z_mach::ReverseDiff.TrackedArray,
                                                  σ_mach::ReverseDiff.TrackedReal, σ::ReverseDiff.TrackedReal,
                                                  πr::ReverseDiff.TrackedArray, λr::ReverseDiff.TrackedArray,
                                                  g::GapData, season_counts::Vector{Int})

"""
Regulation regimes chosen a priori for changes plausibly affecting incident
frequency or time lost per incident (first season of each regime):

    1950 long races (500 km / 3 h)       1984 refuelling banned (turbo era)
    1958 races cut to 300 km / 2 h       1989 turbos banned; safety car from 1993
    1961 1.5 L formula                   1994 refuelling returns, driver aids banned
    1966 3.0 L formula                   2010 refuelling banned again
                                         2014 hybrid power units
"""
const REGIME_STARTS = [1950, 1958, 1961, 1966, 1984, 1989, 1994, 2010, 2014]

"""
    LossCovariates(g::GapData)

Race-level covariates for the loss component: standardised log race duration
(winner's time), and three era groupings of the race's season: decade,
regulation regime (`REGIME_STARTS`), and season index (for a random walk).
"""
struct LossCovariates
    log_duration::Vector{Float64}   # standardised log(winner's race minutes)
    decade::Vector{Int}             # 1 = earliest decade in the data
    n_decades::Int
    regime::Vector{Int}             # index into REGIME_STARTS
    n_regimes::Int
    season::Vector{Int}             # 1 = earliest season in the data
    n_seasons::Int
end
function LossCovariates(g::GapData)
    ld = log.(g.race_minutes)
    decades = g.race_season .÷ 10
    d0 = minimum(decades)
    regime = [searchsortedlast(REGIME_STARTS, s) for s in g.race_season]
    s0 = minimum(g.race_season)
    return LossCovariates((ld .- mean(ld)) ./ std(ld), decades .- d0 .+ 1, maximum(decades) - d0 + 1,
                          regime, length(REGIME_STARTS),
                          g.race_season .- s0 .+ 1, maximum(g.race_season) - s0 + 1)
end

const ERA_TERMS = (:none, :decade, :regime, :rw)

# Random-walk path from its increments, centred to mean zero over seasons:
# path[1] = 0, path[s] = Σ_{t<s} steps[t]. Written as a product with a constant
# lower-triangular matrix (no cumsum on tracked arrays).
rw_path(steps) = sum_to_zero(tril(ones(length(steps) + 1, length(steps)), -1) * steps)

"""
    race_loss(θ, cov::LossCovariates, loss_duration, era) -> (π_race, λ_race)

Per-race incident probability and mean loss from the loss parameters in `θ`.
It is shared by the model, the pointwise log-likelihoods and diagnostics, so
all three use the same definition. Era effects act on both logit π and log λ:

- `:decade` / `:regime`: hierarchical group effects, sum to zero, scales
  `τ_π_era`, `τ_λ_era`, standardised effects `z_π_era`, `z_λ_era`;
- `:rw`: random walks over seasons with Student-t(3) steps (mostly smooth, with
  occasional jumps: inferred era boundaries), step scales `τ_π_rw`, `τ_λ_rw`,
  standardised steps `e_π_rw`, `e_λ_rw`; paths centred over seasons.
"""
function race_loss(θ, cov::LossCovariates, loss_duration::Bool, era::Symbol)
    n = length(cov.log_duration)
    logitπ = fill(θ.a_π, n)
    logλ = fill(θ.a_λ, n)
    loss_duration && (logλ = logλ .+ θ.β_dur .* cov.log_duration)
    if era === :decade || era === :regime
        idx = era === :decade ? cov.decade : cov.regime
        logitπ = logitπ .+ θ.τ_π_era .* sum_to_zero(θ.z_π_era)[idx]
        logλ = logλ .+ θ.τ_λ_era .* sum_to_zero(θ.z_λ_era)[idx]
    elseif era === :rw
        logitπ = logitπ .+ rw_path(θ.τ_π_rw .* θ.e_π_rw)[cov.season]
        logλ = logλ .+ rw_path(θ.τ_λ_rw .* θ.e_λ_rw)[cov.season]
    end
    return logistic.(logitπ), exp.(logλ)
end

"Names of the loss parameters for a model variant, in sampling order."
function loss_param_names(loss_duration::Bool, era::Symbol)
    names = (:a_π, :a_λ)
    loss_duration && (names = (names..., :β_dur))
    era in (:decade, :regime) && (names = (names..., :τ_π_era, :τ_λ_era, :z_π_era, :z_λ_era))
    era === :rw && (names = (names..., :τ_π_rw, :τ_λ_rw, :e_π_rw, :e_λ_rw))
    return names
end

"""
    paceloss_effects(g::GapData; loss_duration = false, era = :none)

% gap to winner = race intercept + competitor effect + machine-season effect
+ pace noise + incident loss, with lapped finishers as interval-censored
observations. The pace structure is as in `gap_effects`. The noise is split
into two independent sources (see the file header):

- pace noise `σ` (half-normal(1), % units);
- incident probability `π_race = logistic(a_π [+ era])`, with prior centred on ~18%;
- mean incident loss `λ_race = exp(a_λ [+ β·log duration] [+ era])`, with
  prior centred on 2%.

`loss_duration` adds a slope on standardised log race duration (loss size
only). `era` (`:none`, `:decade`, `:regime`, `:rw`) adds era effects to both
the incident probability and the loss size (see `race_loss`).
"""
@model function paceloss_effects(g::GapData, season_counts::Vector{Int}, cov::LossCovariates,
                                 loss_duration::Bool, era::Symbol)
    σ_comp ~ truncated(Normal(0, 2); lower = 0)
    σ_mach ~ truncated(Normal(0, 2); lower = 0)
    σ ~ truncated(Normal(0, 1); lower = 0)
    z_comp ~ filldist(Normal(), length(g.competitors))
    z_mach ~ filldist(Normal(), length(g.machines))
    γ ~ filldist(Normal(0, 5), length(g.races))
    a_π ~ Normal(-1.5, 1)
    a_λ ~ Normal(log(2), 1)
    θ = (; a_π, a_λ)
    if loss_duration
        β_dur ~ Normal(0, 1)
        θ = (; θ..., β_dur)
    end
    if era === :decade || era === :regime
        k = era === :decade ? cov.n_decades : cov.n_regimes
        τ_π_era ~ truncated(Normal(0, 0.5); lower = 0)
        τ_λ_era ~ truncated(Normal(0, 0.5); lower = 0)
        z_π_era ~ filldist(Normal(), k)
        z_λ_era ~ filldist(Normal(), k)
        θ = (; θ..., τ_π_era, τ_λ_era, z_π_era, z_λ_era)
    elseif era === :rw
        τ_π_rw ~ truncated(Normal(0, 0.25); lower = 0)
        τ_λ_rw ~ truncated(Normal(0, 0.25); lower = 0)
        e_π_rw ~ filldist(TDist(3), cov.n_seasons - 1)
        e_λ_rw ~ filldist(TDist(3), cov.n_seasons - 1)
        θ = (; θ..., τ_π_rw, τ_λ_rw, e_π_rw, e_λ_rw)
    end
    πr, λr = race_loss(θ, cov, loss_duration, era)
    @addlogprob! paceloss_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr, g, season_counts)
end

function paceloss_effects(g::GapData; loss_duration::Bool = false, era::Symbol = :none)
    era in ERA_TERMS || throw(ArgumentError("era must be one of $ERA_TERMS"))
    cov = LossCovariates(g)
    era === :rw && cov.n_seasons < 2 && throw(ArgumentError("era = :rw needs at least 2 seasons"))
    return paceloss_effects(g, season_counts(g), cov, loss_duration, era)
end

"""
    fit_paceloss(g::GapData; loss_duration=false, era=:none, n_samples=1000, n_chains=1, ...)

Sample `paceloss_effects`, starting chains with race intercepts at each race's
mean timed gap and the loss parameters at their prior centres, era effects near
zero (jittered per chain). Sampler, progress and ensemble options as in `fit_gaps`.
"""
function fit_paceloss(g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                      n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                      sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                      progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100, kwargs...)
    adtype = hasproperty(sampler, :adtype) ? sampler.adtype : nothing
    adtype isa AutoReverseDiff && adtype.compile &&
        throw(ArgumentError("paceloss_effects needs uncompiled ReverseDiff (see gap_sampler)"))
    model = paceloss_effects(g; loss_duration, era)
    cov = model.args.cov
    race_mean = [let ys = g.y[g.t_race .== r]; isempty(ys) ? 0.0 : mean(ys) end
                 for r in eachindex(g.races)]
    scale() = 0.3 + 0.4 * rand(rng)
    function init()
        p = (; σ_comp = scale(), σ_mach = scale(), σ = scale(),
             z_comp = 0.1 .* randn(rng, length(g.competitors)),
             z_mach = 0.1 .* randn(rng, length(g.machines)),
             γ = race_mean .+ 0.1 .* randn(rng, length(race_mean)),
             a_π = -1.5 + 0.1 * randn(rng), a_λ = log(2) + 0.1 * randn(rng))
        loss_duration && (p = (; p..., β_dur = 0.1 * randn(rng)))
        if era === :decade || era === :regime
            k = era === :decade ? cov.n_decades : cov.n_regimes
            p = (; p..., τ_π_era = 0.1 + 0.1 * rand(rng), τ_λ_era = 0.1 + 0.1 * rand(rng),
                 z_π_era = 0.1 .* randn(rng, k), z_λ_era = 0.1 .* randn(rng, k))
        elseif era === :rw
            p = (; p..., τ_π_rw = 0.05 + 0.05 * rand(rng), τ_λ_rw = 0.05 + 0.05 * rand(rng),
                 e_π_rw = 0.1 .* randn(rng, cov.n_seasons - 1), e_λ_rw = 0.1 .* randn(rng, cov.n_seasons - 1))
        end
        return InitFromParams(p)
    end
    return run_nuts(model, [init() for _ in 1:n_chains]; n_samples, n_chains, ensemble, sampler,
                    rng, progress, progress_log, log_every, kwargs...)
end
