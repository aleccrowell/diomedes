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

"""
    emg_logkernel(x, σ, λ)

`σ²/(2λ²) - x/λ + log Φ(z)` with `z = x/σ - σ/λ`: log G(x), and the EMG log
density up to `-log λ`. When λ ≪ σ the three terms are huge and nearly cancel
(at λ ~ 1e-11 the naive sum gave a log density of +2.7e7 with NaN gradients,
which trapped NUTS). For z < 0, Φ(z) = erfcx(-z/√2)·exp(-z²/2)/2 cancels them
exactly: the kernel is -x²/(2σ²) + log(erfcx(-z/√2)/2).
"""
function emg_logkernel(x, σ, λ)
    z = x / σ - σ / λ
    return z < 0 ? -x^2 / (2σ^2) + log(erfcx(-z / sqrt(2)) / 2) : σ^2 / (2λ^2) - x / λ + normlogcdf(z)
end

"log G(x), with G as above."
paceloss_logG(x, σ, λ) = emg_logkernel(x, σ, λ)

"""
    paceloss_logpdf(x, σ, π, λ)

Log-density of the residual `x = y - μ` under pace noise + mixture loss.
"""
function paceloss_logpdf(x, σ, π, λ)
    lnorm = normlogpdf(x / σ) - log(σ)
    lemg = -log(λ) + emg_logkernel(x, σ, λ)
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
    return logaddexp(log1p(-π) + logD, log(π) + log_pemg(lo, hi, σ, λ, logD))
end

"log P_EMG(lo < x < hi) for loss mean `λ`, given `logD = log(Φ(hi/σ) - Φ(lo/σ))` (see above)."
function log_pemg(lo, hi, σ, λ, logD)
    lGlo, lGhi = paceloss_logG(lo, σ, λ), paceloss_logG(hi, σ, λ)
    if lGlo > lGhi
        return logaddexp(logD, lGlo + log1mexp(lGhi - lGlo))
    else
        r = lGhi + log1mexp(lGlo - lGhi) - logD          # log(|G(lo) - G(hi)| / D)
        return r < 0 ? logD + log1mexp(r) : oftype(logD, -Inf)
    end
end

log_emg_pdf(x, σ, λ) = -log(λ) + emg_logkernel(x, σ, λ)

"""
    pacebig_logpdf(x, σ, π, λ, ρ, λ2)
    pacebig_loginterval(lo, hi, σ, π, λ, ρ, λ2)

Pace + loss noise with a **two-component incident loss** (#16): an incident
(probability π) costs Exponential(mean λ) with probability 1 - ρ, or a big loss
Exponential(mean λ2 > λ) with probability ρ (repairs, long stops, many laps
down):
    f(x) = (1-π)·φ_σ(x) + π·[(1-ρ)·EMG(x; λ) + ρ·EMG(x; λ2)].
With ρ → 0 these reduce to `paceloss_logpdf` / `paceloss_loginterval`.
"""
function pacebig_logpdf(x, σ, π, λ, ρ, λ2)
    lnorm = normlogpdf(x / σ) - log(σ)
    lloss = logaddexp(log1p(-ρ) + log_emg_pdf(x, σ, λ), log(ρ) + log_emg_pdf(x, σ, λ2))
    return logaddexp(log1p(-π) + lnorm, log(π) + lloss)
end

function pacebig_loginterval(lo, hi, σ, π, λ, ρ, λ2)
    logD = logdiffΦ(lo / σ, hi / σ)
    lloss = logaddexp(log1p(-ρ) + log_pemg(lo, hi, σ, λ, logD), log(ρ) + log_pemg(lo, hi, σ, λ2, logD))
    return logaddexp(log1p(-π) + logD, log(π) + lloss)
end

"""
    paceloss_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr, κr, g::GapData, season_counts)

Log-likelihood of `paceloss_effects`: per-row means as in `gap_loglik`
(field-centred race intercepts, sum-to-zero effects), pace noise `σ`, and
per-race incident probability `πr[race]` and mean loss `λr[race]`. `κr` is a
per-race pace scale (`nothing` = 1): μ = γ + κ·(c - c̄) with c = driver + car.
"""
function paceloss_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr, κr, g, season_counts)
    a = σ_comp .* sum_to_zero(z_comp)
    b = σ_mach .* sum_to_zero_by(z_mach, g.mach_season, season_counts)
    ct = a[g.t_comp] .+ b[g.t_mach]
    cc = a[g.c_comp] .+ b[g.c_mach]
    T = promote_type(eltype(ct), eltype(γ), typeof(σ), eltype(πr), eltype(λr),
                     κr === nothing ? Float64 : eltype(κr))
    csum, n = zeros(T, length(γ)), zeros(Int, length(γ))
    for (c, r) in zip(ct, g.t_race); csum[r] += c; n[r] += 1; end
    for (c, r) in zip(cc, g.c_race); csum[r] += c; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    k(r) = κr === nothing ? one(T) : κr[r]
    ll = zero(T)
    for i in eachindex(g.y)
        r = g.t_race[i]
        ll += paceloss_logpdf(g.y[i] - (γ[r] + k(r) * (ct[i] - cbar[r])), σ, πr[r], λr[r])
    end
    for i in eachindex(g.lo)
        r = g.c_race[i]
        μ = γ[r] + k(r) * (cc[i] - cbar[r])
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

# Value and gradient of a scalar function of N arguments via ForwardDiff duals.
@inline function value_gradN(f, xs::Vararg{Real,N}) where {N}
    D = FD.Dual{Nothing}
    duals = ntuple(i -> D(xs[i], ntuple(j -> Float64(i == j), N)...), N)
    r = f(duals...)
    return FD.value(r), FD.partials(r)
end

# Per-row values and derivatives with the big-loss component (#16): as
# paceloss_row_grads, plus scalar derivatives for ρ and λ2.
function pacebig_row_grads(st, σ, πr, λr, ρ, λ2, g)
    dμt, dμc = zeros(length(st.μt)), zeros(length(st.μc))
    dπ, dλ = zeros(length(πr)), zeros(length(λr))
    val, dσ, dρ, dλ2 = 0.0, 0.0, 0.0, 0.0
    for i in eachindex(st.μt)
        r, y = g.t_race[i], g.y[i]
        v, p = value_gradN((μ, s, q, l, h, l2) -> pacebig_logpdf(y - μ, s, q, l, h, l2),
                           st.μt[i], σ, πr[r], λr[r], ρ, λ2)
        val += v; dμt[i] = p[1]; dσ += p[2]; dπ[r] += p[3]; dλ[r] += p[4]; dρ += p[5]; dλ2 += p[6]
    end
    for i in eachindex(st.μc)
        r, lo, hi = g.c_race[i], g.lo[i], g.hi[i]
        v, p = value_gradN((μ, s, q, l, h, l2) -> pacebig_loginterval(lo - μ, hi - μ, s, q, l, h, l2),
                           st.μc[i], σ, πr[r], λr[r], ρ, λ2)
        val += v; dμc[i] = p[1]; dσ += p[2]; dπ[r] += p[3]; dλ[r] += p[4]; dρ += p[5]; dλ2 += p[6]
    end
    return val, dμt, dμc, dσ, dπ, dλ, dρ, dλ2
end

# Per-row values and (μ, σ, π, λ) derivatives from 4-component duals, given
# the row means in `st` (from row_means_ab).
function paceloss_row_grads(st, σ, πr, λr, g)
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
    return val, dμt, dμc, dσ, dπ, dλ
end

# Fused reverse rule: per-row derivatives scattered into races and effects (and
# κ) by the shared gap-model pullback.
function ChainRulesCore.rrule(::typeof(paceloss_loglik), γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr,
                              κr, g, season_counts)
    st = gap_row_means(γ, z_comp, σ_comp, z_mach, σ_mach, g, season_counts; κ = κr)
    val, dμt, dμc, dσ, dπ, dλ = paceloss_row_grads(st, σ, πr, λr, g)
    d = gap_effects_pullback(dμt, dμc, st, σ_comp, σ_mach, g, season_counts; κ = κr)
    dκ = κr === nothing ? NoTangent() : d.dκ
    pullback(Δ) = (NoTangent(), Δ .* d.dγ, Δ .* d.dz_comp, Δ * d.dσ_comp, Δ .* d.dz_mach,
                   Δ * d.dσ_mach, Δ * dσ, Δ .* dπ, Δ .* dλ, dκ isa NoTangent ? dκ : Δ .* dκ,
                   NoTangent(), NoTangent())
    return val, pullback
end

"""
    paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, fage, g::GapData, season_counts)

Log-likelihood of the pace-scale model (#15) with **centred** car effects.
Driver effects `a = σ_comp·(z_comp - mean)` are scaled by the per-race `κr`;
car effects are `b = b_mach - season mean` (their spread by season is set in
the prior, b_mach ~ N(0, σ_mach·κ_season)), so μ = γ + κ·a + b - (field mean).
Centred car effects avoid the ridge between κ_season and the spread of that
season's standardised effects that the non-centred form has; each car-season
has ~30+ results, so its effect is well determined by the data.

`fage` (`nothing`, or a value per age bin; see `age_curve`) adds a career
curve to each row's driver part: κ·(a[driver] + fage[age bin]). Rows of
unknown age (bin 0) get no age term.
"""
function paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, fage, g, season_counts)
    return paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, fage, nothing, g, season_counts)
end

# `big` = nothing, or [ρ, λ2] for the two-component incident loss (#16).
function paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, fage, big, g, season_counts)
    a = σ_comp .* sum_to_zero(z_comp)
    b = sum_to_zero_by(b_mach, g.mach_season, season_counts)
    T = promote_type(eltype(a), eltype(b), eltype(γ), typeof(σ), eltype(πr), eltype(λr), eltype(κr),
                     fage === nothing ? Float64 : eltype(fage), big === nothing ? Float64 : eltype(big))
    lpdf(x, r) = big === nothing ? paceloss_logpdf(x, σ, πr[r], λr[r]) :
        pacebig_logpdf(x, σ, πr[r], λr[r], big[1], big[2])
    lint(lo, hi, r) = big === nothing ? paceloss_loginterval(lo, hi, σ, πr[r], λr[r]) :
        pacebig_loginterval(lo, hi, σ, πr[r], λr[r], big[1], big[2])
    fa(bin) = (fage === nothing || bin == 0) ? zero(T) : fage[bin]
    ct = [κr[g.t_race[i]] * (a[g.t_comp[i]] + fa(g.t_age[i])) + b[g.t_mach[i]] for i in eachindex(g.t_race)]
    cc = [κr[g.c_race[i]] * (a[g.c_comp[i]] + fa(g.c_age[i])) + b[g.c_mach[i]] for i in eachindex(g.c_race)]
    csum, n = zeros(T, length(γ)), zeros(Int, length(γ))
    for (c, r) in zip(ct, g.t_race); csum[r] += c; n[r] += 1; end
    for (c, r) in zip(cc, g.c_race); csum[r] += c; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    ll = zero(T)
    for i in eachindex(g.y)
        r = g.t_race[i]
        ll += lpdf(g.y[i] - (γ[r] + ct[i] - cbar[r]), r)
    end
    for i in eachindex(g.lo)
        r = g.c_race[i]
        μ = γ[r] + cc[i] - cbar[r]
        ll += lint(g.lo[i] - μ, g.hi[i] - μ, r)
    end
    return ll
end

function ChainRulesCore.rrule(::typeof(paceloss_loglik_cc), γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr,
                              fage, g::GapData, season_counts)
    val, pb = ChainRulesCore.rrule(paceloss_loglik_cc, γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, fage,
                                   nothing, g, season_counts)
    pullback(Δ) = (d = pb(Δ); (d[1:10]..., d[12], d[13]))     # drop the `big` slot
    return val, pullback
end

function ChainRulesCore.rrule(::typeof(paceloss_loglik_cc), γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr,
                              fage, big, g, season_counts)
    zc = z_comp .- mean(z_comp)
    b = b_mach .- group_means(b_mach, g.mach_season, season_counts)[g.mach_season]
    st = row_means_ab(γ, σ_comp .* zc, b, g; κ = κr, κ_cars = false, fage)
    if big === nothing
        val, dμt, dμc, dσ, dπ, dλ = paceloss_row_grads(st, σ, πr, λr, g)
        dbig = NoTangent()
    else
        val, dμt, dμc, dσ, dπ, dλ, dρ, dλ2 = pacebig_row_grads(st, σ, πr, λr, big[1], big[2], g)
        dbig = [dρ, dλ2]
    end
    d = row_pullback_ab(dμt, dμc, st, g; κ = κr, κ_cars = false,
                        nage = fage === nothing ? 0 : length(fage))
    dσ_comp = sum(d.da .* zc)
    dz_comp = σ_comp .* (d.da .- mean(d.da))
    db_mach = d.db .- group_means(d.db, g.mach_season, season_counts)[g.mach_season]
    dfage = fage === nothing ? NoTangent() : d.dfage
    pullback(Δ) = (NoTangent(), Δ .* d.dγ, Δ .* dz_comp, Δ * dσ_comp, Δ .* db_mach, Δ * dσ,
                   Δ .* dπ, Δ .* dλ, Δ .* d.dκ, dfage isa NoTangent ? dfage : Δ .* dfage,
                   dbig isa NoTangent ? dbig : Δ .* dbig, NoTangent(), NoTangent())
    return val, pullback
end

# With the big-loss component (#16), with and without an age curve.
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     fage::Nothing, big::ReverseDiff.TrackedArray, g::GapData,
                                                     season_counts::Vector{Int})
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     fage::ReverseDiff.TrackedArray, big::ReverseDiff.TrackedArray,
                                                     g::GapData, season_counts::Vector{Int})

# With and without an age curve (uncompiled tapes only; see gap_loglik).
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     fage::Nothing, g::GapData, season_counts::Vector{Int})
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     fage::ReverseDiff.TrackedArray, g::GapData,
                                                     season_counts::Vector{Int})

# Uncompiled tapes only (see gap_loglik); one binding without and one with a pace scale.
ReverseDiff.@grad_from_chainrules paceloss_loglik(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                  σ_comp::ReverseDiff.TrackedReal, z_mach::ReverseDiff.TrackedArray,
                                                  σ_mach::ReverseDiff.TrackedReal, σ::ReverseDiff.TrackedReal,
                                                  πr::ReverseDiff.TrackedArray, λr::ReverseDiff.TrackedArray,
                                                  κr::Nothing, g::GapData, season_counts::Vector{Int})
ReverseDiff.@grad_from_chainrules paceloss_loglik(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                  σ_comp::ReverseDiff.TrackedReal, z_mach::ReverseDiff.TrackedArray,
                                                  σ_mach::ReverseDiff.TrackedReal, σ::ReverseDiff.TrackedReal,
                                                  πr::ReverseDiff.TrackedArray, λr::ReverseDiff.TrackedArray,
                                                  κr::ReverseDiff.TrackedArray, g::GapData,
                                                  season_counts::Vector{Int})

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
    mach_season::Vector{Int}        # season index (as `season`) of each machine level
end
function LossCovariates(g::GapData)
    ld = log.(g.race_minutes)
    decades = g.race_season .÷ 10
    d0 = minimum(decades)
    regime = [searchsortedlast(REGIME_STARTS, s) for s in g.race_season]
    s0 = minimum(g.race_season)
    data_seasons = sort(unique(g.race_season))         # GapData.mach_season indexes these
    return LossCovariates((ld .- mean(ld)) ./ std(ld), decades .- d0 .+ 1, maximum(decades) - d0 + 1,
                          regime, length(REGIME_STARTS),
                          g.race_season .- s0 .+ 1, maximum(g.race_season) - s0 + 1,
                          data_seasons[g.mach_season] .- s0 .+ 1)
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
all three use the same definition. Race duration (standardised log winner's
time) has separate slopes on logit π (`β_dur_π`: more laps, more chances of
an incident) and log λ (`β_dur_λ`: loss size in % of a longer or shorter race).
Era effects act on both logit π and log λ:

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
    if loss_duration
        logitπ = logitπ .+ θ.β_dur_π .* cov.log_duration
        logλ = logλ .+ θ.β_dur_λ .* cov.log_duration
    end
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
    loss_duration && (names = (names..., :β_dur_π, :β_dur_λ))
    era in (:decade, :regime) && (names = (names..., :τ_π_era, :τ_λ_era, :z_π_era, :z_λ_era))
    era === :rw && (names = (names..., :τ_π_rw, :τ_λ_rw, :e_π_rw, :e_λ_rw))
    return names
end

"""
    AgeCurveBasis(g::GapData)

Fixed matrices for the population career curve (#15 stage 2): `walk` maps
standardised steps to a random walk over one-year age bins (0 at the youngest
bin), and `proj` removes the constant and linear parts of a curve (weighted
least squares, weights = rows per bin).

Only the curve's shape up to a linear tilt is identifiable: with constant
driver effects and field-centred races, a linear age term
β·age = β·year - β·birth year is removed by the field centring (β·year) and
absorbed by the driver effect (β·birth year). This is the age–period–cohort
problem. So the curve is projected onto functions with no constant or linear
part in age; its curvature (e.g. peak then decline) is identified, but not
the absolute age of the peak.
"""
struct AgeCurveBasis
    walk::Matrix{Float64}
    proj::Matrix{Float64}
end
function AgeCurveBasis(g::GapData)
    n = length(g.age_years)
    walk = tril(ones(n, n - 1), -1)
    X = hcat(ones(n), Float64.(g.age_years))
    W = Diagonal(g.age_rows)
    proj = I - X * ((X' * W * X) \ (X' * W))
    return AgeCurveBasis(walk, proj)
end

"Career curve per age bin from step scale `τ_age` and standardised steps `e_age`."
age_curve(τ_age, e_age, basis::AgeCurveBasis) = basis.proj * (basis.walk * (τ_age .* e_age))

"Names of the pace-scale parameters (#15), in sampling order (after the loss parameters)."
pace_param_names(pace_scale::Bool) = pace_scale ? (:τ_κ, :e_κ, :b_mach) : ()

"Names of the age-curve parameters (#15 stage 2), in sampling order (after the pace scale)."
age_param_names(age::Bool) = age ? (:τ_age, :e_age) : ()

"Names of the big-loss parameters (#16), in sampling order (after the age curve)."
big_param_names(big::Bool) = big ? (:a_ρ, :δ_λ2) : ()

"""
    race_pace_scale(θ, cov) -> Vector

Per-race pace scale κ (#15): `log κ_season` is a random walk over seasons with
Student-t(3) steps (step scale `τ_κ`, standardised steps `e_κ`), centred so the
geometric mean of κ over seasons is 1. Driver and car effects are then in
average-season units.
"""
season_pace_scale(θ) = exp.(rw_path(θ.τ_κ .* θ.e_κ))
race_pace_scale(θ, cov::LossCovariates) = season_pace_scale(θ)[cov.season]

"""
    paceloss_effects(g::GapData; loss_duration = false, era = :none, pace_scale = false)

% gap to winner = race intercept + competitor effect + machine-season effect
+ pace noise + incident loss, with lapped finishers as interval-censored
observations. The pace structure is as in `gap_effects`. The noise is split
into two independent sources (see the file header):

- pace noise `σ` (half-normal(1), % units);
- incident probability `π_race = logistic(a_π [+ β_π·log duration] [+ era])`,
  with prior centred on ~18%;
- mean incident loss `λ_race = exp(a_λ [+ β_λ·log duration] [+ era])`, with
  prior centred on 2%.

`loss_duration` adds slopes on standardised log race duration to both the
incident probability and the loss size. `era` (`:none`, `:decade`, `:regime`,
`:rw`) adds era effects to both (see `race_loss`). Duration varies within
seasons as well as between them, so with both terms the era effects capture
only what duration does not.

`big_loss` (#16, needs `pace_scale`) splits incident losses into normal
incidents (mean λ) and big losses (share ρ, mean λ2 = λ_baseline·e^δ, δ > 0),
so extreme results (repairs, many laps down) don't have to be explained by λ
(see `pacebig_logpdf`).

`age` (#15 stage 2, needs `pace_scale`) adds a population career curve over
driver age, with only its shape up to a linear tilt identified (see
`AgeCurveBasis`).

`pace_scale` (#15) lets the spread of pace, in % terms, change by season
with κ_season (see `race_pace_scale`): driver differences are scaled by κ in
the likelihood, and car-effect spread through the car prior, with centred car
effects `b_mach ~ N(0, σ_mach·κ_season)` (see `paceloss_loglik_cc`). Driver
effects are then comparable across eras.
"""
@model function paceloss_effects(g::GapData, season_counts::Vector{Int}, cov::LossCovariates,
                                 loss_duration::Bool, era::Symbol, pace_scale::Bool, age_basis,
                                 big_loss::Bool)
    σ_comp ~ truncated(Normal(0, 2); lower = 0)
    σ_mach ~ truncated(Normal(0, 2); lower = 0)
    σ ~ truncated(Normal(0, 1); lower = 0)
    z_comp ~ filldist(Normal(), length(g.competitors))
    if !pace_scale
        z_mach ~ filldist(Normal(), length(g.machines))
    end
    γ ~ filldist(Normal(0, 5), length(g.races))
    a_π ~ Normal(-1.5, 1)
    a_λ ~ Normal(log(2), 1)
    θ = (; a_π, a_λ)
    if loss_duration
        β_dur_π ~ Normal(0, 1)
        β_dur_λ ~ Normal(0, 1)
        θ = (; θ..., β_dur_π, β_dur_λ)
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
    if pace_scale
        τ_κ ~ truncated(Normal(0, 0.25); lower = 0)
        e_κ ~ filldist(TDist(3), cov.n_seasons - 1)
        κs = season_pace_scale((; τ_κ, e_κ))
        b_mach ~ arraydist(Normal.(0, σ_mach .* κs[cov.mach_season]))     # centred car effects
        fage = nothing
        if age_basis !== nothing
            τ_age ~ truncated(Normal(0, 0.25); lower = 0)
            e_age ~ filldist(Normal(), length(g.age_years) - 1)
            fage = age_curve(τ_age, e_age, age_basis)
        end
        big = nothing
        if big_loss       # two-component incident loss (#16)
            a_ρ ~ Normal(log(0.05 / 0.95), 1)                       # share of incidents that are big, ~5%
            δ_λ2 ~ truncated(Normal(log(10), 1); lower = 0)         # λ2 = λ_baseline·e^δ > λ_baseline
            big = [logistic(a_ρ), exp(a_λ + δ_λ2)]
        end
        @addlogprob! paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κs[cov.season], fage, big, g,
                                        season_counts)
    else
        @addlogprob! paceloss_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr, nothing, g, season_counts)
    end
end

function paceloss_effects(g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                          pace_scale::Bool = false, age::Bool = false, big_loss::Bool = false)
    era in ERA_TERMS || throw(ArgumentError("era must be one of $ERA_TERMS"))
    cov = LossCovariates(g)
    (era === :rw || pace_scale) && cov.n_seasons < 2 &&
        throw(ArgumentError("random walks over seasons need at least 2 seasons"))
    age && !pace_scale && throw(ArgumentError("the age curve is implemented for pace-scale models (pace_scale = true)"))
    big_loss && !pace_scale && throw(ArgumentError("the big-loss component is implemented for pace-scale models (pace_scale = true)"))
    age && length(g.age_years) < 3 && throw(ArgumentError("the age curve needs ages spanning at least 3 years"))
    return paceloss_effects(g, season_counts(g), cov, loss_duration, era, pace_scale,
                            age ? AgeCurveBasis(g) : nothing, big_loss)
end

"""
    fit_paceloss(g::GapData; loss_duration=false, era=:none, pace_scale=false, n_samples=1000, n_chains=1, ...)

Sample `paceloss_effects`, starting chains with race intercepts at each race's
mean timed gap and the loss parameters at their prior centres, era effects near
zero (jittered per chain). Sampler, progress and ensemble options as in `fit_gaps`.
"""
function fit_paceloss(g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                      pace_scale::Bool = false, age::Bool = false, big_loss::Bool = false,
                      n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                      sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                      progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100, kwargs...)
    adtype = hasproperty(sampler, :adtype) ? sampler.adtype : nothing
    adtype isa AutoReverseDiff && adtype.compile &&
        throw(ArgumentError("paceloss_effects needs uncompiled ReverseDiff (see gap_sampler)"))
    model = paceloss_effects(g; loss_duration, era, pace_scale, age, big_loss)
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
        loss_duration && (p = (; p..., β_dur_π = 0.1 * randn(rng), β_dur_λ = 0.1 * randn(rng)))
        if era === :decade || era === :regime
            k = era === :decade ? cov.n_decades : cov.n_regimes
            p = (; p..., τ_π_era = 0.1 + 0.1 * rand(rng), τ_λ_era = 0.1 + 0.1 * rand(rng),
                 z_π_era = 0.1 .* randn(rng, k), z_λ_era = 0.1 .* randn(rng, k))
        elseif era === :rw
            p = (; p..., τ_π_rw = 0.05 + 0.05 * rand(rng), τ_λ_rw = 0.05 + 0.05 * rand(rng),
                 e_π_rw = 0.1 .* randn(rng, cov.n_seasons - 1), e_λ_rw = 0.1 .* randn(rng, cov.n_seasons - 1))
        end
        if pace_scale      # centred car effects replace z_mach
            p = (; (k => v for (k, v) in pairs(p) if k !== :z_mach)...,
                 τ_κ = 0.05 + 0.05 * rand(rng), e_κ = 0.1 .* randn(rng, cov.n_seasons - 1),
                 b_mach = 0.1 .* randn(rng, length(g.machines)))
        end
        age && (p = (; p..., τ_age = 0.05 + 0.05 * rand(rng), e_age = 0.1 .* randn(rng, length(g.age_years) - 1)))
        big_loss && (p = (; p..., a_ρ = log(0.05 / 0.95) + 0.1 * randn(rng), δ_λ2 = log(10) + 0.1 * randn(rng)))
        return InitFromParams(p)
    end
    return run_nuts(model, [init() for _ in 1:n_chains]; n_samples, n_chains, ensemble, sampler,
                    rng, progress, progress_log, log_every, kwargs...)
end
