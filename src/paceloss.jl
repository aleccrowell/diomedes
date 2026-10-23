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

# A row's incident probability: the race's, shifted on the logit scale by the
# driver's incident offset (#101) if the model has one.
row_π(πr, r, πd, d) = πd === nothing ? πr[r] : logistic(logit(πr[r]) + πd[d])

# Scatter ∂ℓ/∂π_row into the race's π and the driver's offset (#101): with
# π_row = logistic(logit(π_r) + o_d), ∂π_row/∂o_d = π_row(1 - π_row) and
# ∂π_row/∂π_r = π_row(1 - π_row) / (π_r(1 - π_r)).
@inline function scatter_π!(dπ, dπd, p, πr, r, πd, d)
    if πd === nothing
        dπ[r] += p
    else
        q = row_π(πr, r, πd, d)
        w = p * q * (1 - q)
        dπ[r] += w / (πr[r] * (1 - πr[r]))
        dπd[d] += w
    end
end

# Per-row values and derivatives with the big-loss component (#16): as
# paceloss_row_grads, plus scalar derivatives for ρ and λ2. `πd`: per-driver
# logit offsets on π (#101), or nothing; their derivatives come back in dπd.
function pacebig_row_grads(st, σ, πr, λr, ρ, λ2, g; πd = nothing)
    dμt, dμc = zeros(length(st.μt)), zeros(length(st.μc))
    dπ, dλ = zeros(length(πr)), zeros(length(λr))
    dπd = πd === nothing ? nothing : zeros(length(πd))
    val, dσ, dρ, dλ2 = 0.0, 0.0, 0.0, 0.0
    for i in eachindex(st.μt)
        r, y = g.t_race[i], g.y[i]
        v, p = value_gradN((μ, s, q, l, h, l2) -> pacebig_logpdf(y - μ, s, q, l, h, l2),
                           st.μt[i], σ, row_π(πr, r, πd, g.t_comp[i]), λr[r], ρ, λ2)
        val += v; dμt[i] = p[1]; dσ += p[2]; dλ[r] += p[4]; dρ += p[5]; dλ2 += p[6]
        scatter_π!(dπ, dπd, p[3], πr, r, πd, g.t_comp[i])
    end
    for i in eachindex(st.μc)
        r, lo, hi = g.c_race[i], g.lo[i], g.hi[i]
        v, p = value_gradN((μ, s, q, l, h, l2) -> pacebig_loginterval(lo - μ, hi - μ, s, q, l, h, l2),
                           st.μc[i], σ, row_π(πr, r, πd, g.c_comp[i]), λr[r], ρ, λ2)
        val += v; dμc[i] = p[1]; dσ += p[2]; dλ[r] += p[4]; dρ += p[5]; dλ2 += p[6]
        scatter_π!(dπ, dπd, p[3], πr, r, πd, g.c_comp[i])
    end
    return val, dμt, dμc, dσ, dπ, dλ, dρ, dλ2, dπd
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
    paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv[, big], g::GapData, season_counts)

Log-likelihood of the pace-scale model (#15) with **centred** car effects.
Driver effects `a = σ_comp·(z_comp - mean)` are scaled by the per-race `κr`;
car effects are `b = b_mach - season mean` (their spread by season is set in
the prior, b_mach ~ N(0, σ_mach·κ_season)), so μ = γ + κ·a + b - (field mean).
Centred car effects avoid the ridge between κ_season and the spread of that
season's standardised effects that the non-centred form has; each car-season
has ~30+ results, so its effect is well determined by the data.

`hdrv` (`nothing`, or a value per row: timed rows, then lapped rows) is added
to each row's driver part, κ·(a[driver] + hdrv[row]). It carries the career
terms (#15): the population age curve and per-driver slopes (see
`driver_offsets`).
"""
function paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, g, season_counts)
    return paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, nothing, g, season_counts)
end

# `big` = nothing, or [ρ, λ2] for the two-component incident loss (#16).
paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, big, g, season_counts) =
    _loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, big, nothing, g, season_counts)

# With per-driver incident offsets (#101): `πd` shifts each row's π on the logit
# scale by its driver's offset (see `row_π`). Big-loss models only.
paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, ρ::Real, λ2::Real, πd::AbstractVector, g,
                   season_counts) =
    _loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, [ρ, λ2], πd, g, season_counts)

function _loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, big, πd, g, season_counts)
    a = σ_comp .* sum_to_zero(z_comp)
    b = sum_to_zero_by(b_mach, g.mach_season, season_counts)
    T = promote_type(eltype(a), eltype(b), eltype(γ), typeof(σ), eltype(πr), eltype(λr), eltype(κr),
                     hdrv === nothing ? Float64 : eltype(hdrv), big === nothing ? Float64 : eltype(big),
                     πd === nothing ? Float64 : eltype(πd))
    lpdf(x, r, d) = big === nothing ? paceloss_logpdf(x, σ, row_π(πr, r, πd, d), λr[r]) :
        pacebig_logpdf(x, σ, row_π(πr, r, πd, d), λr[r], big[1], big[2])
    lint(lo, hi, r, d) = big === nothing ? paceloss_loginterval(lo, hi, σ, row_π(πr, r, πd, d), λr[r]) :
        pacebig_loginterval(lo, hi, σ, row_π(πr, r, πd, d), λr[r], big[1], big[2])
    nt = length(g.t_race)
    h(i) = hdrv === nothing ? zero(T) : hdrv[i]
    ct = [κr[g.t_race[i]] * (a[g.t_comp[i]] + h(i)) + b[g.t_mach[i]] for i in eachindex(g.t_race)]
    cc = [κr[g.c_race[i]] * (a[g.c_comp[i]] + h(nt + i)) + b[g.c_mach[i]] for i in eachindex(g.c_race)]
    csum, n = zeros(T, length(γ)), zeros(Int, length(γ))
    for (c, r) in zip(ct, g.t_race); csum[r] += c; n[r] += 1; end
    for (c, r) in zip(cc, g.c_race); csum[r] += c; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    ll = zero(T)
    for i in eachindex(g.y)
        r = g.t_race[i]
        ll += lpdf(g.y[i] - (γ[r] + ct[i] - cbar[r]), r, g.t_comp[i])
    end
    for i in eachindex(g.lo)
        r = g.c_race[i]
        μ = γ[r] + cc[i] - cbar[r]
        ll += lint(g.lo[i] - μ, g.hi[i] - μ, r, g.c_comp[i])
    end
    return ll
end

# Reverse rule with per-driver incident offsets (#101): as the big-loss rule,
# plus the offsets' gradient.
function ChainRulesCore.rrule(::typeof(paceloss_loglik_cc), γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr,
                              hdrv, ρ::Real, λ2::Real, πd::AbstractVector, g::GapData, season_counts)
    zc = z_comp .- mean(z_comp)
    b = b_mach .- group_means(b_mach, g.mach_season, season_counts)[g.mach_season]
    st = row_means_ab(γ, σ_comp .* zc, b, g; κ = κr, κ_cars = false, hdrv)
    val, dμt, dμc, dσ, dπ, dλ, dρ, dλ2, dπd = pacebig_row_grads(st, σ, πr, λr, ρ, λ2, g; πd)
    d = row_pullback_ab(dμt, dμc, st, g; κ = κr, κ_cars = false, hdrv = hdrv !== nothing)
    dσ_comp = sum(d.da .* zc)
    dz_comp = σ_comp .* (d.da .- mean(d.da))
    db_mach = d.db .- group_means(d.db, g.mach_season, season_counts)[g.mach_season]
    dh = hdrv === nothing ? NoTangent() : d.dh
    pullback(Δ) = (NoTangent(), Δ .* d.dγ, Δ .* dz_comp, Δ * dσ_comp, Δ .* db_mach, Δ * dσ,
                   Δ .* dπ, Δ .* dλ, Δ .* d.dκ, dh isa NoTangent ? dh : Δ .* dh, Δ * dρ, Δ * dλ2,
                   Δ .* dπd, NoTangent(), NoTangent())
    return val, pullback
end
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     hdrv::ReverseDiff.TrackedArray, ρ::ReverseDiff.TrackedReal,
                                                     λ2::ReverseDiff.TrackedReal, πd::ReverseDiff.TrackedArray,
                                                     g::GapData, season_counts::Vector{Int})

function ChainRulesCore.rrule(::typeof(paceloss_loglik_cc), γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr,
                              hdrv, g::GapData, season_counts)
    val, pb = ChainRulesCore.rrule(paceloss_loglik_cc, γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv,
                                   nothing, g, season_counts)
    pullback(Δ) = (d = pb(Δ); (d[1:10]..., d[12], d[13]))     # drop the `big` slot
    return val, pullback
end

function ChainRulesCore.rrule(::typeof(paceloss_loglik_cc), γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr,
                              hdrv, big, g, season_counts)
    zc = z_comp .- mean(z_comp)
    b = b_mach .- group_means(b_mach, g.mach_season, season_counts)[g.mach_season]
    st = row_means_ab(γ, σ_comp .* zc, b, g; κ = κr, κ_cars = false, hdrv)
    if big === nothing
        val, dμt, dμc, dσ, dπ, dλ = paceloss_row_grads(st, σ, πr, λr, g)
        dbig = NoTangent()
    else
        val, dμt, dμc, dσ, dπ, dλ, dρ, dλ2, _ = pacebig_row_grads(st, σ, πr, λr, big[1], big[2], g)
        dbig = [dρ, dλ2]
    end
    d = row_pullback_ab(dμt, dμc, st, g; κ = κr, κ_cars = false, hdrv = hdrv !== nothing)
    dσ_comp = sum(d.da .* zc)
    dz_comp = σ_comp .* (d.da .- mean(d.da))
    db_mach = d.db .- group_means(d.db, g.mach_season, season_counts)[g.mach_season]
    dh = hdrv === nothing ? NoTangent() : d.dh
    pullback(Δ) = (NoTangent(), Δ .* d.dγ, Δ .* dz_comp, Δ * dσ_comp, Δ .* db_mach, Δ * dσ,
                   Δ .* dπ, Δ .* dλ, Δ .* d.dκ, dh isa NoTangent ? dh : Δ .* dh,
                   dbig isa NoTangent ? dbig : Δ .* dbig, NoTangent(), NoTangent())
    return val, pullback
end

# The big-loss component as two scalars, ρ and λ2 (#16). The model passes them
# this way because [ρ, λ2] built from tracked scalars is a Vector of TrackedReal,
# not a TrackedArray, so it matches no binding: ReverseDiff then traces the
# scalar loop over every row, ~30× slower (gradients still correct).
function paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, ρ::Real, λ2::Real, g, season_counts)
    return paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv, [ρ, λ2], g, season_counts)
end

function ChainRulesCore.rrule(::typeof(paceloss_loglik_cc), γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr,
                              hdrv, ρ::Real, λ2::Real, g::GapData, season_counts)
    val, pb = ChainRulesCore.rrule(paceloss_loglik_cc, γ, z_comp, σ_comp, b_mach, σ, πr, λr, κr, hdrv,
                                   [ρ, λ2], g, season_counts)
    pullback(Δ) = (d = pb(Δ); (d[1:10]..., d[11][1], d[11][2], d[12], d[13]))   # split the `big` slot
    return val, pullback
end

# With the big-loss component (#16), with and without career offsets.
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     hdrv::Nothing, ρ::ReverseDiff.TrackedReal, λ2::ReverseDiff.TrackedReal,
                                                     g::GapData, season_counts::Vector{Int})
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     hdrv::ReverseDiff.TrackedArray, ρ::ReverseDiff.TrackedReal,
                                                     λ2::ReverseDiff.TrackedReal, g::GapData, season_counts::Vector{Int})

# With and without career offsets (uncompiled tapes only; see gap_loglik).
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     hdrv::Nothing, g::GapData, season_counts::Vector{Int})
ReverseDiff.@grad_from_chainrules paceloss_loglik_cc(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                                     σ_comp::ReverseDiff.TrackedReal, b_mach::ReverseDiff.TrackedArray,
                                                     σ::ReverseDiff.TrackedReal, πr::ReverseDiff.TrackedArray,
                                                     λr::ReverseDiff.TrackedArray, κr::ReverseDiff.TrackedArray,
                                                     hdrv::ReverseDiff.TrackedArray, g::GapData,
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
    LossCovariates(g::GapData; circuits = nothing)
    LossCovariates(g::GapData, spec)

Race-level covariates for the loss component: standardised log race duration
(winner's time), and three era groupings of the race's season: decade,
regulation regime (`REGIME_STARTS`), and season index (for a random walk).
With `circuits` (a table like `circuit_table()`), also the race's circuit
covariates (`RaceCircuits`, #81). The second form takes a model spec (see
`model_spec`) and reads `circuit_table()` when the model has circuit terms.
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
    circuits::Union{Nothing,RaceCircuits}   # circuit covariates (#81), if given
    cars::Union{Nothing,CarWalks}           # chassis and engine walks (#95), if given
end
LossCovariates(g::GapData, spec) =
    LossCovariates(g; circuits = get(spec, :circ, false) || get(spec, :track, false) ? circuit_table() : nothing,
                   cars = get(spec, :walk, false) ? car_table() : nothing)
function LossCovariates(g::GapData; circuits = nothing, cars = nothing)
    ld = log.(g.race_minutes)
    decades = g.race_season .÷ 10
    d0 = minimum(decades)
    regime = [searchsortedlast(REGIME_STARTS, s) for s in g.race_season]
    s0 = minimum(g.race_season)
    data_seasons = sort(unique(g.race_season))         # GapData.mach_season indexes these
    return LossCovariates((ld .- mean(ld)) ./ std(ld), decades .- d0 .+ 1, maximum(decades) - d0 + 1,
                          regime, length(REGIME_STARTS),
                          g.race_season .- s0 .+ 1, maximum(g.race_season) - s0 + 1,
                          data_seasons[g.mach_season] .- s0 .+ 1,
                          circuits === nothing ? nothing : RaceCircuits(g, circuits),
                          cars === nothing ? nothing : CarWalks(g, cars))
end

const ERA_TERMS = (:none, :decade, :regime, :rw)

# Independent normals N(0, s[i]) as one multivariate distribution (#102). Its
# log density is a few vectorised operations, so through ReverseDiff it costs
# far less than `arraydist(Normal.(0, s))`, which tracks every element
# (centred car effects: ~13 ms and ~65k allocations less per gradient).
# A subtype of AbstractMvNormal so the samplers' transforms treat it as
# unconstrained (Bijectors maps that type to the identity).
struct IndepNormal{T<:AbstractVector{<:Real}} <: Turing.Distributions.AbstractMvNormal
    s::T
end
Base.length(d::IndepNormal) = length(d.s)
Turing.Distributions._logpdf(d::IndepNormal, x::AbstractVector{<:Real}) =
    -sum(log, d.s) - sum(abs2, x ./ d.s) / 2 - length(x) * log(2π) / 2
Turing.Distributions._rand!(rng::AbstractRNG, d::IndepNormal, x::AbstractVector{<:Real}) =
    (x .= randn(rng, length(x)) .* d.s; x)
Turing.Distributions.mean(d::IndepNormal) = zeros(eltype(d.s), length(d.s))
Turing.Distributions.var(d::IndepNormal) = abs2.(d.s)

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

With track terms (#81, `θ` has `β_track_π`, `β_track_λ`; needs `cov.circuits`),
the race's tight score and mixed and wet indicators (`RaceCircuits.track`) have
slopes on logit π and log λ: walls turn mistakes into retirements and safety
cars, and rain into spins.
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
    if haskey(θ, :β_track_π)
        cov.circuits === nothing && throw(ArgumentError("track terms need circuit covariates (LossCovariates(g; circuits))"))
        logitπ = logitπ .+ cov.circuits.track * θ.β_track_π
        logλ = logλ .+ cov.circuits.track * θ.β_track_λ
    end
    return logistic.(logitπ), exp.(logλ)
end

"Names of the loss parameters for a model variant, in sampling order."
function loss_param_names(loss_duration::Bool, era::Symbol, track::Bool = false)
    names = (:a_π, :a_λ)
    loss_duration && (names = (names..., :β_dur_π, :β_dur_λ))
    era in (:decade, :regime) && (names = (names..., :τ_π_era, :τ_λ_era, :z_π_era, :z_λ_era))
    era === :rw && (names = (names..., :τ_π_rw, :τ_λ_rw, :e_π_rw, :e_λ_rw))
    track && (names = (names..., :β_track_π, :β_track_λ))
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
AgeCurveBasis(g::GapData) = AgeCurveBasis(g.age_years, g.age_rows)
# from the age bins (whole years) and the rows per bin; also used by the lap model (#12)
function AgeCurveBasis(age_years::AbstractVector, age_rows::AbstractVector)
    n = length(age_years)
    walk = tril(ones(n, n - 1), -1)
    X = hcat(ones(n), Float64.(age_years))
    W = Diagonal(Float64.(age_rows))
    proj = I - X * ((X' * W * X) \ (X' * W))
    return AgeCurveBasis(walk, proj)
end

"Career curve per age bin from step scale `τ_age` and standardised steps `e_age`."
age_curve(τ_age, e_age, basis::AgeCurveBasis) = basis.proj * (basis.walk * (τ_age .* e_age))

"""
    CareerRows(g::GapData)

Per-row data for the career terms (#15), rows in GapData order (timed, then
lapped): the age bin (`bin`, with unknown ages pointing at bin 1 and masked out
by `known`), the driver (`comp`), and the row's age minus that driver's mean
age over their rows, in decades (`agec`; 0 when the age is unknown).

A per-driver slope s_d multiplies `agec`, so it is orthogonal to the driver's
constant effect. The slopes' mean is not identified (a common slope is the
linear age term of the age–period–cohort problem, see `AgeCurveBasis`), so the
slopes are centred to mean zero: they measure how a driver's career trajectory
differs from the average driver's, not whether drivers improve with age.
"""
struct CareerRows
    bin::Vector{Int}
    known::Vector{Float64}
    comp::Vector{Int}
    agec::Vector{Float64}
end
function CareerRows(g::GapData; train = nothing)     # driver mean ages over `train` rows only, if given
    bins = vcat(g.t_age, g.c_age)
    comp = vcat(g.t_comp, g.c_comp)
    known = Float64.(bins .> 0)
    yrs = [k == 0 ? 0.0 : Float64(g.age_years[k]) for k in bins]
    s, n = zeros(length(g.competitors)), zeros(length(g.competitors))
    for i in eachindex(bins)
        (train === nothing || train[i]) || continue
        s[comp[i]] += known[i] * yrs[i]; n[comp[i]] += known[i]
    end
    m = s ./ max.(n, 1)
    agec = [known[i] * (yrs[i] - m[comp[i]]) / 10 for i in eachindex(bins)]
    return CareerRows(max.(bins, 1), known, comp, agec)
end

"""
    driver_offsets(cr::CareerRows, fage, slope) -> per-row vector or `nothing`

Career terms added to each row's driver part (see `paceloss_loglik_cc`): the
age curve `fage` (per age bin, or `nothing`) and per-driver slopes `slope` (in
% per decade of age, or `nothing`).
"""
function driver_offsets(cr::CareerRows, fage, slope)
    h = fage === nothing ? nothing : fage[cr.bin] .* cr.known
    slope === nothing && return h
    hs = slope[cr.comp] .* cr.agec
    return h === nothing ? hs : h .+ hs
end

"Centred per-driver career slopes (% per decade) from scale `τ_slope` and standardised `u_slope`."
career_slopes(τ_slope, u_slope) = τ_slope .* (u_slope .- mean(u_slope))

"""
    center_by(u, group, counts)

`u` minus the mean of its group (`group[i]` in 1:length(counts), `counts` the
group sizes), as `sum_to_zero_by`, but with a hand-written adjoint (centring is
self-adjoint) so ReverseDiff keeps a tracked vector a TrackedArray.
"""
center_by(u::AbstractVector, group::Vector{Int}, counts::Vector{Int}) = sum_to_zero_by(u, group, counts)
function ChainRulesCore.rrule(::typeof(center_by), u::AbstractVector, group::Vector{Int}, counts::Vector{Int})
    pullback(ȳ) = (NoTangent(), sum_to_zero_by(collect(ChainRulesCore.unthunk(ȳ)), group, counts), NoTangent(), NoTangent())
    return center_by(u, group, counts), pullback
end
ReverseDiff.@grad_from_chainrules center_by(u::ReverseDiff.TrackedArray, group::Vector{Int}, counts::Vector{Int})

"""
    DevRows(g::GapData; train = nothing)

Per-row data for in-season car development (#14), rows in GapData order (timed,
then lapped): the machine level of each row (`mach`) and the race's position
in its season minus the mean position over that car's rows (`posc`). Position
runs from 0 at a season's first round in the data to 1 at its last.

A car's development trend t multiplies `posc`, so it is orthogonal to the
car-season effect. A trend shared by every car in a season is absorbed by the
race intercepts, so trends are centred within each season (see `dev_trends`):
they measure how a car's form over the season differed from the field's.
"""
struct DevRows
    mach::Vector{Int}
    posc::Vector{Float64}
end
function DevRows(g::GapData; train = nothing)     # car mean positions over `train` rows only, if given
    races = vcat(g.t_race, g.c_race)
    mach = vcat(g.t_mach, g.c_mach)
    # each race's round, then its position among its season's rounds
    round_of = Dict(r.race_key => r.round for r in eachrow(g.rows))
    rnd = [round_of[k] for k in g.races]
    pos = zeros(length(g.races))
    for s in unique(g.race_season)
        idx = findall(==(s), g.race_season)
        rounds = sort(unique(rnd[idx]))
        n = length(rounds)
        for r in idx
            pos[r] = n == 1 ? 0.0 : (searchsortedfirst(rounds, rnd[r]) - 1) / (n - 1)
        end
    end
    p = pos[races]
    s, n = zeros(length(g.machines)), zeros(length(g.machines))
    for i in eachindex(p)
        (train === nothing || train[i]) || continue
        s[mach[i]] += p[i]; n[mach[i]] += 1
    end
    return DevRows(mach, p .- (s ./ max.(n, 1))[mach])
end

"In-season development trends (% per season), centred within season, from scale `τ_dev` and standardised `u_dev`."
dev_trends(τ_dev, u_dev, g::GapData, season_counts) = τ_dev .* center_by(u_dev, g.mach_season, season_counts)

"""
    car_row_offsets(cx, mach, x, κs, season) -> Vector

Per-row car part Σ_k cx[(k - 1)·n_machines + mach[i]]·x[i, k] / κs[season[i]], for
terms that are not scaled by κ but enter the fused likelihood through the
κ-scaled driver offset (the development trends with `x = posc`, the circuit
slopes with `x = xc`). One pass with a hand-written rule: through ReverseDiff
the indexing, products and division allocated a tracked vector per step, and the
garbage collection that followed doubled the sampler's cost per step (#92).
"""
function car_row_offsets(cx, mach::Vector{Int}, x::AbstractMatrix, κs, season::Vector{Int})
    nm = length(cx) ÷ size(x, 2)
    T = promote_type(eltype(cx), eltype(κs), eltype(x))
    out = zeros(T, length(mach))
    for k in axes(x, 2), i in eachindex(mach)
        out[i] += cx[(k - 1) * nm + mach[i]] * x[i, k]
    end
    for i in eachindex(out); out[i] /= κs[season[i]]; end
    return out
end
function ChainRulesCore.rrule(::typeof(car_row_offsets), cx, mach::Vector{Int}, x::AbstractMatrix, κs,
                              season::Vector{Int})
    out = car_row_offsets(cx, mach, x, κs, season)
    function pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        nm = length(cx) ÷ size(x, 2)
        ∂cx, ∂κ = zeros(length(cx)), zeros(length(κs))
        for i in eachindex(mach)
            w = ȳ[i] / κs[season[i]]
            for k in axes(x, 2); ∂cx[(k - 1) * nm + mach[i]] += w * x[i, k]; end
            ∂κ[season[i]] -= w * out[i]
        end
        return NoTangent(), ∂cx, NoTangent(), NoTangent(), ∂κ, NoTangent()
    end
    return out, pullback
end
ReverseDiff.@grad_from_chainrules car_row_offsets(cx::ReverseDiff.TrackedArray, mach::Vector{Int}, x::AbstractMatrix,
                                                  κs::ReverseDiff.TrackedArray, season::Vector{Int})

"Names of the pace-scale parameters (#15), in sampling order (after the loss parameters)."
pace_param_names(pace_scale::Bool) = pace_scale ? (:τ_κ, :e_κ, :b_mach) : ()

"Names of the age-curve parameters (#15 stage 2), in sampling order (after the pace scale)."
age_param_names(age::Bool) = age ? (:τ_age, :e_age) : ()

"Names of the big-loss parameters (#16), in sampling order (after the age curve)."
big_param_names(big::Bool) = big ? (:a_ρ, :δ_λ2) : ()

"Names of the career-slope parameters (#15 stage 2b), in sampling order (after the big-loss component)."
slope_param_names(slopes::Bool) = slopes ? (:τ_slope, :u_slope) : ()

"Names of the in-season development parameters (#14), in sampling order (after the career slopes)."
dev_param_names(dev::Bool) = dev ? (:τ_dev, :u_dev) : ()

"Names of the circuit-slope parameters (#81), in sampling order (after the development trends)."
circ_param_names(circ::Bool) = circ ? (:τ_cx, :u_cx) : ()

"Names of the chassis and engine walk parameters (#95), if the model has them."
walk_param_names(walk::Bool, walk1::Bool = false) =
    !walk ? () : walk1 ? (:τ_ch, :σ0_ch, :z_ch, :τ_en, :σ0_en, :z_en, :δ_own, :δ_reg) :
    (:τ_ch, :σ0_ch, :z_ch, :τ_en, :σ0_en, :z_en, :δ_own, :δ_reg_ch, :δ_reg_en)

"Names of the works-term (#100) and per-driver incident (#101) parameters, in sampling order."
works_param_names(works::Bool) = works ? (:β_works,) : ()
dinc_param_names(dinc::Bool) = dinc ? (:τ_dinc, :u_dinc) : ()

"""
    race_pace_scale(θ, cov) -> Vector

Per-race pace scale κ (#15): `log κ_season` is a random walk over seasons with
Student-t(3) steps (step scale `τ_κ`, standardised steps `e_κ`), centred so the
geometric mean of κ over seasons is 1. Driver and car effects are then in
average-season units.
"""
race_pace_scale(θ, cov::LossCovariates) = season_pace_scale(θ)[cov.season]
"Per-season pace scale κ (see `race_pace_scale`)."
season_pace_scale(θ) = exp.(rw_path(θ.τ_κ .* θ.e_κ))

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
`AgeCurveBasis`). `slopes` (#15 stage 2b, needs `pace_scale`) adds a linear
career slope per driver, centred across drivers (see `CareerRows`). `dev` (#14,
needs `pace_scale`) adds a linear in-season development trend per car-season,
centred within season (see `DevRows`). `centred_drivers` (#28) samples driver
effects in % units, a ~ N(0, σ_comp), instead of a = σ_comp·z with z ~ N(0, 1):
it removes the funnel between σ_comp and the effects when most drivers are
data-rich. `driver_ν` (#28; `nothing` = Normal) gives driver effects a
Student-t(ν) prior instead (scale σ_comp), so an outlier driver can sit far from
the field without its results being explained as a string of incidents.
`race_hier` (#32) gives the race intercepts a learned common level and spread,
γ ~ N(μ_γ, τ_γ), instead of the fixed N(m, 4s). With fixed priors, shifting
every intercept together trades against the routine incident loss (σ vs π, λ)
at almost no likelihood cost, and the summed prior over ~1,000 races then
chose the split; a learned μ_γ removes that pull. Learning τ_γ as well
shrinks the intercepts (F1: sd 0.81 → 0.41, right tail from 7% to 2.3%), which
pushes race-level gaps into the loss component. `race_mean` (#32) learns only the
common level, γ ~ N(μ_γ, 4s), keeping the fixed spread; it is the default (#36),
and `race_mean = false` (with `race_hier = false`) restores the fixed N(m, 4s).

`track` (#81) gives the race's tight score (walls) and mixed and wet indicators
slopes on logit π and log λ (see `race_loss`). `circ` (#81, needs `pace_scale`)
adds car-season slopes on the race's speed, rough score (surface) and
temperature, standardised, centred per car-season over its races and within
season (see `CircuitRows`, `circuit_slopes`). Both read `circuits` (a table
like `circuit_table()`, the default).

`pace_scale` (#15) lets the spread of pace, in % terms, change by season
with κ_season (see `race_pace_scale`): driver differences are scaled by κ in
the likelihood, and car-effect spread through the car prior, with centred car
effects `b_mach ~ N(0, σ_mach·κ_season)` (see `paceloss_loglik_cc`). Driver
effects are then comparable across eras.
"""
@model function paceloss_effects(g::GapData, season_counts::Vector{Int}, cov::LossCovariates,
                                 loss_duration::Bool, era::Symbol, pace_scale::Bool, age_basis,
                                 big_loss::Bool, career_rows, dev_rows, centred_drivers::Bool, driver_ν,
                                 race_hier::Bool, race_mean::Bool, ps, circ_rows, track::Bool, ext)
    # priors on %-of-winner quantities are in units of the dataset's gap scale ps.s
    # (#30; see PriorScale); unitless ones (probabilities, log/logit steps) are not
    σ_comp ~ truncated(Normal(0, 2ps.s); lower = 0)
    σ_mach ~ truncated(Normal(0, 2ps.s); lower = 0)
    σ ~ truncated(Normal(0, ps.s); lower = 0)
    if centred_drivers      # (#28) driver effects in % units: a ~ N(0, σ_comp), as option B did for cars
        if driver_ν === nothing
            a_comp ~ filldist(Normal(0, σ_comp), length(g.competitors))
        else                # heavy-tailed drivers: a ~ σ_comp·t(ν)
            a_comp ~ filldist(σ_comp * TDist(driver_ν), length(g.competitors))
        end
        z_comp = a_comp ./ σ_comp       # the fused rules form σ_comp·(z - z̄) = a - ā
    else
        z_comp ~ filldist(driver_ν === nothing ? Normal() : TDist(driver_ν), length(g.competitors))
    end
    if !pace_scale
        z_mach ~ filldist(Normal(), length(g.machines))
    end
    if race_hier            # (#32) hierarchical race intercepts: the common level is learned
        μ_γ ~ Normal(ps.m, 4ps.s)
        τ_γ ~ truncated(Normal(0, 4ps.s); lower = 0)
        γ ~ filldist(Normal(μ_γ, τ_γ), length(g.races))
    elseif race_mean        # (#32) learned common level only, spread fixed as in the default prior
        μ_γ ~ Normal(ps.m, 4ps.s)
        γ ~ filldist(Normal(μ_γ, 4ps.s), length(g.races))
    else
        γ ~ filldist(Normal(ps.m, 4ps.s), length(g.races))
    end
    a_π ~ Normal(-1.5, 1)
    a_λ ~ Normal(log(2ps.s), 1)
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
    if track                # tight, mixed and wet on incidents (#81)
        nx = size(cov.circuits.track, 2)
        β_track_π ~ filldist(Normal(), nx)
        β_track_λ ~ filldist(Normal(), nx)
        θ = (; θ..., β_track_π, β_track_λ)
    end
    πr, λr = race_loss(θ, cov, loss_duration, era)
    if pace_scale
        τ_κ ~ truncated(Normal(0, 0.25); lower = 0)
        e_κ ~ filldist(TDist(3), cov.n_seasons - 1)
        κs = season_pace_scale((; τ_κ, e_κ))
        b_mach ~ IndepNormal(σ_mach .* κs[cov.mach_season])       # centred car effects (vectorised, #102)
        fage = nothing
        if age_basis !== nothing
            τ_age ~ truncated(Normal(0, 0.2ps.s); lower = 0)
            e_age ~ filldist(Normal(), length(g.age_years) - 1)
            fage = age_curve(τ_age, e_age, age_basis)
        end
        if big_loss       # two-component incident loss (#16)
            a_ρ ~ Normal(log(0.05 / 0.95), 1)                       # share of incidents that are big, ~5%
            δ_λ2 ~ truncated(Normal(log(10), 1); lower = 0)         # λ2 = λ_baseline·e^δ > λ_baseline
        end
        slope = nothing
        if career_rows !== nothing && career_rows.slopes     # per-driver career slopes (#15 stage 2b)
            τ_slope ~ truncated(Normal(0, 0.5ps.s); lower = 0)
            u_slope ~ filldist(Normal(), length(g.competitors))
            slope = career_slopes(τ_slope, u_slope)
        end
        hdrv = career_rows === nothing ? nothing : driver_offsets(career_rows.rows, fage, slope)
        row_season = cov.season[vcat(g.t_race, g.c_race)]
        if dev_rows !== nothing     # in-season car development (#14)
            τ_dev ~ truncated(Normal(0, ps.s); lower = 0)
            u_dev ~ filldist(Normal(), length(g.machines))
            # the car part is not scaled by κ, but hdrv is: pass the car offset as offset/κ
            hcar = car_row_offsets(dev_trends(τ_dev, u_dev, g, season_counts), dev_rows.mach,
                                   reshape(dev_rows.posc, :, 1), κs, row_season)
            hdrv = hdrv === nothing ? hcar : hdrv .+ hcar
        end
        if circ_rows !== nothing    # car-season circuit slopes (#81), a car part like dev: offset/κ
            nk = size(circ_rows.xc, 2)
            τ_cx ~ filldist(truncated(Normal(0, 0.5ps.s); lower = 0), nk)
            u_cx ~ filldist(Normal(), length(g.machines) * nk)
            hcx = car_row_offsets(circuit_slopes(τ_cx, u_cx, g, season_counts), circ_rows.mach, circ_rows.xc, κs,
                                  row_season)
            hdrv = hdrv === nothing ? hcx : hdrv .+ hcx
        end
        if cov.cars !== nothing     # chassis and engine walks over seasons (#95), average-season units like hdrv
            cw = cov.cars
            τ_ch ~ truncated(Normal(0, ps.s); lower = 0)          # step sd per year
            σ0_ch ~ truncated(Normal(0, 2ps.s); lower = 0)        # spread of a works' first season
            z_ch ~ filldist(Normal(), length(cw.chassis.prev))
            τ_en ~ truncated(Normal(0, ps.s); lower = 0)
            σ0_en ~ truncated(Normal(0, 2ps.s); lower = 0)
            z_en ~ filldist(Normal(), length(cw.engine.prev))
            δ_own ~ truncated(Normal(0, 1); lower = 0)            # larger steps: ownership, regulations
            if ext.walk1            # one for chassis and engine rules (#97)
                δ_reg ~ truncated(Normal(0, 1); lower = 0)
                θw = (; τ_ch, σ0_ch, z_ch, τ_en, σ0_en, z_en, δ_own, δ_reg)
            else                    # one per family, as in the #95 fit
                δ_reg_ch ~ truncated(Normal(0, 1); lower = 0)
                δ_reg_en ~ truncated(Normal(0, 1); lower = 0)
                θw = (; τ_ch, σ0_ch, z_ch, τ_en, σ0_en, z_en, δ_own, δ_reg_ch, δ_reg_en)
            end
            xc, xe = car_walk_states(θw, cw)
            hw = walk_offsets(xc, cw.row_ch, xe, cw.row_en)
            if ext.works            # the engine builder's own team (#100), scaled by κ like the walks
                β_works ~ Normal(0, ps.s)
                hw = hw .+ β_works .* cw.row_works
            end
            hdrv = hdrv === nothing ? hw : hdrv .+ hw
        end
        πd = nothing
        if ext.dinc                 # per-driver incident proneness on logit π (#101), centred across drivers
            τ_dinc ~ truncated(Normal(0, 0.5); lower = 0)
            u_dinc ~ filldist(Normal(), length(g.competitors))
            πd = τ_dinc .* (u_dinc .- mean(u_dinc))
        end
        # each call matches a ReverseDiff binding of the fused rule (see paceloss_loglik_cc)
        if big_loss && πd !== nothing
            @addlogprob! paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κs[cov.season], hdrv,
                                            logistic(a_ρ), exp(a_λ + δ_λ2), πd, g, season_counts)
        elseif big_loss
            @addlogprob! paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κs[cov.season], hdrv,
                                            logistic(a_ρ), exp(a_λ + δ_λ2), g, season_counts)
        else
            @addlogprob! paceloss_loglik_cc(γ, z_comp, σ_comp, b_mach, σ, πr, λr, κs[cov.season], hdrv, g,
                                            season_counts)
        end
    else
        @addlogprob! paceloss_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ, πr, λr, nothing, g, season_counts)
    end
end

function paceloss_effects(g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                          pace_scale::Bool = false, age::Bool = false, big_loss::Bool = false,
                          slopes::Bool = false, dev::Bool = false, centred_drivers::Bool = false,
                          driver_ν = nothing, race_hier::Bool = false, race_mean::Bool = !race_hier,
                          circ::Bool = false, track::Bool = false, walk::Bool = false, walk1::Bool = false,
                          works::Bool = false, dinc::Bool = false,
                          circuits = circ || track ? circuit_table() : nothing, cars = walk ? car_table() : nothing,
                          prior_scale = PriorScale(g))
    era in ERA_TERMS || throw(ArgumentError("era must be one of $ERA_TERMS"))
    race_hier && race_mean && throw(ArgumentError("race_hier and race_mean are alternatives"))
    (circ || track) && circuits === nothing && throw(ArgumentError("circ and track need a circuit table (circuits)"))
    walk && cars === nothing && throw(ArgumentError("walk needs the car tables (cars)"))
    cov = LossCovariates(g; circuits = circ || track ? circuits : nothing, cars = walk ? cars : nothing)
    (era === :rw || pace_scale) && cov.n_seasons < 2 &&
        throw(ArgumentError("random walks over seasons need at least 2 seasons"))
    age && !pace_scale && throw(ArgumentError("the age curve is implemented for pace-scale models (pace_scale = true)"))
    big_loss && !pace_scale && throw(ArgumentError("the big-loss component is implemented for pace-scale models (pace_scale = true)"))
    slopes && !pace_scale && throw(ArgumentError("career slopes are implemented for pace-scale models (pace_scale = true)"))
    dev && !pace_scale && throw(ArgumentError("in-season development is implemented for pace-scale models (pace_scale = true)"))
    circ && !pace_scale && throw(ArgumentError("circuit slopes are implemented for pace-scale models (pace_scale = true)"))
    walk && !pace_scale && throw(ArgumentError("chassis and engine walks are implemented for pace-scale models (pace_scale = true)"))
    (walk1 || works) && !walk && throw(ArgumentError("walk1 and works need the chassis and engine walks (walk = true)"))
    dinc && !big_loss && throw(ArgumentError("per-driver incident proneness is implemented for big-loss models (big_loss = true)"))
    age && length(g.age_years) < 3 && throw(ArgumentError("the age curve needs ages spanning at least 3 years"))
    career_rows = (age || slopes) ? (; rows = CareerRows(g), slopes) : nothing
    return paceloss_effects(g, season_counts(g), cov, loss_duration, era, pace_scale,
                            age ? AgeCurveBasis(g) : nothing, big_loss, career_rows, dev ? DevRows(g) : nothing,
                            centred_drivers, driver_ν === nothing ? nothing : Float64(driver_ν), race_hier, race_mean, prior_scale,
                            circ ? CircuitRows(g, cov.circuits) : nothing, track, (; walk1, works, dinc))
end

"""
    fit_paceloss(g::GapData; loss_duration=false, era=:none, pace_scale=false, n_samples=1000, n_chains=1, ...)

Sample `paceloss_effects`, starting chains with race intercepts at each race's
mean timed gap and the loss parameters at their prior centres, era effects near
zero (jittered per chain). Sampler, progress and ensemble options as in `fit_gaps`.
With `checkpoint` (a path prefix), one chain is sampled in saved, resumable
blocks (see `run_nuts_checkpointed`; `seed` sets its RNGs).
"""
function fit_paceloss(g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                      pace_scale::Bool = false, age::Bool = false, big_loss::Bool = false,
                      slopes::Bool = false, dev::Bool = false, centred_drivers::Bool = false, driver_ν = nothing,
                      race_hier::Bool = false, race_mean::Bool = !race_hier, circ::Bool = false, track::Bool = false,
                      walk::Bool = false, walk1::Bool = false, works::Bool = false, dinc::Bool = false,
                      circuits = circ || track ? circuit_table() : nothing,
                      cars = walk ? car_table() : nothing, n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                      sampler = gap_sampler(), rng = Random.default_rng(), progress::Bool = true,
                      progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100,
                      checkpoint::Union{Nothing,AbstractString} = nothing, seed::Int = 1, kwargs...)
    adtype = hasproperty(sampler, :adtype) ? sampler.adtype : nothing
    adtype isa AutoReverseDiff && adtype.compile &&
        throw(ArgumentError("paceloss_effects needs uncompiled ReverseDiff (see gap_sampler)"))
    checkpoint === nothing || n_chains == 1 || throw(ArgumentError("checkpointed fits run one chain per call"))
    model = paceloss_effects(g; loss_duration, era, pace_scale, age, big_loss, slopes, dev, centred_drivers, driver_ν,
                             race_hier, race_mean, circ, track, walk, walk1, works, dinc, circuits, cars)
    cov = model.args.cov
    race_gap = [let ys = g.y[g.t_race .== r]; isempty(ys) ? 0.0 : mean(ys) end
                 for r in eachindex(g.races)]
    ps = model.args.ps
    scale() = (0.3 + 0.4 * rand(rng)) * ps.s          # start scales at the dataset's gap scale (#30)
    function init()
        p = (; σ_comp = scale(), σ_mach = scale(), σ = scale(),
             z_comp = 0.1 .* randn(rng, length(g.competitors)),
             z_mach = 0.1 .* randn(rng, length(g.machines)),
             γ = race_gap .+ 0.1 .* randn(rng, length(race_gap)),
             a_π = -1.5 + 0.1 * randn(rng), a_λ = log(2ps.s) + 0.1 * randn(rng))
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
        slopes && (p = (; p..., τ_slope = 0.05 + 0.05 * rand(rng), u_slope = 0.1 .* randn(rng, length(g.competitors))))
        dev && (p = (; p..., τ_dev = 0.05 + 0.05 * rand(rng), u_dev = 0.1 .* randn(rng, length(g.machines))))
        if circ
            nk = size(cov.circuits.x, 2)
            p = (; p..., τ_cx = 0.05 .+ 0.05 .* rand(rng, nk), u_cx = 0.1 .* randn(rng, length(g.machines) * nk))
        end
        if track
            nx = size(cov.circuits.track, 2)
            p = (; p..., β_track_π = 0.1 .* randn(rng, nx), β_track_λ = 0.1 .* randn(rng, nx))
        end
        if walk
            cw = cov.cars
            p = (; p..., τ_ch = 0.05 + 0.05 * rand(rng), σ0_ch = scale(), z_ch = 0.1 .* randn(rng, length(cw.chassis.prev)),
                 τ_en = 0.05 + 0.05 * rand(rng), σ0_en = scale(), z_en = 0.1 .* randn(rng, length(cw.engine.prev)),
                 δ_own = 0.1 + 0.1 * rand(rng),
                 (walk1 ? (; δ_reg = 0.1 + 0.1 * rand(rng)) :
                  (; δ_reg_ch = 0.1 + 0.1 * rand(rng), δ_reg_en = 0.1 + 0.1 * rand(rng)))...)
            works && (p = (; p..., β_works = 0.1 * randn(rng)))
        end
        dinc && (p = (; p..., τ_dinc = 0.05 + 0.05 * rand(rng), u_dinc = 0.1 .* randn(rng, length(g.competitors))))
        race_hier && (p = (; p..., μ_γ = mean(race_gap) + 0.1 * randn(rng), τ_γ = std(race_gap) * (0.9 + 0.2 * rand(rng))))
        race_mean && (p = (; p..., μ_γ = mean(race_gap) + 0.1 * randn(rng)))
        centred_drivers && (p = (; (k => v for (k, v) in pairs(p) if k !== :z_comp)...,
                                 a_comp = 0.05 .* randn(rng, length(g.competitors))))
        return InitFromParams(p)
    end
    # one chain saved in blocks, resumable (see run_nuts_checkpointed)
    checkpoint === nothing ||
        return run_nuts_checkpointed(model, init(), checkpoint; n_samples, seed, progress_log, log_every, kwargs...)
    return run_nuts(model, [init() for _ in 1:n_chains]; n_samples, n_chains, ensemble, sampler,
                    rng, progress, progress_log, log_every, kwargs...)
end
