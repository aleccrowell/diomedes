# Likelihood for `gap_effects`, with a hand-written reverse-mode rule.
#
# Letting ReverseDiff differentiate the likelihood elementwise was the
# bottleneck of `gap_effects` (11.3 ms per gradient on the full F1 data, mostly
# the censored term). `gap_loglik` computes the whole likelihood and its
# gradient in a few passes over the rows. The primal stays generic, so tests can
# check the rule against ForwardDiff (Gaussian) or finite differences
# (Student-t, whose CDF ForwardDiff cannot differentiate).

"""
    gap_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ_y, g::GapData, season_counts, noise)

The whole `gap_effects` log-likelihood as one function of the parameters:
- competitor effects `a = σ_comp·(z_comp - mean)`
- machine effects `b = σ_mach·(z_mach - season mean)`
- per-row mean `μᵢ = γ[race] + cᵢ - c̄[race]`, where `cᵢ = a[comp] + b[mach]` and
  `c̄[race]` is the mean of `c` over that race's rows (timed and lapped)
- timed rows: density of `(y - μ)/σ_y` under `noise`; lapped rows: probability
  of the interval `[lo, hi)`.

Subtracting the field mean makes `γ[race]` the expected gap of the race's
average entrant, which is easier to interpret and to give a prior than a gap
anchored to an arbitrary zero. (It did not measurably change sampling cost:
probes with 100 adaptation iterations took ~260 steps per iteration either
way, and with 300 adaptation iterations the timed-only model settles at ~31.)

Its reverse rule does the whole backward pass by hand (scatter row gradients
into races, competitors and machines, then undo the centrings), so the AD tape
only holds this single call.
"""
function gap_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ_y, g, season_counts,
                    noise::Noise = NormalNoise())
    a = σ_comp .* sum_to_zero(z_comp)
    b = σ_mach .* sum_to_zero_by(z_mach, g.mach_season, season_counts)
    ct = a[g.t_comp] .+ b[g.t_mach]
    cc = a[g.c_comp] .+ b[g.c_mach]
    T = promote_type(eltype(ct), eltype(γ), typeof(σ_y))
    csum, n = zeros(T, length(γ)), zeros(Int, length(γ))
    for (c, r) in zip(ct, g.t_race); csum[r] += c; n[r] += 1; end
    for (c, r) in zip(cc, g.c_race); csum[r] += c; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    ll = zero(T)
    for i in eachindex(g.y)
        r = g.t_race[i]
        ll += logpdf_std(noise, (g.y[i] - (γ[r] + ct[i] - cbar[r])) / σ_y)
    end
    ll -= length(g.y) * log(σ_y)
    for i in eachindex(g.lo)
        r = g.c_race[i]
        μ = γ[r] + cc[i] - cbar[r]
        ll += logdiffcdf_std(noise, (g.lo[i] - μ) / σ_y, (g.hi[i] - μ) / σ_y)
    end
    return ll
end

# Mean of `z` within each group (allocation-light, for the reverse pass).
function group_means(z, group, counts)
    m = zeros(length(counts))
    for i in eachindex(z)
        m[group[i]] += z[i]
    end
    return m ./ counts
end

function ChainRulesCore.rrule(::typeof(gap_loglik), γ, z_comp, σ_comp, z_mach, σ_mach, σ_y, g,
                              season_counts, noise::Noise = NormalNoise())
    zc = z_comp .- mean(z_comp)
    zm = z_mach .- group_means(z_mach, g.mach_season, season_counts)[g.mach_season]
    a = σ_comp .* zc
    b = σ_mach .* zm
    nt, nc, nr = length(g.y), length(g.lo), length(γ)
    # field means of c = a[comp] + b[mach] per race
    csum, n = zeros(nr), zeros(Int, nr)
    for i in 1:nt
        r = g.t_race[i]; csum[r] += a[g.t_comp[i]] + b[g.t_mach[i]]; n[r] += 1
    end
    for i in 1:nc
        r = g.c_race[i]; csum[r] += a[g.c_comp[i]] + b[g.c_mach[i]]; n[r] += 1
    end
    cbar = csum ./ max.(n, 1)
    # likelihood and dℓ/dμ per row; dγ is the race sum of dμ
    dμt, dμc = zeros(nt), zeros(nc)
    dγ = zeros(nr)
    val, dσy = 0.0, 0.0
    for i in 1:nt
        r = g.t_race[i]
        ρ = (g.y[i] - γ[r] - a[g.t_comp[i]] - b[g.t_mach[i]] + cbar[r]) / σ_y
        val += logpdf_std(noise, ρ)
        s = score_std(noise, ρ)          # dℓ/dρ; ρ falls as μ rises: dℓ/dμ = -s/σ_y
        dμt[i] = -s / σ_y
        dγ[r] += dμt[i]
        dσy += (-s * ρ - 1) / σ_y
    end
    val -= nt * log(σ_y)
    for i in 1:nc
        r = g.c_race[i]
        μ = γ[r] + a[g.c_comp[i]] + b[g.c_mach[i]] - cbar[r]
        lo, hi = (g.lo[i] - μ) / σ_y, (g.hi[i] - μ) / σ_y
        ℓ = logdiffcdf_std(noise, lo, hi)
        val += ℓ
        # f(x)/D in log space: D = F(hi) - F(lo) = exp(ℓ) can underflow
        pa, pb = exp(logpdf_std(noise, lo) - ℓ), exp(logpdf_std(noise, hi) - ℓ)
        dμc[i] = (pa - pb) / σ_y
        dγ[r] += dμc[i]
        dσy += (lo * pa - hi * pb) / σ_y
    end
    # through cᵢ - c̄[race] (race centring is self-adjoint): dc = dμ - dγ[race]/n[race]
    da, db = zeros(length(a)), zeros(length(b))
    for i in 1:nt
        r = g.t_race[i]; dc = dμt[i] - dγ[r] / n[r]
        da[g.t_comp[i]] += dc; db[g.t_mach[i]] += dc
    end
    for i in 1:nc
        r = g.c_race[i]; dc = dμc[i] - dγ[r] / n[r]
        da[g.c_comp[i]] += dc; db[g.c_mach[i]] += dc
    end
    # through a = σ_comp·(z - mean z) and b = σ_mach·(z - group mean z); centring is self-adjoint
    dσ_comp = sum(da .* zc)
    dz_comp = σ_comp .* (da .- mean(da))
    dσ_mach = sum(db .* zm)
    dz_mach = σ_mach .* (db .- group_means(db, g.mach_season, season_counts)[g.mach_season])
    pullback(Δ) = (NoTangent(), Δ .* dγ, Δ .* dz_comp, Δ * dσ_comp, Δ .* dz_mach, Δ * dσ_mach,
                   Δ * dσy, NoTangent(), NoTangent(), NoTangent())
    return val, pullback
end

# Uncompiled tapes only: a compiled tape replays the pullback recorded at compile
# time instead of re-running the rule (verified: stale gradients at new points).
ReverseDiff.@grad_from_chainrules gap_loglik(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                             σ_comp::ReverseDiff.TrackedReal, z_mach::ReverseDiff.TrackedArray,
                                             σ_mach::ReverseDiff.TrackedReal, σ_y::ReverseDiff.TrackedReal,
                                             g::GapData, season_counts::Vector{Int}, noise::Noise)
