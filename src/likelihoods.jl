# Log-likelihood terms with hand-written reverse-mode rules.
#
# Letting ReverseDiff differentiate these elementwise (through ForwardDiff duals
# in broadcast) was the bottleneck of `gap_effects`: 11.3 ms per gradient on the
# full F1 data, mostly the censored term. The closed forms below are
# differentiated once per call in a single pass. The primal functions stay
# generic, so ForwardDiff differentiates them directly and the tests can compare
# the two independent derivative paths.

"""
    gaussian_loglik(μ, σ, y)

Σᵢ log N(yᵢ | μᵢ, σ).
"""
function gaussian_loglik(μ::AbstractVector, σ::Real, y::AbstractVector)
    s = zero(promote_type(eltype(μ), typeof(σ)))
    for i in eachindex(y, μ)
        s += normlogpdf((y[i] - μ[i]) / σ)
    end
    return s - length(y) * log(σ)
end

function ChainRulesCore.rrule(::typeof(gaussian_loglik), μ::AbstractVector, σ::Real, y::AbstractVector)
    n = length(y)
    dμ = similar(μ, Float64)
    s, ss = 0.0, 0.0
    for i in eachindex(y, μ)
        r = (y[i] - μ[i]) / σ
        s += normlogpdf(r)
        ss += r^2
        dμ[i] = r / σ
    end
    val = s - n * log(σ)
    dσ = (ss - n) / σ
    pullback(Δ) = (NoTangent(), Δ .* dμ, Δ * dσ, NoTangent())
    return val, pullback
end

"""
    interval_loglik(μ, σ, lo, hi)

Σᵢ log P(loᵢ < Yᵢ < hiᵢ) for Yᵢ ~ N(μᵢ, σ), with finite `lo < hi`.
"""
function interval_loglik(μ::AbstractVector, σ::Real, lo::AbstractVector, hi::AbstractVector)
    s = zero(promote_type(eltype(μ), typeof(σ)))
    for i in eachindex(μ, lo, hi)
        s += logdiffΦ((lo[i] - μ[i]) / σ, (hi[i] - μ[i]) / σ)
    end
    return s
end

function ChainRulesCore.rrule(::typeof(interval_loglik), μ::AbstractVector, σ::Real,
                              lo::AbstractVector, hi::AbstractVector)
    dμ = similar(μ, Float64)
    val, dσ = 0.0, 0.0
    for i in eachindex(μ, lo, hi)
        a = (lo[i] - μ[i]) / σ
        b = (hi[i] - μ[i]) / σ
        ℓ = logdiffΦ(a, b)
        val += ℓ
        # φ(x)/D computed in log space: D = Φ(b) - Φ(a) = exp(ℓ) can underflow
        pa = exp(normlogpdf(a) - ℓ)
        pb = exp(normlogpdf(b) - ℓ)
        dμ[i] = (pa - pb) / σ
        dσ += (a * pa - b * pb) / σ
    end
    pullback(Δ) = (NoTangent(), Δ .* dμ, Δ * dσ, NoTangent(), NoTangent())
    return val, pullback
end

"""
    gap_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ_y, g::GapData, season_counts)

The whole `gap_effects` log-likelihood as one function of the parameters:
competitor effects `σ_comp·(z_comp - mean)`, machine effects
`σ_mach·(z_mach - season mean)`, per-row means `γ[race] + a[comp] + b[mach]`,
the Gaussian term for timed rows and the interval term for lapped rows.

Its reverse rule does the whole backward pass by hand (scatter row gradients
into races, competitors and machines, then undo the centring). The AD tape
then only holds this single call.
"""
function gap_loglik(γ, z_comp, σ_comp, z_mach, σ_mach, σ_y, g, season_counts)
    a = σ_comp .* sum_to_zero(z_comp)
    b = σ_mach .* sum_to_zero_by(z_mach, g.mach_season, season_counts)
    μt = γ[g.t_race] .+ a[g.t_comp] .+ b[g.t_mach]
    ll = gaussian_loglik(μt, σ_y, g.y)
    if !isempty(g.lo)
        μc = γ[g.c_race] .+ a[g.c_comp] .+ b[g.c_mach]
        ll += interval_loglik(μc, σ_y, g.lo, g.hi)
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
                              season_counts)
    zc = z_comp .- mean(z_comp)
    zm = z_mach .- group_means(z_mach, g.mach_season, season_counts)[g.mach_season]
    a = σ_comp .* zc
    b = σ_mach .* zm
    dγ, da, db = zeros(length(γ)), zeros(length(a)), zeros(length(b))
    val, dσy = 0.0, 0.0
    n = length(g.y)
    for i in 1:n
        r = (g.y[i] - γ[g.t_race[i]] - a[g.t_comp[i]] - b[g.t_mach[i]]) / σ_y
        val += normlogpdf(r)
        dμ = r / σ_y
        dγ[g.t_race[i]] += dμ; da[g.t_comp[i]] += dμ; db[g.t_mach[i]] += dμ
        dσy += (r^2 - 1) / σ_y
    end
    val -= n * log(σ_y)
    for i in eachindex(g.lo)
        μ = γ[g.c_race[i]] + a[g.c_comp[i]] + b[g.c_mach[i]]
        lo, hi = (g.lo[i] - μ) / σ_y, (g.hi[i] - μ) / σ_y
        ℓ = logdiffΦ(lo, hi)
        val += ℓ
        pa, pb = exp(normlogpdf(lo) - ℓ), exp(normlogpdf(hi) - ℓ)
        dμ = (pa - pb) / σ_y
        dγ[g.c_race[i]] += dμ; da[g.c_comp[i]] += dμ; db[g.c_mach[i]] += dμ
        dσy += (lo * pa - hi * pb) / σ_y
    end
    # through a = σ_comp·(z - mean z) and b = σ_mach·(z - group mean z); centring is self-adjoint
    dσ_comp = sum(da .* zc)
    dz_comp = σ_comp .* (da .- mean(da))
    dσ_mach = sum(db .* zm)
    dz_mach = σ_mach .* (db .- group_means(db, g.mach_season, season_counts)[g.mach_season])
    pullback(Δ) = (NoTangent(), Δ .* dγ, Δ .* dz_comp, Δ * dσ_comp, Δ .* dz_mach, Δ * dσ_mach,
                   Δ * dσy, NoTangent(), NoTangent())
    return val, pullback
end

# Uncompiled tapes only: a compiled tape replays the pullback recorded at compile
# time instead of re-running the rule (verified: stale gradients at new points).
ReverseDiff.@grad_from_chainrules gap_loglik(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                             σ_comp::ReverseDiff.TrackedReal, z_mach::ReverseDiff.TrackedArray,
                                             σ_mach::ReverseDiff.TrackedReal, σ_y::ReverseDiff.TrackedReal,
                                             g::GapData, season_counts::Vector{Int})
