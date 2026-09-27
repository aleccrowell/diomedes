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

# Shared forward/backward passes for the fused gap-model rules (Float64 only).
#
# Forward: centred effects a, b and per-row means μ = γ[race] + c - c̄[race]
# with c = a[comp] + b[mach]. Backward: given dℓ/dμ per row, return gradients
# for γ, z_comp, σ_comp, z_mach, σ_mach. Race centring and both effect
# centrings are self-adjoint, so each backward step mirrors its forward step.
function gap_row_means(γ, z_comp, σ_comp, z_mach, σ_mach, g, season_counts; κ = nothing)
    zc = z_comp .- mean(z_comp)
    zm = z_mach .- group_means(z_mach, g.mach_season, season_counts)[g.mach_season]
    a, b = σ_comp .* zc, σ_mach .* zm
    nr = length(γ)
    c0t = [a[g.t_comp[i]] + b[g.t_mach[i]] for i in eachindex(g.t_race)]
    c0c = [a[g.c_comp[i]] + b[g.c_mach[i]] for i in eachindex(g.c_race)]
    csum, n = zeros(nr), zeros(Int, nr)
    for i in eachindex(c0t); r = g.t_race[i]; csum[r] += c0t[i]; n[r] += 1; end
    for i in eachindex(c0c); r = g.c_race[i]; csum[r] += c0c[i]; n[r] += 1; end
    cbar = csum ./ max.(n, 1)
    k(r) = κ === nothing ? 1.0 : κ[r]
    μt = [γ[g.t_race[i]] + k(g.t_race[i]) * (c0t[i] - cbar[g.t_race[i]]) for i in eachindex(c0t)]
    μc = [γ[g.c_race[i]] + k(g.c_race[i]) * (c0c[i] - cbar[g.c_race[i]]) for i in eachindex(c0c)]
    return (; zc, zm, n, μt, μc, c0t, c0c, cbar)
end

# With a per-race pace scale κ (μ = γ + κ·(c - c̄)), also returns dκ;
# dκ[r] = Σᵢ dμᵢ (cᵢ - c̄[r]) and the gradient to c is scaled by κ[r].
function gap_effects_pullback(dμt, dμc, st, σ_comp, σ_mach, g, season_counts; κ = nothing)
    nr = length(st.n)
    dγ = zeros(nr)
    for i in eachindex(dμt); dγ[g.t_race[i]] += dμt[i]; end
    for i in eachindex(dμc); dγ[g.c_race[i]] += dμc[i]; end
    dκ = κ === nothing ? nothing : zeros(nr)
    if κ !== nothing
        for i in eachindex(dμt); r = g.t_race[i]; dκ[r] += dμt[i] * (st.c0t[i] - st.cbar[r]); end
        for i in eachindex(dμc); r = g.c_race[i]; dκ[r] += dμc[i] * (st.c0c[i] - st.cbar[r]); end
    end
    k(r) = κ === nothing ? 1.0 : κ[r]
    # through κ·(cᵢ - c̄[race]): dc = κ·(dμ - dγ[race]/n[race])
    da, db = zeros(length(st.zc)), zeros(length(st.zm))
    for i in eachindex(dμt)
        r = g.t_race[i]; dc = k(r) * (dμt[i] - dγ[r] / st.n[r])
        da[g.t_comp[i]] += dc; db[g.t_mach[i]] += dc
    end
    for i in eachindex(dμc)
        r = g.c_race[i]; dc = k(r) * (dμc[i] - dγ[r] / st.n[r])
        da[g.c_comp[i]] += dc; db[g.c_mach[i]] += dc
    end
    # through a = σ_comp·(z - mean z) and b = σ_mach·(z - season mean z)
    dσ_comp = sum(da .* st.zc)
    dz_comp = σ_comp .* (da .- mean(da))
    dσ_mach = sum(db .* st.zm)
    dz_mach = σ_mach .* (db .- group_means(db, g.mach_season, season_counts)[g.mach_season])
    return (; dγ, dz_comp, dσ_comp, dz_mach, dσ_mach, dκ)
end

function ChainRulesCore.rrule(::typeof(gap_loglik), γ, z_comp, σ_comp, z_mach, σ_mach, σ_y, g,
                              season_counts, noise::Noise = NormalNoise())
    st = gap_row_means(γ, z_comp, σ_comp, z_mach, σ_mach, g, season_counts)
    dμt, dμc = zeros(length(st.μt)), zeros(length(st.μc))
    val, dσy = 0.0, 0.0
    for i in eachindex(st.μt)
        ρ = (g.y[i] - st.μt[i]) / σ_y
        val += logpdf_std(noise, ρ)
        s = score_std(noise, ρ)          # dℓ/dρ; ρ falls as μ rises: dℓ/dμ = -s/σ_y
        dμt[i] = -s / σ_y
        dσy += (-s * ρ - 1) / σ_y
    end
    val -= length(st.μt) * log(σ_y)
    for i in eachindex(st.μc)
        lo, hi = (g.lo[i] - st.μc[i]) / σ_y, (g.hi[i] - st.μc[i]) / σ_y
        ℓ = logdiffcdf_std(noise, lo, hi)
        val += ℓ
        # f(x)/D in log space: D = F(hi) - F(lo) = exp(ℓ) can underflow
        pa, pb = exp(logpdf_std(noise, lo) - ℓ), exp(logpdf_std(noise, hi) - ℓ)
        dμc[i] = (pa - pb) / σ_y
        dσy += (lo * pa - hi * pb) / σ_y
    end
    d = gap_effects_pullback(dμt, dμc, st, σ_comp, σ_mach, g, season_counts)
    pullback(Δ) = (NoTangent(), Δ .* d.dγ, Δ .* d.dz_comp, Δ * d.dσ_comp, Δ .* d.dz_mach,
                   Δ * d.dσ_mach, Δ * dσy, NoTangent(), NoTangent(), NoTangent())
    return val, pullback
end

# Uncompiled tapes only: a compiled tape replays the pullback recorded at compile
# time instead of re-running the rule (verified: stale gradients at new points).
ReverseDiff.@grad_from_chainrules gap_loglik(γ::ReverseDiff.TrackedArray, z_comp::ReverseDiff.TrackedArray,
                                             σ_comp::ReverseDiff.TrackedReal, z_mach::ReverseDiff.TrackedArray,
                                             σ_mach::ReverseDiff.TrackedReal, σ_y::ReverseDiff.TrackedReal,
                                             g::GapData, season_counts::Vector{Int}, noise::Noise)
