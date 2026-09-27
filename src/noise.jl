# Noise families for `gap_effects` (see `gap_loglik`).

"""
Noise family for `gap_effects`, standardised to location 0 and scale 1. Each
family provides `logpdf_std(n, ρ)`, `score_std(n, ρ)` (d logpdf / dρ) and
`logdiffcdf_std(n, a, b) = log(F(b) - F(a))` for a < b.
"""
abstract type Noise end

"Gaussian noise."
struct NormalNoise <: Noise end

"""
    StudentTNoise(ν)

Student-t noise with fixed degrees of freedom `ν`. Heavy tails let occasional
incidents (a spin, a slow stop, a car a few laps down) sit in the tails instead
of inflating the scale for every row. `ν` is fixed rather than learned: the
gradient of the t CDF with respect to ν has no simple closed form.

`ν = 4` uses closed forms (see below), about 5x cheaper than the general
incomplete-beta CDF; other `ν` use StatsFuns.
"""
struct StudentTNoise <: Noise
    ν::Float64
    logc::Float64     # log normalising constant of the standard t density
end
StudentTNoise(ν::Real) = StudentTNoise(Float64(ν), loggamma((ν + 1) / 2) - loggamma(ν / 2) - log(ν * π) / 2)

logpdf_std(::NormalNoise, ρ) = normlogpdf(ρ)
score_std(::NormalNoise, ρ) = -ρ
logdiffcdf_std(::NormalNoise, a, b) = logdiffΦ(a, b)

logpdf_std(n::StudentTNoise, ρ) = n.logc - (n.ν + 1) / 2 * log1p(ρ^2 / n.ν)
score_std(n::StudentTNoise, ρ) = -(n.ν + 1) * ρ / (n.ν + ρ^2)

# ν = 4 closed forms, with s = x/√(4 + x²) ∈ (-1, 1):
#   F(x) = 1/2 + 3s/4 - s³/4,   1 - F(x) = (1 - s)²(2 + s)/4,
# where 1 - s = 4 / (√(4 + x²)(√(4 + x²) + x)) avoids cancellation for large x.
function t4_logccdf(x)          # x ≥ 0
    r = sqrt(4 + x^2)
    return 2 * log(4 / (r * (r + x))) + log(2 + x / r) - log(4)
end
t4_cdf(x) = (s = x / sqrt(4 + x^2); 1 / 2 + 3s / 4 - s^3 / 4)

function logdiffcdf_std(n::StudentTNoise, a, b)
    if a >= 0                       # upper tail
        la, lb = n.ν == 4 ? (t4_logccdf(a), t4_logccdf(b)) :
                            (tdistlogccdf(n.ν, a), tdistlogccdf(n.ν, b))
        return la + log1mexp(lb - la)
    elseif b <= 0                   # lower tail, by symmetry
        return logdiffcdf_std(n, -b, -a)
    else                            # straddles 0
        return n.ν == 4 ? log(t4_cdf(b) - t4_cdf(a)) : log(tdistcdf(n.ν, b) - tdistcdf(n.ν, a))
    end
end
