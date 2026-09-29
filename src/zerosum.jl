# Sum-to-zero parametrisation of effect vectors (#25).
#
# Driver effects, car effects (within each season) and career slopes enter the
# likelihood only through their centred values, so their mean is identified
# only by the prior. Sampling all n coordinates leaves one prior-only direction
# per vector (or per season), along which chains drift slowly and which makes
# R-hat and ESS on the raw coordinates misleading. Instead the models sample
# the n - 1 identified coordinates x and map them into the sum-to-zero subspace
# with an orthonormal basis H: z = H·x. If x ~ N(0, I), then H·x ~ N(0, I - 𝟙𝟙ᵀ/n),
# exactly the distribution of the centred z - z̄ with z ~ N(0, I), so the
# posterior of every identified quantity is unchanged.

"""
    zerosum_basis(n) -> Matrix{Float64}   (n × n-1)

Orthonormal basis of the vectors of length `n` that sum to zero (Helmert):
column k has 1/√(k(k+1)) in rows 1..k and -k/√(k(k+1)) in row k+1.
"""
function zerosum_basis(n::Int)
    H = zeros(n, max(n - 1, 0))
    for k in 1:(n - 1)
        c = 1 / sqrt(k * (k + 1))
        H[1:k, k] .= c
        H[k + 1, k] = -k * c
    end
    return H
end

"""
    ZeroSumBases(g::GapData)

Bases for the gap models' effect vectors: `comp` (competitors, one block) and
`mach` (machine levels, one block per season, block-diagonal), with for each
free car coordinate a representative machine level (`mach_rep`, to look up its
season's pace scale). A season with a single machine level has no free
coordinate: its centred effect is 0.
"""
struct ZeroSumBases
    comp::Matrix{Float64}
    mach::Matrix{Float64}
    mach_rep::Vector{Int}
end
function ZeroSumBases(g::GapData)
    groups = [findall(==(s), g.mach_season) for s in 1:maximum(g.mach_season)]
    M = zeros(length(g.machines), sum(length(idx) - 1 for idx in groups))
    rep = Int[]
    j = 0
    for idx in groups
        n = length(idx)
        M[idx, (j + 1):(j + n - 1)] = zerosum_basis(n)
        append!(rep, fill(first(idx), n - 1))
        j += n - 1
    end
    return ZeroSumBases(zerosum_basis(length(g.competitors)), M, rep)
end

"""
    expand_effects(θ, B::ZeroSumBases) -> NamedTuple

Parameter values `θ` with the full-length effect vectors added from their
sum-to-zero coordinates: `z_comp` from `x_comp`, `z_mach` from `x_mach`,
`b_mach` from `bx_mach`, `u_slope` from `x_slope`. Values from chains saved
before #25 (which have the full vectors already) are returned unchanged.
"""
function expand_effects(θ::NamedTuple, B::ZeroSumBases)
    haskey(θ, :x_comp) && (θ = (; θ..., z_comp = B.comp * θ.x_comp))
    haskey(θ, :x_mach) && (θ = (; θ..., z_mach = B.mach * θ.x_mach))
    haskey(θ, :bx_mach) && (θ = (; θ..., b_mach = B.mach * θ.bx_mach))
    haskey(θ, :x_slope) && (θ = (; θ..., u_slope = B.comp * θ.x_slope))
    return θ
end
