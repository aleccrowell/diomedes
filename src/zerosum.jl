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
    ZeroSumMap(groups, n_out)

The linear map x ↦ H·x for a block-diagonal Helmert basis: `groups[b]` are the
output rows of block b (a block of size m takes m - 1 inputs, in order), and
rows in no group are 0. Applied in O(n) (a reverse cumulative sum minus a
shifted term) with a hand-written adjoint, so the sampler's gradient does not
pay for a dense matrix product (`M * x` works as for the matrix; `Matrix(M)`
gives the dense form).

With w_k = x_k/√(k(k+1)) within a block: (H·x)_i = Σ_{k≥i} w_k - (i-1)·w_{i-1};
adjoint: x̄_k = (Σ_{i≤k} ȳ_i - k·ȳ_{k+1})/√(k(k+1)).
"""
struct ZeroSumMap
    groups::Vector{Vector{Int}}
    n_out::Int
    n_in::Int
end
ZeroSumMap(groups::Vector{Vector{Int}}, n_out::Int) = ZeroSumMap(groups, n_out, sum(length(r) - 1 for r in groups; init = 0))
Base.size(M::ZeroSumMap) = (M.n_out, M.n_in)
Base.size(M::ZeroSumMap, d::Integer) = size(M)[d]
Base.:*(M::ZeroSumMap, x::AbstractVector) = zerosum_apply(M, x)
function Base.Matrix(M::ZeroSumMap)
    H = zeros(M.n_out, M.n_in)
    j = 0
    for r in M.groups
        m = length(r)
        H[r, (j + 1):(j + m - 1)] = zerosum_basis(m)
        j += m - 1
    end
    return H
end

function zerosum_apply(M::ZeroSumMap, x::AbstractVector)
    length(x) == M.n_in || throw(DimensionMismatch("ZeroSumMap takes $(M.n_in) inputs, got $(length(x))"))
    y = zeros(eltype(x), M.n_out)
    j = 0
    for r in M.groups
        m = length(r)
        S = zero(eltype(x))
        for i in m:-1:1
            i <= m - 1 && (S += x[j + i] / sqrt(i * (i + 1)))
            y[r[i]] = i >= 2 ? S - (i - 1) * x[j + i - 1] / sqrt((i - 1) * i) : S
        end
        j += m - 1
    end
    return y
end

function zerosum_adjoint(M::ZeroSumMap, ȳ::AbstractVector)
    x̄ = zeros(eltype(ȳ), M.n_in)
    j = 0
    for r in M.groups
        m = length(r)
        P = zero(eltype(ȳ))
        for k in 1:(m - 1)
            P += ȳ[r[k]]
            x̄[j + k] = (P - k * ȳ[r[k + 1]]) / sqrt(k * (k + 1))
        end
        j += m - 1
    end
    return x̄
end

function ChainRulesCore.rrule(::typeof(zerosum_apply), M::ZeroSumMap, x::AbstractVector)
    pullback(ȳ) = (NoTangent(), NoTangent(), zerosum_adjoint(M, ChainRulesCore.unthunk(ȳ)))
    return zerosum_apply(M, x), pullback
end
ReverseDiff.@grad_from_chainrules zerosum_apply(M::ZeroSumMap, x::ReverseDiff.TrackedArray)

"""
    ZeroSumBases(g::GapData)

Bases for the gap models' effect vectors: `comp` (competitors, one block) and
`mach` (machine levels, one block per season), as `ZeroSumMap`s, with for each
free car coordinate a representative machine level (`mach_rep`, to look up its
season's pace scale). A season with a single machine level has no free
coordinate: its centred effect is 0.
"""
struct ZeroSumBases
    comp::ZeroSumMap
    mach::ZeroSumMap
    mach_rep::Vector{Int}
end
function ZeroSumBases(g::GapData)
    groups = [findall(==(s), g.mach_season) for s in 1:maximum(g.mach_season)]
    rep = reduce(vcat, [fill(first(r), length(r) - 1) for r in groups]; init = Int[])
    comp = ZeroSumMap([collect(eachindex(g.competitors))], length(g.competitors))
    return ZeroSumBases(comp, ZeroSumMap(groups, length(g.machines)), rep)
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
