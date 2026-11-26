# A NUTS metric that is diagonal except for one dense block (#142, #154).
#
# The race fit's slowest direction is a combination of its 33 scalars (σ_mach
# against τ_ch, with τ_en, τ_cx, δ_own, …): correlated coordinates that a diagonal
# metric can't match. A dense metric over all ~10k parameters would cost O(n²) per
# leapfrog step; this one is dense over the scalars only and diagonal elsewhere,
# written as AdvancedHMC's `RankUpdateEuclideanMetric` (M⁻¹ = A + B·D·Bᵀ with B
# selecting the block), so a step costs O(n·k) for k block coordinates.

"""
    BlockInvMetric(diag, idx, block)

An inverse metric M⁻¹ that is diagonal except over the coordinates `idx`, where it
is the dense matrix `block`. `diag` holds the whole diagonal (its entries at `idx`
equal `diag(block)`). Plain numbers, so it can be saved in a block file and read
back after the model's types change (see `run_nuts_checkpointed`).
"""
struct BlockInvMetric
    diag::Vector{Float64}
    idx::Vector{Int}
    block::Matrix{Float64}
    function BlockInvMetric(diag::AbstractVector, idx::AbstractVector{<:Integer}, block::AbstractMatrix)
        k = length(idx)
        size(block) == (k, k) || throw(DimensionMismatch("block is $(size(block)), idx has $k entries"))
        allunique(idx) && all(i -> 1 <= i <= length(diag), idx) || throw(ArgumentError("idx must be distinct indices into diag"))
        d = Vector{Float64}(diag)
        B = Matrix{Float64}(Symmetric(Matrix{Float64}(block)))
        d[idx] .= LinearAlgebra.diag(B)
        return new(d, Vector{Int}(idx), B)
    end
end
Base.length(b::BlockInvMetric) = length(b.diag)
Base.string(b::BlockInvMetric) = "BlockInvMetric($(length(b.diag)) coordinates, dense over $(length(b.idx)))"

"Block metric from a diagonal `Minv`, with the block starting diagonal (e.g. a warm start's)."
BlockInvMetric(Minv::AbstractVector, idx::AbstractVector{<:Integer}) =
    BlockInvMetric(Minv, idx, Matrix(Diagonal(Vector{Float64}(Minv)[idx])))

"""
    rank_update_metric(b::BlockInvMetric) -> AdvancedHMC.RankUpdateEuclideanMetric

The metric with inverse M⁻¹ = A + B·D·Bᵀ equal to `b`: A is `b.diag`, B selects the
block's coordinates, D = block − diag(block).
"""
function rank_update_metric(b::BlockInvMetric)
    n, k = length(b.diag), length(b.idx)
    B = zeros(n, k)
    for (j, i) in enumerate(b.idx)
        B[i, j] = 1.0
    end
    D = b.block - Diagonal(LinearAlgebra.diag(b.block))
    return AdvancedHMC.RankUpdateEuclideanMetric(Diagonal(copy(b.diag)), B, D)
end

"The `BlockInvMetric` of a metric built by `rank_update_metric`."
function block_inv_metric(m::AdvancedHMC.RankUpdateEuclideanMetric)
    idx = [findfirst(!iszero, c) for c in eachcol(m.B)]
    a = Vector{Float64}(m.A.diag)
    return BlockInvMetric(a, idx, Matrix(Diagonal(a[idx])) + Matrix(m.D))
end

AdvancedHMC.renew(::AdvancedHMC.RankUpdateEuclideanMetric, b::BlockInvMetric) = rank_update_metric(b)

"""
    BlockCovAdaptor(M0::BlockInvMetric)

Mass-matrix adaptor for Stan's windowed warm-up (`StanHMCAdaptor`): a Welford
variance of every coordinate and a Welford covariance of the block's, with Stan's
regularisation (`WelfordVar`, `WelfordCov`). Starts from `M0`.
"""
mutable struct BlockCovAdaptor <: AdvancedHMC.Adaptation.MassMatrixAdaptor
    idx::Vector{Int}
    var::AdvancedHMC.WelfordVar
    cov::AdvancedHMC.WelfordCov
    current::BlockInvMetric
end
function BlockCovAdaptor(M0::BlockInvMetric)
    return BlockCovAdaptor(M0.idx, AdvancedHMC.WelfordVar{Float64}((length(M0),); var = copy(M0.diag)),
                           AdvancedHMC.WelfordCov{Float64}((length(M0.idx),); cov = copy(M0.block)), M0)
end
Base.show(io::IO, a::BlockCovAdaptor) = print(io, "BlockCovAdaptor(dense over $(length(a.idx)))")
AdvancedHMC.Adaptation.getM⁻¹(a::BlockCovAdaptor) = a.current
function AdvancedHMC.Adaptation.resize_adaptor!(a::BlockCovAdaptor, size_θ::Tuple{Int})
    first(size_θ) == length(a.current) ||
        throw(DimensionMismatch("the block metric has $(length(a.current)) coordinates, the sampler $(first(size_θ))"))
    return nothing
end
function AdvancedHMC.Adaptation.reset!(a::BlockCovAdaptor)
    AdvancedHMC.Adaptation.reset!(a.var)
    AdvancedHMC.Adaptation.reset!(a.cov)
    return nothing
end
function Base.push!(a::BlockCovAdaptor, θ::AbstractVector{<:AbstractFloat})
    push!(a.var, θ)
    push!(a.cov, θ[a.idx])
    return nothing
end
function AdvancedHMC.Adaptation.update!(a::BlockCovAdaptor)
    a.var.n >= a.var.n_min || return nothing
    a.current = BlockInvMetric(AdvancedHMC.Adaptation.get_estimation(a.var), a.idx,
                               AdvancedHMC.Adaptation.get_estimation(a.cov))
    return nothing
end

"""
    block_nuts(M0::BlockInvMetric, ϵ; δ = 0.8, adapt = true)

NUTS (AdvancedHMC via `externalsampler`, uncompiled ReverseDiff) with the block
metric `M0` and step size `ϵ`. With `adapt`, Stan's windowed adaptation of the step
size and of the metric (diagonal plus the dense block, `BlockCovAdaptor`), as
Turing's `NUTS` does for a diagonal one; without, both stay fixed (as `fixed_nuts`).
"""
function block_nuts(M0::BlockInvMetric, ϵ::Real; δ::Real = 0.8, adapt::Bool = true)
    metric = rank_update_metric(M0)
    integrator = AdvancedHMC.Leapfrog(Float64(ϵ))
    κ = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
    adaptor = adapt ? AdvancedHMC.StanHMCAdaptor(BlockCovAdaptor(M0), AdvancedHMC.StepSizeAdaptor(δ, integrator)) :
              AdvancedHMC.NoAdaptation()
    return externalsampler(AdvancedHMC.HMCSampler(κ, metric, adaptor); adtype = gap_sampler().adtype)
end

fixed_nuts(Minv::BlockInvMetric, ϵ::Real) = block_nuts(Minv, ϵ; adapt = false)

"""
    scalar_indices(model; maxlen = 3) -> (n, idx)

The length `n` of `model`'s sampler (linked, unconstrained) vector, and the
positions in it of every variable with at most `maxlen` elements: the scalars,
and short vectors such as the three circuit-slope scales τ_cx and the track
coefficients. For the published race model these are the 33 quantities of the
`scalars` group of `identified_draws`.
"""
function scalar_indices(model; maxlen::Int = 3)
    DPPL = Turing.DynamicPPL
    vi = DPPL.link!!(DPPL.VarInfo(model), model)
    idx, pos = Int[], 0
    for vn in keys(vi)
        len = length(DPPL.getindex_internal(vi, vn))
        len <= maxlen && append!(idx, (pos + 1):(pos + len))
        pos += len
    end
    pos == length(vi[:]) || error("variable lengths ($pos) don't add up to the vector's ($(length(vi[:])))")
    return pos, idx
end
