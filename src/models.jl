# Turing models.

const HalfNormal1 = truncated(Normal(0, 1); lower = 0)

"Subtract the mean so the entries sum to zero. Used for the competitor effects; see `crossed_effects`."
sum_to_zero(z::AbstractVector) = z .- mean(z)

"""
    PairStats(d::ModelData)

Sufficient statistics of the Gaussian likelihood, grouped by (competitor,
machine-season) pair: count `n`, sum of y `s`, plus the total `syy = Σy²` and
number of observations `N`. Every observation in a pair has the same mean μ_p,
so

    Σᵢ (yᵢ - μᵢ)² = syy - 2 Σₚ sₚ μₚ + Σₚ nₚ μₚ²

exactly. The likelihood then costs O(pairs) instead of O(observations) and has
a much smaller memory footprint: 1311 pairs vs 6400 rows for F1, and far fewer
pairs than rows for rallying, where a pair repeats on every stage.
"""
struct PairStats
    comp::Vector{Int}
    mach::Vector{Int}
    n::Vector{Float64}
    s::Vector{Float64}
    syy::Float64
    N::Int
end

function PairStats(d::ModelData)
    acc = Dict{Tuple{Int,Int},Tuple{Int,Float64}}()
    for (c, m, y) in zip(d.competitor, d.machine, d.y)
        n, s = get(acc, (c, m), (0, 0.0))
        acc[(c, m)] = (n + 1, s + y)
    end
    keys_ = sort!(collect(keys(acc)))
    return PairStats(first.(keys_), last.(keys_), [Float64(acc[k][1]) for k in keys_],
                     [acc[k][2] for k in keys_], sum(abs2, d.y), length(d.y))
end

"""
    crossed_effects(d::ModelData; σ_y = 1.0, intercept = false)

Standardised stage time = [intercept +] competitor effect + machine-season effect + noise.

This is the Turing version of the legacy TFP model, with these changes:
- it is fully Bayesian. The legacy model estimated the intercept and the
  effect scales by gradient steps inside MCMC (Monte Carlo EM); here they get
  weakly informative priors and are sampled along with everything else.
- the effects are non-centred (`effect = σ * z`, `z ~ N(0, 1)`), which samples
  much better when the scales are small or poorly identified.
- no intercept by default. Times are z-scored within each stage, so `y` has
  mean exactly 0 and an intercept is only identified jointly with the mean of
  the effects. The resulting posterior ridge made NUTS take ~6x more leapfrog
  steps per iteration on the full F1 data (92 vs 15), and the intercept
  wandered (0.30 ± 0.04) while predictions stayed the same. Pass
  `intercept = true` to include it anyway.
- competitor effects sum to zero (`σ_comp * (z .- mean(z))`), so they read as
  "relative to the average competitor in the data" and the machine-season
  effects carry the overall level. Without this, adding c to every competitor
  effect and subtracting it from every machine effect leaves predictions
  unchanged. That direction is set only by the priors (mean driver and mean car
  effect correlated at -0.99 across draws) and mixes slowly. Subtracting the
  mean takes it out of the likelihood: `mean(z_comp)` then just samples its N(0, 1/n) prior.
- the likelihood is evaluated from per-pair sufficient statistics (`PairStats`),
  which gives an identical posterior to the per-observation form
  (`crossed_effects_obs`, kept as the reference) with much less memory traffic.
  On a Raspberry Pi 5, gradients over the per-observation form were memory
  bound: 4 parallel chains each ran ~4x slower than one chain alone.

Pass `σ_y = 1.0` to fix the noise scale as the legacy model did, or `nothing`
to estimate it.
"""
crossed_effects(d::ModelData; σ_y = 1.0, intercept::Bool = false) =
    crossed_effects_pairs(PairStats(d), length(d.competitors), length(d.machines), σ_y, intercept)

@model function crossed_effects_pairs(ps::PairStats, n_comp, n_mach, σ_y_fixed, intercept)
    if intercept
        α ~ Normal(0, 1)
    else
        α = 0.0
    end
    σ_comp ~ HalfNormal1
    σ_mach ~ HalfNormal1
    z_comp ~ filldist(Normal(), n_comp)
    z_mach ~ filldist(Normal(), n_mach)
    if σ_y_fixed === nothing
        σ_y ~ HalfNormal1
    else
        σ_y = σ_y_fixed
    end
    μ = α .+ σ_comp .* sum_to_zero(z_comp)[ps.comp] .+ σ_mach .* z_mach[ps.mach]
    sse = ps.syy - 2 * dot(ps.s, μ) + dot(ps.n, μ .^ 2)
    @addlogprob! -ps.N * (log(σ_y) + log(2π) / 2) - sse / (2 * σ_y^2)
end

"Per-observation form of `crossed_effects`; same posterior, used as the reference in tests."
@model function crossed_effects_obs(y, competitor, machine, n_comp, n_mach, σ_y_fixed, intercept)
    if intercept
        α ~ Normal(0, 1)
    else
        α = 0.0
    end
    σ_comp ~ HalfNormal1
    σ_mach ~ HalfNormal1
    z_comp ~ filldist(Normal(), n_comp)
    z_mach ~ filldist(Normal(), n_mach)
    if σ_y_fixed === nothing
        σ_y ~ HalfNormal1
    else
        σ_y = σ_y_fixed
    end
    μ = α .+ σ_comp .* sum_to_zero(z_comp)[competitor] .+ σ_mach .* z_mach[machine]
    y ~ MvNormal(μ, σ_y^2 * I)
end

crossed_effects_obs(d::ModelData; σ_y = 1.0, intercept::Bool = false) =
    crossed_effects_obs(d.y, d.competitor, d.machine, length(d.competitors), length(d.machines),
                        σ_y, intercept)

"""
    default_sampler()

NUTS with reverse-mode AD. With hundreds to thousands of effects, Turing's
default forward-mode AD is far too slow; the model has no value-dependent
control flow, so a compiled ReverseDiff tape is safe.
"""
default_sampler() = NUTS(0.8; adtype = AutoReverseDiff(; compile = true))

"""
    fit_effects(d::ModelData; σ_y=1.0, intercept=false, n_samples=1000, n_chains=1,
                ensemble=MCMCSerial(), sampler=default_sampler(), rng=Random.default_rng(), kwargs...) -> Chains

Sample the posterior of `crossed_effects`. Chains run in parallel threads when
`n_chains > 1` according to `ensemble`.

`ensemble` defaults to `MCMCSerial()`: on a Raspberry Pi 5, 4 chains under
`MCMCThreads()` achieved less total throughput than running them one after
another (per-gradient time rose from 0.23 to 1.5 ms per thread), and separate
processes only reached ~1.5x. Pass `ensemble = MCMCThreads()` (with `julia -t N`)
on hardware where threads scale.

`init = :near_prior_centre` (default) starts each chain near the prior centre
(see `near_prior_centre`); `init = :uniform` uses Turing's default, which
draws unconstrained values from U(-2, 2), i.e. scales anywhere from 0.14 to
7.4.

Pass an `IO` as `progress_log` to get flushed progress lines with an ETA every
`log_every` iterations (see `progress_logger`); useful when output goes to a
file. Other `kwargs` are passed to `sample`.
"""
function fit_effects(d::ModelData; σ_y = 1.0, intercept::Bool = false,
                     n_samples::Int = 1000, n_chains::Int = 1, ensemble = MCMCSerial(),
                     sampler = default_sampler(), rng = Random.default_rng(),
                     init::Symbol = :near_prior_centre,
                     progress::Bool = true, progress_log::Union{Nothing,IO} = nothing,
                     log_every::Int = 100, kwargs...)
    init in (:near_prior_centre, :uniform) ||
        throw(ArgumentError("init must be :near_prior_centre or :uniform"))
    inits = init === :uniform ? nothing :
        [near_prior_centre(rng, d; σ_y, intercept) for _ in 1:n_chains]
    return run_nuts(crossed_effects(d; σ_y, intercept), inits; n_samples, n_chains, ensemble,
                    sampler, rng, progress, progress_log, log_every, kwargs...)
end

# Shared NUTS driver: initial values (one per chain, or `nothing` for Turing's
# default), optional progress log, serial or parallel chains.
#
# AbstractMCMC does not call the callback during discarded warm-up iterations,
# and Turing discards all adaptation iterations by default. A progress log would
# then only start after warm-up, which is often most of the run. So when
# logging, warm-up draws are kept (every iteration reaches the callback) and
# trimmed from the returned chain, which is therefore the same as without
# logging. Callers passing their own `discard_*` options are left alone.
function run_nuts(model, inits; n_samples, n_chains, ensemble, sampler, rng, progress,
                  progress_log, log_every, kwargs...)
    if inits !== nothing
        kwargs = (; kwargs..., initial_params = n_chains == 1 ? only(inits) : inits)
    end
    n_warmup = warmup_iterations(sampler, n_samples)
    trim = progress_log !== nothing && n_warmup > 0 &&
           !haskey(kwargs, :discard_initial) && !haskey(kwargs, :discard_adapt)
    N = n_samples
    if trim
        N = n_samples + n_warmup
        kwargs = (; kwargs..., nadapts = n_warmup, discard_adapt = false, discard_initial = 0)
    end
    if progress_log !== nothing
        kwargs = (; kwargs..., callback = progress_logger(progress_log; every = log_every,
                                                          total = n_samples + n_warmup,
                                                          n_warmup = trim ? n_warmup : 0,
                                                          n_chains = ensemble isa MCMCSerial ? n_chains : 1))
    end
    chain = n_chains == 1 ?
        sample(rng, model, sampler, N; progress, kwargs...) :
        sample(rng, model, sampler, ensemble, N, n_chains; progress, kwargs...)
    return trim ? chain[iter = (n_warmup + 1):N] : chain
end

"""
    step_size(state), with_step_size(state, ϵ)

The leapfrog step size in a sampler state, and a copy of the state with it
replaced: Turing's NUTS state, or a `TuringState` wrapping AdvancedHMC's (an
`externalsampler` run).
"""
step_size(st) = hasproperty(st, :kernel) ? st.kernel.τ.integrator.ϵ :
                hasproperty(st, :state) ? st.state.κ.τ.integrator.ϵ : st.κ.τ.integrator.ϵ
with_step_size(st, ϵ) = hasproperty(st, :kernel) ? (@set st.kernel.τ.integrator.ϵ = Float64(ϵ)) :
                        hasproperty(st, :state) ? (@set st.state.κ.τ.integrator.ϵ = Float64(ϵ)) :
                        (@set st.κ.τ.integrator.ϵ = Float64(ϵ))

"""
    fixed_nuts(Minv, ϵ)

NUTS (AdvancedHMC, via `externalsampler`, uncompiled ReverseDiff) with a fixed
diagonal inverse metric `Minv` and step size `ϵ`: no adaptation. The sampler of
every block after the first in `run_nuts_checkpointed`.
"""
function fixed_nuts(Minv::AbstractVector, ϵ::Real)
    metric = AdvancedHMC.DiagEuclideanMetric(Vector{Float64}(Minv))
    κ = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(AdvancedHMC.Leapfrog(Float64(ϵ)),
                                                                               AdvancedHMC.GeneralisedNoUTurn()))
    return externalsampler(AdvancedHMC.HMCSampler(κ, metric, AdvancedHMC.NoAdaptation()); adtype = gap_sampler().adtype)
end

# `n` draws from `last` (a chain's final draw) with a fixed metric and step size. The
# sampler's first transition is its starting point, a repeat of `last`'s, so one
# extra iteration is drawn and the first dropped.
function draws_from(model, last, Minv, ϵ, n; rng)
    c = run_nuts(model, [InitFromParams(last, Turing.FlexiChains.niters(last), 1)]; n_samples = n + 1, n_chains = 1,
                 ensemble = MCMCSerial(), sampler = fixed_nuts(Minv, ϵ), rng, progress = false, progress_log = nothing,
                 log_every = 100, n_adapts = 0)
    return c[iter = 2:(n + 1)]
end

# A copy of a chain without the sampler state, which would embed the model and its data
# (a later change to the data's type would then make the file unreadable).
strip_state(ch::Turing.FlexiChains.FlexiChain{T}) where {T} =
    Turing.FlexiChains.FlexiChain{T}(Turing.FlexiChains.niters(ch), Turing.FlexiChains.nchains(ch), ch._data; structures = ch._structures,
                              iter_indices = collect(Turing.FlexiChains.iter_indices(ch)),
                              chain_indices = collect(Turing.FlexiChains.chain_indices(ch)),
                              sampling_time = Turing.FlexiChains.sampling_time(ch),
                              last_sampler_state = fill(missing, Turing.FlexiChains.nchains(ch)))

"""
    read_block(file) -> (; chain, Minv, ϵ)

A block saved by `run_nuts_checkpointed`: its draws and the metric and step size
to continue with. Reads blocks of the earlier format too (a chain carrying the
sampler state), as long as the types the state embeds haven't changed.
"""
function read_block(file::AbstractString)
    x = deserialize(file)
    x isa NamedTuple && return (; chain = rehash_chain!(x.chain), Minv = x.Minv, ϵ = x.ϵ)
    ch = rehash_chain!(x)
    ad = lap_adapted(only(Turing.loadstate(ch)))
    return (; chain = ch, Minv = ad.Minv, ϵ = ad.ϵ)
end

"""
    block_chain(file) -> FlexiChain

The draws of a block saved by `run_nuts_checkpointed`, in either format. Unlike
`read_block`, it doesn't load the sampler state of an earlier-format block.
"""
function block_chain(file::AbstractString)
    x = deserialize(file)
    return rehash_chain!(x isa NamedTuple ? x.chain : x)
end

"""
    checkpoint_adapted(path) -> (; Minv, ϵ)

The metric and step size in the last saved block of a `run_nuts_checkpointed` run
(e.g. for a later warm start).
"""
function checkpoint_adapted(path::AbstractString)
    dir, base = dirname(path), basename(path) * "_block"
    files = filter(f -> startswith(f, base) && endswith(f, ".jls"), readdir(isempty(dir) ? "." : dir))
    isempty(files) && throw(ArgumentError("no saved blocks for $path"))
    last = argmax(f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]), files)
    b = read_block(joinpath(dir, last))
    return (; Minv = b.Minv, ϵ = b.ϵ)
end

"""
    tune_step_size(model, last, Minv, ϵ0, δ; factors = (1.15, 1.3, 1.45, 1.6), iters = 15, seed = 1)
        -> (ϵ, probe)

Raise an adapted step size `ϵ0` to the largest of `ϵ0 · factors` whose mean
acceptance over `iters` draws (metric `Minv`, no adaptation, from the chain
`last`'s final draw) is at least `δ` with no divergences, stopping at the first
factor that fails; keeps ϵ0 if none passes. Stan-style warm-up ends on a step
size whose acceptance in sampling is above the target (0.875 for δ = 0.8 on the
1996–2025 lap fit); a step size that still meets δ can halve the leapfrog steps
per draw at a tree-depth boundary (#62). The probe draws are discarded. `probe`
lists (factor, step size, acceptance, tree depth, divergences).
"""
function tune_step_size(model, last, Minv, ϵ0, δ; factors = (1.15, 1.3, 1.45, 1.6), iters::Int = 15, seed::Int = 1)
    best, probe = ϵ0, Tuple[]
    for (k, f) in enumerate(factors)
        c = draws_from(model, last, Minv, ϵ0 * f, iters; rng = Xoshiro(seed + 10_000 + k))
        acc, depth, div = mean(c[:acceptance_rate]), mean(c[:tree_depth]), count(c[:numerical_error] .> 0)
        push!(probe, (f, ϵ0 * f, round(acc; digits = 3), round(depth; digits = 2), div))
        (acc >= δ && div == 0) || break
        best = ϵ0 * f
    end
    return best, probe
end

"""
    run_nuts_checkpointed(model, init, path; n_samples, n_adapts = 500, δ = 0.8, block = 100,
                          seed = 1, progress_log = nothing, log_every = 100, external = nothing,
                          tune_step = false, fixed_step = nothing) -> chain

One NUTS chain sampled in blocks, each saved to `"\$(path)_block<j>.jls"`, so a
run that is killed can be resumed by calling this again: saved blocks are loaded
and sampling continues from the last one.

Block 1 holds the `n_adapts` warm-up iterations (discarded) and the first
`block` draws, sampled with Turing's NUTS or `external` (an AdvancedHMC sampler
wrapped by `externalsampler`, e.g. `lap_warm_sampler`). Every later block is
sampled with `fixed_nuts` at the adapted metric and step size, from the previous
block's last draw. A block file holds its draws and that metric and step size as
plain numbers (`read_block`), not the sampler's state: the state embeds the
model and its data, so a later change to their types made earlier blocks
unreadable. Block j uses `Xoshiro(seed + j)` and every block after the first is
sampled the same way whether or not the run was resumed, so a resumed run gives
the same draws as an uninterrupted one. A kill loses at most the block in
progress (the warm-up, if it is in block 1). With `tune_step = true`, the step
size is raised after warm-up to the largest that still meets the target
acceptance (`tune_step_size`, #62); the probe is written to
`"\$(path)_steptune.txt"`. `fixed_step` sets the step size of the blocks after the
first that this call samples (saved blocks keep theirs), e.g. to resume a chain
whose tuned step size left it at a deeper tree than the other chains (#66).
Returns the blocks concatenated.
"""
function run_nuts_checkpointed(model, init, path::AbstractString; n_samples::Int, n_adapts::Int = 500,
                               δ::Real = 0.8, block::Int = 100, seed::Int = 1,
                               progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100,
                               external = nothing, tune_step::Bool = false,
                               fixed_step::Union{Nothing,Real} = nothing)
    adtype = gap_sampler().adtype
    chains = []
    Minv, ϵ = nothing, nothing
    for j in 1:cld(n_samples, block)
        file = "$(path)_block$(j).jls"
        if isfile(file)
            b = read_block(file)
            ch, Minv, ϵ = b.chain, b.Minv, b.ϵ
            progress_log === nothing || (println(progress_log, "block $j: loaded from $file"); flush(progress_log))
        else
            nj = min(block, n_samples - (j - 1) * block)
            if j == 1
                ch = if external === nothing
                    run_nuts(model, [init]; n_samples = nj, n_chains = 1, ensemble = MCMCSerial(),
                             sampler = NUTS(n_adapts, δ; adtype), rng = Xoshiro(seed + j), progress = false,
                             progress_log, log_every, save_state = true)
                else        # warm-up run, then trimmed here (`discard_initial` is not applied to external samplers)
                    c = run_nuts(model, [init]; n_samples = nj + n_adapts, n_chains = 1, ensemble = MCMCSerial(),
                                 sampler = external, rng = Xoshiro(seed + j), progress = false, progress_log, log_every,
                                 save_state = true, n_adapts)
                    c[iter = (n_adapts + 1):(n_adapts + nj)]
                end
                ad = lap_adapted(only(Turing.loadstate(ch)))
                Minv, ϵ = ad.Minv, ad.ϵ
                ch = strip_state(ch)
            else
                fixed_step === nothing || (ϵ = Float64(fixed_step))
                ch = draws_from(model, last(chains), Minv, ϵ, nj; rng = Xoshiro(seed + j))
                ch = strip_state(ch)
                progress_log === nothing ||
                    (println(progress_log, "  thread 1: iter $nj/$nj (block $j)"); flush(progress_log))
            end
            serialize(file * ".tmp", (; chain = ch, Minv, ϵ, format = 2))
            mv(file * ".tmp", file; force = true)      # a kill mid-write leaves no partial block
            progress_log === nothing || (println(progress_log, "block $j: saved $file"); flush(progress_log))
        end
        push!(chains, ch)
        # after warm-up, once: before block 2 is first sampled (a resumed run whose
        # block 2 exists keeps the step size saved in it)
        if tune_step && fixed_step === nothing && j == 1 && cld(n_samples, block) > 1 && !isfile("$(path)_block2.jls")
            ϵ, rows = tune_step_size(model, ch, Minv, ϵ, δ; seed)
            open("$(path)_steptune.txt", "w") do io
                println(io, "factor\tstep_size\tacceptance\ttree_depth\tdivergences")
                for r in rows; println(io, join(r, "\t")); end
                println(io, "chosen\t", ϵ)
            end
            progress_log === nothing ||
                (println(progress_log, "step size tuned: ", round(ϵ; sigdigits = 3), " (probe in $(path)_steptune.txt)"); flush(progress_log))
        end
    end
    return reduce(vcat, chains)
end

"""
    near_prior_centre(rng, d; σ_y, intercept)

Initial values near the prior centre, jittered per chain so that R-hat still
compares chains started from different points: scales drawn from U(0.3, 0.7)
and standardised effects from N(0, 0.1²).
"""
function near_prior_centre(rng, d::ModelData; σ_y = 1.0, intercept::Bool = false)
    scale() = 0.3 + 0.4 * rand(rng)
    p = (; σ_comp = scale(), σ_mach = scale(),
         z_comp = 0.1 .* randn(rng, length(d.competitors)),
         z_mach = 0.1 .* randn(rng, length(d.machines)))
    σ_y === nothing && (p = (; p..., σ_y = scale()))
    intercept && (p = (; p..., α = 0.0))
    return InitFromParams(p)
end

# Adaptation (warm-up) iterations per chain, mirroring Turing's defaults:
# NUTS(δ) adapts for min(1000, N ÷ 2) iterations, run on top of the N kept.
function warmup_iterations(sampler, n_samples)
    hasproperty(sampler, :n_adapts) || return 0
    n = sampler.n_adapts
    return n == -1 ? min(1000, n_samples ÷ 2) : n
end

"""
    progress_logger(io=stdout; every=100, total=nothing, n_warmup=0, n_chains=1)

An AbstractMCMC `callback` that prints progress every `every` iterations,
flushing `io` so it is visible when output goes to a file. Given `total`
iterations per chain it adds ETAs.

- The chain ETA uses the rate over the last `every` iterations, not the average
  since the start: early adaptation is much slower than later sampling, so an
  average overstates the remaining time several-fold.
- With `n_chains > 1` run one after another (`MCMCSerial`), each line says
  which chain it is ("chain 2/4"; a new chain is detected when the iteration
  count resets). Once a chain has finished, it also gives a total ETA from the
  mean duration of finished chains, since every chain repeats the same
  slow-then-fast pattern.
- Threaded chains report separately, labelled by thread id (approximate, since
  tasks can migrate between threads).
- Iterations up to `n_warmup` are marked "(warm-up)". The callback only sees
  warm-up iterations if they are not discarded; `fit_effects` / `fit_gaps`
  arrange that when logging.
"""
function progress_logger(io::IO = stdout; every::Int = 100, total = nothing, n_warmup::Int = 0,
                         n_chains::Int = 1)
    t0 = time()
    last = Dict{Int,Float64}()        # time of the last report, per thread
    last_iter = Dict{Int,Int}()       # last iteration seen, per thread
    chain = Dict{Int,Int}()           # current chain number, per thread (serial runs)
    chain_start = Dict{Int,Float64}()
    finished = Float64[]              # durations of finished chains
    lk = ReentrantLock()
    return function (rng, model, sampler, transition, state, iteration; kwargs...)
        lock(lk) do
            now, tid = time(), Threads.threadid()
            if iteration < get(last_iter, tid, 0)          # iteration count reset: new chain
                push!(finished, now - get(chain_start, tid, t0))
                chain[tid] = get(chain, tid, 1) + 1
                chain_start[tid] = now
                last[tid] = now
            end
            last_iter[tid] = iteration
            iteration % every == 0 || return
            recent = now - get(last, tid, t0)
            last[tid] = now
            k = get(chain, tid, 1)
            label = n_chains > 1 ? "chain $k/$n_chains" : "thread $tid"
            eta = ""
            if total !== nothing
                chain_eta = (total - iteration) * recent / every
                eta = string(", chain ETA ", round(chain_eta / 60; digits = 1), " min")
                if n_chains > 1 && !isempty(finished)
                    total_eta = chain_eta + (n_chains - k) * mean(finished)
                    eta *= string(", total ETA ", round(total_eta / 60; digits = 1), " min")
                end
            end
            phase = iteration <= n_warmup ? " (warm-up)" : ""
            println(io, "  $label: iter $iteration", total === nothing ? "" : "/$total", phase,
                    ", $(round((now - t0) / 60; digits = 1)) min elapsed", eta)
            flush(io)
        end
    end
end

"""
    convergence_summary(chain) -> (; max_rhat, min_ess)

Worst-case R-hat and bulk ESS over all parameters. Rules of thumb: R-hat
below 1.01 and ESS above ~400 for 4 chains.
"""
function convergence_summary(chain)
    FC = Turing.FlexiChains
    worst(f, summ) = f(summ[vn] for vn in FC.parameters(summ))
    return (; max_rhat = worst(maximum, FC.rhat(chain)), min_ess = worst(minimum, FC.ess(chain)))
end

"""
    effects_table(chain, d::ModelData) -> (; competitors, machines)

Posterior summaries of the effects on the standardised-time scale (negative =
faster). Each table has the level label, posterior mean, sd and a 90% interval,
plus the number of observations behind it.
"""
function effects_table(chain, d::ModelData)
    # Turing returns a FlexiChain: `chain[sym]` is an (iter × chain) matrix whose
    # entries are scalars or, for vector parameters, vectors. Flatten to
    # (draws × levels); `vec` orders draws the same way for every parameter.
    draws(sym) = stack(vec(chain[sym]); dims = 1)
    comp = mapslices(sum_to_zero, draws(:z_comp); dims = 2) .* vec(chain[:σ_comp])
    mach = draws(:z_mach) .* vec(chain[:σ_mach])

    names = Dict(zip(d.rows.competitor_id, d.rows.competitor_name))
    summarise(draws, labels, idx) = DataFrame(
        label = labels,
        mean = vec(mean(draws; dims = 1)),
        sd = vec(std(draws; dims = 1)),
        q05 = [quantile(c, 0.05) for c in eachcol(draws)],
        q95 = [quantile(c, 0.95) for c in eachcol(draws)],
        n_obs = [count(==(i), idx) for i in eachindex(labels)],
    )
    competitors = summarise(comp, d.competitors, d.competitor)
    insertcols!(competitors, 2, :name => [names[c] for c in competitors.label])
    machines = summarise(mach, d.machines, d.machine)
    return (; competitors = sort!(competitors, :mean), machines = sort!(machines, :mean))
end
