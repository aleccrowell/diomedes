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
    run_nuts_checkpointed(model, init, path; n_samples, n_adapts = 500, δ = 0.8, block = 100,
                          seed = 1, progress_log = nothing, log_every = 100) -> chain

One NUTS chain sampled in blocks, each saved to `"\$(path)_block<j>.jls"` with
the sampler's final state, so a run that is killed can be resumed by calling
this again: saved blocks are loaded and sampling continues from the last one.

Block 1 holds the `n_adapts` warm-up iterations (discarded) and the first
`block` draws. Later blocks resume from the saved state (step size and mass
matrix as adapted) with adaptation off. Block j uses `Xoshiro(seed + j)`, so a
resumed run gives the same draws as an uninterrupted one. A kill loses at most
the block in progress (the warm-up, if it is in block 1). The sampler uses
uncompiled ReverseDiff (see `gap_sampler`). `external`, an AdvancedHMC sampler
wrapped by `externalsampler` (e.g. `lap_warm_sampler`), replaces Turing's NUTS.
Returns the blocks concatenated.
"""
function run_nuts_checkpointed(model, init, path::AbstractString; n_samples::Int, n_adapts::Int = 500,
                               δ::Real = 0.8, block::Int = 100, seed::Int = 1,
                               progress_log::Union{Nothing,IO} = nothing, log_every::Int = 100,
                               external = nothing)
    adtype = gap_sampler().adtype
    chains, state = [], nothing
    for j in 1:cld(n_samples, block)
        file = "$(path)_block$(j).jls"
        if isfile(file)
            ch = rehash_chain!(deserialize(file))
            progress_log === nothing || (println(progress_log, "block $j: loaded from $file"); flush(progress_log))
        else
            nj = min(block, n_samples - (j - 1) * block)
            first = state === nothing
            ch = if external === nothing
                run_nuts(model, first ? [init] : nothing; n_samples = nj, n_chains = 1, ensemble = MCMCSerial(),
                         sampler = NUTS(first ? n_adapts : 0, δ; adtype), rng = Xoshiro(seed + j), progress = false,
                         progress_log, log_every, save_state = true,
                         (first ? (;) : (; initial_state = state))...)
            else        # an AdvancedHMC sampler (externalsampler): warm-up run, then trimmed here
                c = run_nuts(model, first ? [init] : nothing; n_samples = nj + (first ? n_adapts : 0), n_chains = 1,
                             ensemble = MCMCSerial(), sampler = external, rng = Xoshiro(seed + j), progress = false,
                             progress_log, log_every, save_state = true, n_adapts = first ? n_adapts : 0,
                             (first ? (;) : (; initial_state = state))...)
                first ? c[iter = (n_adapts + 1):(n_adapts + nj)] : c   # (`discard_initial` is not applied to external samplers)
            end
            serialize(file * ".tmp", ch)
            mv(file * ".tmp", file; force = true)      # a kill mid-write leaves no partial block
            progress_log === nothing || (println(progress_log, "block $j: saved $file"); flush(progress_log))
        end
        push!(chains, ch)
        state = only(Turing.loadstate(ch))
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
