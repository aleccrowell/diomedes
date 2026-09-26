# Turing models.

const HalfNormal1 = truncated(Normal(0, 1); lower = 0)

"Subtract the mean so the entries sum to zero. Used for the competitor effects; see `crossed_effects`."
sum_to_zero(z::AbstractVector) = z .- mean(z)

"""
    crossed_effects(y, competitor, machine, n_comp, n_mach, σ_y, intercept)

Standardised stage time = [intercept +] competitor effect + machine-season effect + noise.

This is the Turing version of the legacy TFP model, with four changes:
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

Pass `σ_y = 1.0` to fix the noise scale as the legacy model did, or `nothing`
to estimate it.
"""
@model function crossed_effects(y, competitor, machine, n_comp, n_mach, σ_y_fixed, intercept)
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

"""
    default_sampler()

NUTS with reverse-mode AD. With hundreds to thousands of effects, Turing's
default forward-mode AD is far too slow; the model has no value-dependent
control flow, so a compiled ReverseDiff tape is safe.
"""
default_sampler() = NUTS(0.8; adtype = AutoReverseDiff(; compile = true))

crossed_effects(d::ModelData; σ_y = 1.0, intercept::Bool = false) =
    crossed_effects(d.y, d.competitor, d.machine, length(d.competitors), length(d.machines),
                    σ_y, intercept)

"""
    fit_effects(d::ModelData; σ_y=1.0, intercept=false, n_samples=1000, n_chains=1,
                sampler=default_sampler(), rng=Random.default_rng(), kwargs...) -> Chains

Sample the posterior of `crossed_effects`. Chains run in parallel threads when
`n_chains > 1` (start Julia with `-t N`).

`init = :near_prior_centre` (default) starts each chain near the prior centre
(see `near_prior_centre`); `init = :uniform` uses Turing's default, which
draws unconstrained values from U(-2, 2), i.e. scales anywhere from 0.14 to
7.4.

Pass an `IO` as `progress_log` to get flushed progress lines with an ETA every
`log_every` iterations (see `progress_logger`); useful when output goes to a
file. Other `kwargs` are passed to `sample`.
"""
function fit_effects(d::ModelData; σ_y = 1.0, intercept::Bool = false,
                     n_samples::Int = 1000, n_chains::Int = 1,
                     sampler = default_sampler(), rng = Random.default_rng(),
                     init::Symbol = :near_prior_centre,
                     progress::Bool = true, progress_log::Union{Nothing,IO} = nothing,
                     log_every::Int = 100, kwargs...)
    model = crossed_effects(d; σ_y, intercept)
    if init === :near_prior_centre
        inits = [near_prior_centre(rng, d; σ_y, intercept) for _ in 1:n_chains]
        kwargs = (; kwargs..., initial_params = n_chains == 1 ? only(inits) : inits)
    elseif init !== :uniform
        throw(ArgumentError("init must be :near_prior_centre or :uniform"))
    end
    if progress_log !== nothing
        kwargs = (; kwargs..., callback = progress_logger(progress_log; every = log_every,
                                                          total = total_iterations(sampler, n_samples)))
    end
    return n_chains == 1 ?
        sample(rng, model, sampler, n_samples; progress, kwargs...) :
        sample(rng, model, sampler, MCMCThreads(), n_samples, n_chains; progress, kwargs...)
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

# Iterations per chain including adaptation, mirroring Turing's defaults
# (NUTS(δ) adapts for min(1000, N ÷ 2) iterations, run on top of the N kept).
function total_iterations(sampler, n_samples)
    hasproperty(sampler, :n_adapts) || return n_samples
    n = sampler.n_adapts
    return n_samples + (n == -1 ? min(1000, n_samples ÷ 2) : n)
end

"""
    progress_logger(io=stdout; every=100, total=nothing)

An AbstractMCMC `callback` that prints the iteration, elapsed time and (given
`total` iterations per chain) an ETA every `every` iterations, flushing `io` so
progress is visible when output goes to a file.

The ETA uses the rate over the last `every` iterations, not the average since
the start: early adaptation is much slower than later sampling, so an average
overstates the remaining time several-fold. Threaded chains report separately,
labelled by thread id (approximate, since tasks can migrate between threads).
"""
function progress_logger(io::IO = stdout; every::Int = 100, total = nothing)
    t0 = time()
    last = Dict{Int,Float64}()
    lk = ReentrantLock()
    return function (rng, model, sampler, transition, state, iteration; kwargs...)
        iteration % every == 0 || return
        lock(lk) do
            now, tid = time(), Threads.threadid()
            recent = now - get(last, tid, t0)
            last[tid] = now
            eta = total === nothing ? "" :
                string(", ETA ", round((total - iteration) * recent / every / 60; digits = 1), " min")
            println(io, "  thread $tid: iter $iteration", total === nothing ? "" : "/$total",
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
