# Turing models.

const HalfNormal1 = truncated(Normal(0, 1); lower = 0)

"""
    crossed_effects(y, competitor, machine, n_comp, n_mach, σ_y)

Standardised stage time = intercept + competitor effect + machine-season effect + noise.

This is the Turing version of the legacy TFP model, with two changes:
- it is fully Bayesian. The legacy model estimated the intercept and the
  effect scales by gradient steps inside MCMC (Monte Carlo EM); here they get
  weakly informative priors and are sampled along with everything else.
- the effects are non-centred (`effect = σ * z`, `z ~ N(0, 1)`), which samples
  much better when the scales are small or poorly identified.

Pass `σ_y = 1.0` to fix the noise scale as the legacy model did, or `nothing`
to estimate it.
"""
@model function crossed_effects(y, competitor, machine, n_comp, n_mach, σ_y_fixed)
    α ~ Normal(0, 1)
    σ_comp ~ HalfNormal1
    σ_mach ~ HalfNormal1
    z_comp ~ filldist(Normal(), n_comp)
    z_mach ~ filldist(Normal(), n_mach)
    if σ_y_fixed === nothing
        σ_y ~ HalfNormal1
    else
        σ_y = σ_y_fixed
    end
    μ = α .+ σ_comp .* z_comp[competitor] .+ σ_mach .* z_mach[machine]
    y ~ MvNormal(μ, σ_y^2 * I)
end

"""
    default_sampler()

NUTS with reverse-mode AD. With hundreds to thousands of effects, Turing's
default forward-mode AD is far too slow; the model has no value-dependent
control flow, so a compiled ReverseDiff tape is safe.
"""
default_sampler() = NUTS(0.8; adtype = AutoReverseDiff(; compile = true))

crossed_effects(d::ModelData; σ_y = 1.0) =
    crossed_effects(d.y, d.competitor, d.machine, length(d.competitors), length(d.machines), σ_y)

"""
    fit_effects(d::ModelData; σ_y=1.0, n_samples=1000, n_chains=1,
                sampler=default_sampler(), rng=Random.default_rng()) -> Chains

Sample the posterior of `crossed_effects`. Chains run in parallel threads when
`n_chains > 1` (start Julia with `-t N`).
"""
function fit_effects(d::ModelData; σ_y = 1.0, n_samples::Int = 1000, n_chains::Int = 1,
                     sampler = default_sampler(), rng = Random.default_rng(),
                     progress::Bool = true)
    model = crossed_effects(d; σ_y)
    return n_chains == 1 ?
        sample(rng, model, sampler, n_samples; progress) :
        sample(rng, model, sampler, MCMCThreads(), n_samples, n_chains; progress)
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
    comp = draws(:z_comp) .* vec(chain[:σ_comp])
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
