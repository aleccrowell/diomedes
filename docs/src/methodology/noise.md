# Noise: pace and incidents

## Two sources of variation

Most of a race's spread is small, symmetric variation in pace. Some of it is
something else: a spin, a slow stop, damage, a penalty. That can only *lose*
time, and sometimes a great deal of it. The pace + loss model writes the
residual of each result as

```math
y = \mu + \varepsilon + L, \qquad
\varepsilon \sim \mathcal{N}(0, \sigma), \qquad
L = \begin{cases} 0 & \text{with probability } 1-\pi \\
                  \text{Exponential(mean } \lambda) & \text{with probability } \pi \end{cases}
```

so the density is a mixture of a Gaussian and an exponentially modified
Gaussian (EMG). Both the density and the interval probabilities of lapped cars
have closed forms. These need care in the tails, where naive formulas cancel
catastrophically ([`paceloss_effects`](@ref)).

The alternative, symmetric heavy-tailed noise, is much worse. With Gaussian
noise, adding lapped cars inflated σ from 0.48% to 3.7%, and Student-t(4)
brought it to about 1.0%. Even so, Student-t(4) noise is about 1,660 elpd behind the
plain pace + loss model, and 2,905 ± 110 behind pace + loss with era terms
(`pl_rw`) ([#19](https://github.com/aleccrowell/diomedes/issues/19); fits from before the red-flag correction).

## Incidents vary by race and era

The incident probability ``\pi`` and the mean loss ``\lambda`` vary by race:

```math
\operatorname{logit}\pi_r = a_\pi + \beta^\pi_\text{dur}\, d_r + \text{era}^\pi_{s(r)}, \qquad
\log\lambda_r = a_\lambda + \beta^\lambda_\text{dur}\, d_r + \text{era}^\lambda_{s(r)} ,
```

with ``d_r`` the race's standardised log duration. The era terms are random
walks over seasons with Student-t(3) steps: mostly smooth, with occasional
jumps. That beat effects for each decade (by 53 ± 15 elpd) and for a priori
regulation eras (68 ± 17) ([#19](https://github.com/aleccrowell/diomedes/issues/19)). Grouped K-fold cross-validation confirmed
the ranking ([#35](https://github.com/aleccrowell/diomedes/issues/35)).

## A second, bigger loss

A single result can dominate a season's λ. Malaysia 2011's car 42 laps down
pushed that season's mean loss to 10.7% ([#16](https://github.com/aleccrowell/diomedes/issues/16)). So losses get two components:

```math
L \sim (1-\rho)\,\text{Exponential}(\lambda_r) + \rho\,\text{Exponential}(\lambda_2),
\qquad \lambda_2 = \lambda_\text{base}\, e^{\delta}, \ \delta > 0 .
```

With it, about 2% of incidents are big losses (ρ, with ``\lambda_2`` about 6.2×
the baseline, roughly 10%). The routine component fires about 73% of the time
with a mean of 1.7%, and σ falls from 0.51 to 0.40 ([#23](https://github.com/aleccrowell/diomedes/issues/23)). The fit improved by
155 ± 23 elpd ([#23](https://github.com/aleccrowell/diomedes/issues/23)).

!!! note "Reading π and λ"
    With the big-loss component, the "incident" component mostly describes
    routine time lost in a race (traffic, a slow stop), not crashes. π and λ are
    better read as the shape of the slow tail than as literal incident rates
    ([#23](https://github.com/aleccrowell/diomedes/issues/23)).

## The race level and the pace/loss split

The split between pace noise σ and routine losses (π, λ) is only weakly
identified. Shifting every race intercept trades against it at almost no
likelihood cost. With a fixed prior on the intercepts, the summed prior over
about 1,000 races decided the split, and changing that prior moved σ from 0.36
to 0.40 ([#32](https://github.com/aleccrowell/diomedes/issues/32)). Learning the common race level ``\mu_\gamma`` removes that
pull at equal out-of-sample fit (2.9 ± 6.0 elpd), and is the default ([#36](https://github.com/aleccrowell/diomedes/issues/36)).
The remaining uncertainty is honest: the incident intercept ``a_\pi`` has
posterior sd 0.19 ([#32](https://github.com/aleccrowell/diomedes/issues/32)). The driver and car effects don't depend on the split.
