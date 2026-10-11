# Methodology

How much of a racing result is the driver, and how much is the car? This
section explains how the models answer that from public timing data, what they
can and can't identify, and the evidence behind each modelling choice. It is
written for readers comfortable with Bayesian hierarchical models; the
[manual](@ref "Models overview") and [API reference](@ref "API: Gap and pace + loss models")
cover how to run them.

## Pages

1. [The question and what is identifiable](@ref): crossed driver and car
   effects, what the data can't tell apart, and how convergence is measured.
2. [Data handling](@ref): the % gap outcome, lapped cars as intervals,
   red-flag suspensions, priors in the data's units, rallying, lap times.
3. [Noise: pace and incidents](@ref): Gaussian pace noise plus one-sided
   incident losses, era paths, and big losses.
4. [Model progression](@ref): each added term with its out-of-sample evidence
   and what it found, and how models were compared.
5. [Retirements and selection bias](@ref): a competing-risks model of
   retirements, and whether dropping them biases the pace effects.
6. [The lap-time model](@ref): per-lap pace, inferred pit laps, and agreement
   with the race model.
7. [Limitations](@ref).

Links of the form [#15](https://github.com/aleccrowell/diomedes/issues/15)
point to the GitHub issues and pull requests where each choice was designed and
tested; they hold the full evidence.
