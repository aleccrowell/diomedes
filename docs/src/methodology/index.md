# Methodology

How much of a racing result is the driver, and how much is the car? This
section explains how the models answer that from public timing data, what they
can and can't identify, and the evidence behind each modelling choice. It is
written for readers comfortable with Bayesian hierarchical models; the
[manual](@ref "Models overview") and [API reference](@ref "API: Gap and pace + loss models")
cover how to run them.

!!! note "Work in progress"
    The methodology pages are being written (#45). Until then, the model pages
    of the manual give each model's structure, and the linked GitHub issues hold
    the evidence.

## Outline

1. **The question and what is identifiable.** Crossed driver and car-season
   effects; why car effects are only identified within a season, and how
   drivers who change teams link seasons. See [Mean structure](@ref).
2. **Data.** Results from 1950, lapped cars as interval-censored gaps, red-flag
   suspensions, inferred pit laps. See [Data](@ref).
3. **Noise.** Why pace noise and incident losses are modelled separately.
   See [Pace + loss model](@ref).
4. **Model progression.** Each added term (race duration, era paths, pace
   scale, career curves, big losses, in-season development) and the
   out-of-sample evidence for it. See [Which pace model to use](@ref).
5. **Retirements and selection.** A competing-risks model of retirements, and a
   check of whether dropping retirements biases the pace effects. See
   [Retirement model](@ref) and [Retirements and selection](@ref).
6. **The lap-time model** and how its effects agree with the race model's. See
   [Lap-time model](@ref).
7. **Limitations.** What the data can't tell apart, and known
   mis-specifications.
