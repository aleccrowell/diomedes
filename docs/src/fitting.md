# Running long fits

Full-history fits of the main models take from about an hour (pace + loss with a
few terms) to most of a day (every term, or the lap model) on a Raspberry Pi 5.
These conventions keep that manageable.

## Samplers

- [`default_sampler`](@ref Diomedes.default_sampler): NUTS with compiled
  ReverseDiff, used by the z-score model.
- [`gap_sampler`](@ref Diomedes.gap_sampler): NUTS with **uncompiled**
  ReverseDiff, used by every model with a fused likelihood. These models add
  their whole likelihood in one call with a hand-written reverse rule, so the AD
  tape stays tiny. A compiled tape would replay stale gradients from those
  rules.
- Chains start near the prior centre, jittered per chain, with race intercepts at
  each race's observed mean. Generic starting values put NUTS's first step size
  at about 1e-4 on these posteriors.

## Parallel chains as separate processes

On the Pi, threaded chains (`MCMCThreads`) gave *less* total throughput than
running them one after another: the time per gradient rose about 6× per thread.
Separate processes, one per core, do best. Every fit script therefore has a `chain k` mode
and a `combine` mode:

```
for k in 1 2 3 4; do
    julia --project scripts/gap_fit.jl jolpica1950-2025+partial pl_dur_rw_kappa chain $k > output/log_$k.txt 2>&1 &
done
# when all four finish:
julia --project scripts/gap_fit.jl jolpica1950-2025+partial pl_dur_rw_kappa combine
```

## Checkpoints and resuming

With `checkpoint = "<path prefix>"`, [`fit_paceloss`](@ref),
[`fit_retirement`](@ref) and [`fit_laps`](@ref) sample one chain in blocks
(default 100 draws), each saved as `<prefix>_block<j>.jls` with the adapted metric
and step size to continue from (see [`run_nuts_checkpointed`](@ref Diomedes.run_nuts_checkpointed)).

- Rerunning the same call after a kill resumes from the last saved block.
- Block `j` uses its own seeded RNG, so a resumed run gives the same draws as
  an uninterrupted one.
- A kill loses at most the block in progress. Block 1 holds the warm-up.
- Resume with the block size the chain was saved with: blocks are numbered by
  position, so another `block` is an error.
- Every iteration, warm-up included, is also appended to `<prefix>_trace.tsv`
  as it is drawn, so a block in progress can be inspected.
- `gap_fit.jl` (race and WRC fits) and `lap_fit.jl` take `block=<n>` in `chain`
  mode for this, with `step=<x>` (a fixed step size after block 1) and
  `warm=<tag>` (the adapted metric and step size of an earlier run, e.g. the
  probe). For `gap_fit.jl` the earlier run must have the same model and data (#131).
- Launch fits that run for hours detached, with `scripts/run_detached.sh <log>
  <pidfile> <command>`, which records the job's real PID for watching it.

Saved chains must be passed through [`rehash_chain!`](@ref) after
`deserialize`, or parameter lookups fail with a `KeyError` and `hcat` silently
yields `missing`:

```julia
chain = reduce(vcat, [rehash_chain!(deserialize(f)) for f in block_files])
```

## Progress and ETAs

Pass an `IO` as `progress_log` to get flushed progress lines every `log_every`
iterations ([`progress_logger`](@ref)). The chain ETA uses the rate over the
latest interval, because warm-up is several times slower than sampling.
Redirect to a file and read the file; don't pipe through `tail`, which
block-buffers.

## Convergence

- [`convergence_summary`](@ref): worst R-hat and bulk ESS over every sampled
  coordinate.
- [`identified_convergence`](@ref): the same over the quantities the
  likelihood identifies (centred effects, per-race incident rates), grouped and
  naming the worst level. It leaves out directions only the prior sees, so use it
  for the gap models.
- [`chain_modes`](@ref): groups the chains into modes when they settle in
  different solutions, so results can be reported by mode (#64).
- [`slow_directions`](@ref): when something mixes slowly, which combination of
  quantities the chains disagree on, over [`identified_draws`](@ref)
  (`scripts/slow_directions.jl`, #132). A direction loading a team's drivers
  against its car is a ridge a reparametrisation can remove (see
  [Identification](methodology/identification.md)).

Rules of thumb: R-hat below 1.01 and bulk ESS above about 400 over 4 chains, and
no divergences.

## Probes

Before a full fit, a change is tested with a short **probe**: the same model and
data, 4 chains of 300 warm-up and 200 draws, warm-started where possible, under its
own tag (`lap_fit.jl … probe`). The design work on wet conditions (#116) used about
eight probes before one full fit. What made that work:

- **Set the criteria before the run**, on the issue: what each quantity should do
  and what counts as a failure. Judging against them afterwards is then quick and
  honest. A criterion can still be overridden, but say so and why.
- **Include a check that doesn't depend on a model.** A criterion like "agrees with
  the previous full fit" penalises a fix when the previous fit shares the flaw being
  fixed. In #116 the previous fit had no per-race wet offsets, so its car effects
  carried race severity, and the fix "failed" agreement with it (0.87 against 0.9). A
  raw same-lap comparison showed the fix tracked the data better (0.67 against 0.59).
- **Report mixing, not just speed.** Give each chain's ESS and R-hat for the
  quantities that mix slowest (drivers, walks, any new term), as well as tree
  depth, step size and divergences.
- **Set a hard stop from the worst case.** At maximum tree depth an iteration takes
  1023 gradients. If a chain hasn't reached iteration 100 by 100 × 1023 × the
  gradient time, plus 5 %, it is stuck, whatever the typical times suggest.
- **A probe that misses is evidence, not waste.** Keep its code and its tag. Two of
  #116's failed probes (season and global centring) showed which term was missing.

A criteria comment can follow this template:

```
**Probe <name>: <design>, 4 × (300 + 200), warm-started from <tag>.**
Criteria, set before the run:
1. <quantity> within <range> (why)
2. R-hat ≤ 1.05 on every group; no divergences
3. <a model-free check>, e.g. correlation with raw estimates ≥ <x>
4. agreement with <the previous fit> where the change shouldn't matter
Hard stop: iteration 100 by <time> (100 × 1023 × <gradient time> + 5 %).
```

## Typical run times (Raspberry Pi 5)

| Fit | Time |
|---|---|
| z-score model, all F1 | under 2 min |
| pace + loss `pl_rw`, all F1, 4 chains in parallel | 40–70 min |
| pace + loss, the published model (`…_walk1_works_dinc`), 1950–2026, 4 chains in parallel | about 10.4 h per chain |
| lap model, 2011–2026 (316k laps) with every #108 and #113 term, warm-started | about 9 h per chain (1,000 draws) |
| lap model, 2021–2026 (124k laps) with the conditions terms (#116), warm-started | 6–23 h per chain (1,000 draws), depending on the step size the chain adapts |
| lap model, 2021–2026 with conditions, car and race wet terms and `rwrel` (#116), warm-started, fixed step | 13.5–14 h per chain (1,000 draws) |
