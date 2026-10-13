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
    julia --project scripts/gap_fit.jl all pl_dur_rw_kappa chain $k > output/log_$k.txt 2>&1 &
done
# when all four finish:
julia --project scripts/gap_fit.jl all pl_dur_rw_kappa combine
```

## Checkpoints and resuming

With `checkpoint = "<path prefix>"`, [`fit_paceloss`](@ref),
[`fit_retirement`](@ref) and [`fit_laps`](@ref) sample one chain in blocks
(default 100 draws), each saved as `<prefix>_block<j>.jls` with the sampler state
(see [`run_nuts_checkpointed`](@ref Diomedes.run_nuts_checkpointed)).

- Rerunning the same call after a kill resumes from the last saved block.
- Block `j` uses its own seeded RNG, so a resumed run gives the same draws as
  an uninterrupted one.
- A kill loses at most the block in progress. Block 1 holds the warm-up.

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

Rules of thumb: R-hat below 1.01 and bulk ESS above about 400 over 4 chains, and
no divergences.

## Typical run times (Raspberry Pi 5)

| Fit | Time |
|---|---|
| z-score model, all F1 | under 2 min |
| pace + loss `pl_rw`, all F1, 4 chains in parallel | 40–70 min |
| pace + loss with every term, all F1 | about 5 h |
| lap model, 2011–19 (175k laps) | about 3.5 h warm-up, then 13.5 s per draw |
| lap model, 1996–2019 (410k laps), warm-started | about 10 h for 500 draws per chain |
