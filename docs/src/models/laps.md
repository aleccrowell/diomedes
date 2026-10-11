# Lap-time model

[`lap_effects`](@ref) / [`fit_laps`](@ref): per-lap pace instead of whole-race
times (#12). It uses partial races before a retirement and has far more data
per driver–car pair. It is **in development**: see #12 for the current state.

## Outcome and clean laps

```
y = 100·log(t / median clean lap of the race)
```

This is on the same % scale as the race-time models. [`prepare_laps`](@ref)
keeps clean green-flag laps and excludes:

- lap 1 (the standing start);
- race-wide slow laps (safety car, VSC, red flag), detected when the field's
  median lap is over 1.25× the race median, plus the restart lap after each;
- a car's laps over 2× its own median lap (a red-flag stoppage, or a very long stop);
- pit in- and out-laps, either **recorded** (Ergast's pit stops, 2011 on) or
  **inferred** from the lap times ([`infer_pit_stops`](@ref)).

Inferred stops give the same effects as recorded ones over 2011–2019
(correlation 0.998 for drivers, 1.000 for car-seasons). That makes 1996–2010 usable.

## Model

```
y = γ[race] + δ[race]·(frac − mean frac) + driver + car-season + β₁·s + β₂·s² − (race mean) + noise
```

- `frac` is the lap's fraction of the race distance. The per-race slope δ
  (partially pooled) absorbs fuel burn and track evolution.
- `s` is stint age in tens of laps, a tyre-wear term.
- Noise is pace + loss, as in the race model. Traffic, mistakes and damage are
  incident losses.
- Driver and car-season effects are centred as in the race model. Every term is
  centred within its race, so γ is the race's mean lap.

## Long fits and warm starts

About 175k laps (2011–19) to 410k (1996–2019): a fit takes hours to a day on a
Raspberry Pi 5. Fits are checkpointed (see [Running long fits](@ref)). A
larger fit can warm-start from a smaller one: [`lap_warm_start`](@ref) maps the
adapted step size and diagonal metric across by parameter label, and
[`lap_warm_sampler`](@ref) runs AdvancedHMC's NUTS from there with a shorter
adaptation.

```julia
d = prepare_laps(fetch_laps(src, 2011:2019), fetch_results(src, 2011:2019); pits = :inferred)
chain = fit_laps(d; checkpoint = "output/laps_2011-2019_chain1", seed = 1000)
```

## Known issues before 1996–2019 results are usable

- **Refuelling (1994–2009):** the car lightens as fuel from its last stop burns
  off, which cancels the tyre-wear signal. The single stint term collapses, and
  the car effects shift.
- **Driver pace is constant across the fitted range**, unlike the race model's
  career slopes.
