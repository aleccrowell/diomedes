# Checks against known answers

Agreement between the race and lap models is a consistency check, not validation: they
can share a flaw. These checks depend on neither model's own output
([#120](https://github.com/aleccrowell/diomedes/issues/120)).

## Simulation recovery

`scripts/sim_recovery.jl` fits the race model to synthetic results whose driver, car and
career-slope effects are known, and checks they come back.
- **Grid:** 8 seasons, 10 teams of 2 drivers, 20 races each (3,200 results).
- **Turnover:** each season the 3 oldest drivers retire, 3 rookies join and 4 pairs swap teams.
- **Planted:** a dominant car (about 2 % faster than the field from season 4); a star driver
  (−1.0 %) who joins it that season, so the two then share every result; and a rookie who
  improves by 2.5 % per decade of age.
- **Noise:** pace + loss as the model assumes. Each fit is 4 × 1,000 draws.

**Without walks** ([#143](https://github.com/aleccrowell/diomedes/pull/143)): car-season
effects drawn per season, fitted with `pl_kappa_slope_dev`, 3 seeds.
- **Driver and car effects come back well:** correlation with the truth ≥ 0.96, slope
  of estimate on truth 0.93–0.99. 90 % intervals cover the truth near 90 % in two seeds,
  74–78 % in the third.
- **A dominant car passes some of its advantage to its star.** Where the car was about 2 %
  ahead, the star came out 0.15–0.21 % too fast and the car 0.2–0.3 % too slow in every
  dominant season. That is more than shrinkage alone would take. Where the car wasn't
  dominant, both came back right.
- **Career slopes are shrunk:** slope of estimate on truth 0.20–0.73. The improving rookie came
  out at −2.0 in two seeds and −0.9 in the third.

**With walks** ([#145](https://github.com/aleccrowell/diomedes/pull/145)): car form generated
as the published race model sees it. Each team is a chassis walk; three engine builders also
walk, each the works supplier of one team; there is a residual per car-season. Fitted with
`pl_kappa_slope_walk_walk1_works`, 1 seed.
- **Recovery:** drivers correlate 0.977 with the truth (coverage 90 %) and car-season
  totals 0.997 (94 %). The dominant car comes back within 0.1 % each season, and the star
  is off by only 0.08 %. The improving rookie came out at −2.31 against −2.53.
- **Sampling** struggles as in the published fit: depth 9–10, and a slow split of car
  form between the per-season residual and the chassis walk. Sampling that residual
  non-centred (`_ncres`) mixed the split about three times better per unit of compute on
  this data. On the real data it didn't: a 4 × 200-draw probe mixed the split no better than
  the published fit (ESS 17 against 23), so the synthetic test bed missed what makes the real
  posterior slow ([#142](https://github.com/aleccrowell/diomedes/issues/142)).

Both use data generated from the fitted model's own assumptions, which is the best case.

## The constructors' championship

`scripts/external_check.jl` compares each car-season's effect with its final position in
the constructors' championship, a result neither model is fitted to
([#141](https://github.com/aleccrowell/diomedes/pull/141)). The statistic is the
per-season Spearman correlation over the cars in both.

| model | seasons | median Spearman |
|---|---|---|
| race (total car effects) | 67 | 0.88: 0.64 in the 1950s, rising to 0.95–0.96 from 2010 |
| lap, published fit (2021–2026, with track conditions) | 5 | 0.83 |
| lap, previous fit (2011–2026) | 14 | 0.94 (0.85 over 2021–2025) |

Points also reward reliability, strategy and the drivers, so the agreement can't be
perfect. Over the five seasons both cover, the race model agrees more closely (median
0.96). The published lap fit agrees less than the previous one did over the same
seasons (0.83 against 0.85), and puts some cars well away from their points: Mercedes
2024 fastest of the field (P4), Haas 2025 fourth (P8) and AlphaTauri 2021–22 three
places ahead, with McLaren 2022 and Red Bull 2025 three places behind.
