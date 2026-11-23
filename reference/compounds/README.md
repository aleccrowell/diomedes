# Tyre compound reference tables

## `nominations.csv`: Pirelli's dry compounds behind SOFT / MEDIUM / HARD, per race

At each race, Pirelli nominates three of its dry compounds. The lap data (Jolpica laps,
OpenF1 and FastF1 stints) only carries the relative labels SOFT, MEDIUM and HARD. This table
gives the C-number behind each label, for every Grand Prix from 2021 to 2026 round 21.
See [#148](https://github.com/aleccrowell/diomedes/issues/148) for why it exists, and
[#124](https://github.com/aleccrowell/diomedes/issues/124) for the lap-model work that needs it.

| column | |
|---|---|
| `season`, `round`, `event_id` | as in `fetch_laps` (`event_id` = `<season>-<round, 2 digits>`) |
| `race`, `circuit_id` | Jolpica race name and `circuitId` |
| `hard`, `medium`, `soft` | the C-number behind each label (C1 hardest) |
| `era` | which compound range the C-numbers belong to (below) |
| `sources` | where the row comes from; space-separated |
| `notes` | checks, and anything unusual |

**Eras.** The same C-number is not the same tyre in every era.

- **2019–2021:** 13-inch tyres, C1–C5.
- **2022:** the first year of 18-inch tyres. The 2022 C1 was renamed C0 in 2023.
- **2023–2025:** a new C1 was added between the old C1 and C2. C6 came in 2025 (4 races).
- **2026:** a new range for the 2026 cars, with five slicks (C1–C5) and no C6.

Compare C-numbers across races only within an era.

**Sources.**

- [harningle/fia-doc](https://github.com/harningle/fia-doc) `tyres.json` (Apache-2.0): parsed
  from the FIA event notes and Pirelli previews. It covers 2021, 2022 and 2023 rounds 1–14. The file was removed from
  the repository's main branch in March 2026 and is read here from the commit before (9192387).
- [AnassNadeem/ARIS](https://github.com/AnassNadeem/ARIS) `data/compounds/nominations.json`
  (MIT): one Pirelli press-release URL per row, copied into `sources`. It covers 13 races of 2022,
  2023–2025, and 2026 rounds 1–14.
- formula1.com's "What tyres will the teams and drivers have…" articles, which quote Pirelli,
  cover 2026 rounds 15–21.

**Checks.**

- **Overlap:** the two repositories agree on all 28 races they share (13 in 2022, 14 in 2023,
  and 2021 Netherlands).
- **2021**, from fia-doc alone, agrees with Pirelli's pre-season plan for the whole year
  ([Motorsport Week, 2021-02-19](https://www.motorsportweek.com/2021/02/19/pirelli-outlines-tyre-choices-for-entire-2021-f1-season)),
  with two expected exceptions. The Austrian GP (round 9) got a softer selection than planned,
  and Qatar (round 20) was added to the calendar later.
- **Spot checks:** six randomly chosen single-source rows match primary sources: four ARIS rows
  against their Pirelli releases, and two fia-doc-only 2022 rows against formula1.com and the
  Pirelli previews. They are marked in `notes`.

**Skipped steps.** In 131 of the 135 races, the three compounds are consecutive C-numbers.
The exceptions are 2022 Australia (C2/C3/C5), 2025 Belgium and the United States (C1/C3/C4),
and 2025 Mexico City (C2/C4/C5).
