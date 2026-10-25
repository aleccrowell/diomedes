"""Write reference/cars/engine_builders.csv (#95): the builder behind each F1DB engine name.

An engine walk follows the works that built the engine, as the chassis walk follows
the works that built the car (see write_works.py): rebadged engines are keyed by
their builder. Engine makers not listed in SPECIAL are their own builder. Run from
the repo root:

    python3 scripts/cars/write_engines.py
"""
import csv, collections

WP = "https://en.wikipedia.org/wiki/"
F1DB = "https://github.com/f1db/f1db"

# engine maker -> [(first, last, builder, uncertain, notes, sources)]
SPECIAL = {
    "acer": [(2001, 2001, "ferrari", False, "Prost's 2001 engine: Ferrari's 2000 V10 rebadged.", f"{WP}Prost_AP04")],
    "petronas": [(1997, 2005, "ferrari", False, "Sauber's engines: previous-year Ferrari V10s rebadged.", f"{WP}Sauber_Motorsport")],
    "asiatech": [(2001, 2002, "peugeot", False, "Peugeot's V10 after Peugeot left F1, developed by Asiatech.", f"{WP}Asiatech")],
    "megatron": [(1987, 1988, "bmw", False, "BMW's M12/13 turbo rebadged after BMW's official withdrawal.", f"{WP}BMW_M12")],
    "tag": [(1983, 1987, "porsche", False, "TAG-badged engines were designed and built by Porsche.", f"{WP}TAG_Heuer_TTE_PO1")],
    "tag-heuer": [(2016, 2018, "renault", False, "Red Bull's Renault power units rebadged.", f"{WP}Red_Bull_Racing")],
    "toro-rosso": [(2017, 2017, "renault", False, "Toro Rosso's 2017 Renault power units (F1DB's engine name).", f"{F1DB} {WP}Scuderia_Toro_Rosso")],
    "mecachrome": [(1998, 1998, "renault", True, "Renault's RS9 V10 maintained by Mecachrome after Renault's withdrawal (judgement: Renault lineage).", f"{WP}Mecachrome")],
    "playlife": [(1998, 2000, "renault", True, "Benetton's Mecachrome-built Renault V10s rebadged (judgement: Renault lineage).", f"{WP}Mecachrome")],
    "supertec": [(1999, 2000, "renault", True, "Mecachrome-built Renault V10s sold by Supertec (judgement: Renault lineage).", f"{WP}Mecachrome")],
    "ford": [(1963, 2004, "cosworth", False, "Ford-badged engines of this era were built by Cosworth (DFV, HB, Zetec-R, CR).", f"{WP}Cosworth")],
    "european": [(2001, 2001, "cosworth", False, "Minardi's 2001 engine: Cosworth's Zetec-R V10 rebadged.", f"{WP}Minardi_PS01")],
    "fondmetal": [(2000, 2000, "cosworth", False, "Minardi's 2000 engine: Cosworth's Zetec-R V10 rebadged.", f"{WP}Minardi_M02")],
    "arrows": [(1998, 1999, "hart", True, "Arrows' own V10, developed from the Brian Hart V10 after TWR bought Hart (judgement: Hart lineage).", f"{WP}Arrows_A19")],
    "sauber": [(1993, 1993, "mercedes", False, "Ilmor-built V10 badged Sauber; Ilmor (Brixworth) built Mercedes' engines from 1994.", f"{WP}Sauber_C12")],
    "ilmor": [(1991, 1994, "mercedes", False, "Ilmor Engineering (Brixworth; customer V10s to 1994), which built Mercedes' engines from 1994 and became Mercedes-Benz HighPerformanceEngines: same works.", f"{WP}Ilmor")],
    "mercedes": [(1954, 1955, "mercedes-1954", False, "Daimler-Benz's own straight-8 (Stuttgart); unrelated to the Brixworth works.", f"{WP}Mercedes-Benz_in_Formula_One"),
                 (1994, 2026, "mercedes", False, "Brixworth (Ilmor, then Mercedes-Benz HPE/HPP).", f"{WP}Mercedes_AMG_High_Performance_Powertrains")],
    "bwt-mercedes": [(2019, 2020, "mercedes", False, "Racing Point's Mercedes power units under the BWT name.", f"{F1DB}")],
    "rbpt": [(2022, 2022, "honda", False, "Red Bull Powertrains-badged units built by Honda (HRC).", f"{WP}Red_Bull_Powertrains")],
    "honda-rbpt": [(2023, 2025, "honda", False, "Honda RBPT units built by Honda (HRC).", f"{WP}Red_Bull_Powertrains")],
    "red-bull-ford": [(2026, 2026, "red-bull-powertrains", False, "Red Bull Powertrains' own power unit (Milton Keynes), with Ford as partner; a new builder, not the Honda-built RBPT units of 2022-25.", f"{WP}Red_Bull_Powertrains")],
    "bmw": [(1952, 1954, "bmw-1952", False, "BMW 328-derived engines in privateer specials; unrelated to BMW's works engines.", f"{F1DB}"),
            (1967, 2009, "bmw", False, "BMW works engines: the 1967-68 Formula Two engines (Lola, German GP) and BMW Motorsport from 1982.", f"{WP}BMW_in_Formula_One")],
    "osella": [(1988, 1988, "alfa-romeo", True, "Alfa Romeo's 890T turbo, developed on by Osella after Alfa withdrew (judgement: Alfa lineage).", f"{WP}Osella")],
    "mugen-honda": [(1992, 2000, "mugen", True, "Mugen Motorsports' V10, developed from Honda's 1989-90 V10s; Mugen was never Honda-owned, though Honda took a larger part in development by 1998 (judgement: its own works).", f"{WP}Mugen_Motorsports")],
}

seasons = collections.defaultdict(set)
for r in csv.DictReader(open("reference/cars/entries.csv")):
    seasons[r["f1db_engine"]].add(int(r["season"]))

rows = []
for e, years in sorted(seasons.items()):
    spec = SPECIAL.get(e)
    if spec is None:
        rows.append(dict(f1db_engine=e, first_year=min(years), last_year=max(years), builder_id=e,
                         uncertain="", notes="own builder", sources=F1DB))
        continue
    for a, b, bid, unc, notes, src in spec:
        rows.append(dict(f1db_engine=e, first_year=a, last_year=b, builder_id=bid,
                         uncertain="uncertain" if unc else "", notes=notes, sources=src))
    for y in years:
        assert any(a <= y <= b for a, b, *_ in spec), (e, y)

with open("reference/cars/engine_builders.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "rows,", len({r['builder_id'] for r in rows}), "builders")
