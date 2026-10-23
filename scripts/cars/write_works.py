"""Write reference/cars/works.csv (#95): the chassis works of each F1DB constructor.

One row per constructor and year range in reference/cars/entries.csv. A works is
the organisation that builds the car (factory and core design staff); a rename or
sponsor change keeps it, a takeover that rebuilt the team doesn't. Constructors not
listed in SPECIAL are their own works. Run from the repo root:

    python3 scripts/cars/write_works.py
"""
import csv, collections

F1DB = "https://github.com/f1db/f1db (constructors chronology)"
WP = "https://en.wikipedia.org/wiki/"

# constructor -> [(first, last, works, uncertain, notes, sources)]
SPECIAL = {
    # Enstone: Toleman -> Benetton -> Renault -> Lotus F1 -> Renault -> Alpine
    "toleman": [(1981, 1985, "enstone", False, "Toleman Motorsport (Witney); Benetton bought it for 1986 with its staff (Byrne) and factory.", f"{F1DB} {WP}Toleman")],
    "benetton": [(1986, 2001, "enstone", False, "Benetton Formula: Toleman renamed after the 1985 takeover; moved Witney -> Enstone 1992.", f"{F1DB} {WP}Benetton_Formula")],
    "renault": [(1977, 1985, "renault-viry", False, "Renault Sport works team 1977-85 (Viry-Chatillon); unrelated to the Enstone team that raced as Renault from 2002.", f"{WP}Renault_in_Formula_One"),
                (2002, 2011, "enstone", False, "Benetton renamed after Renault's 2000 purchase; same Enstone factory.", f"{F1DB} {WP}Renault_in_Formula_One"),
                (2016, 2020, "enstone", False, "Lotus F1 bought back by Renault (Dec 2015); same Enstone factory.", f"{F1DB} {WP}Renault_in_Formula_One")],
    "lotus-f1": [(2012, 2015, "enstone", False, "Renault team renamed (Genii ownership); same Enstone factory. Unrelated to Team Lotus.", f"{F1DB} {WP}Lotus_F1")],
    "alpine": [(2021, 2025, "enstone", False, "Renault team renamed; same Enstone factory.", f"{F1DB} {WP}Alpine_F1_Team")],
    # Silverstone: Jordan -> Midland -> Spyker -> Force India -> Racing Point -> Aston Martin
    "jordan": [(1991, 2005, "silverstone", False, "Jordan Grand Prix (Silverstone).", f"{F1DB} {WP}Jordan_Grand_Prix")],
    "midland": [(2006, 2006, "silverstone", False, "Jordan renamed by its owner Midland; same factory.", f"{F1DB} {WP}Midland_F1_Racing")],
    "spyker": [(2007, 2007, "silverstone", False, "Midland bought by Spyker; same factory.", f"{F1DB} {WP}Spyker_F1")],
    "force-india": [(2008, 2018, "silverstone", False, "Spyker bought by Mallya and Mol; same factory. The 2018 administration and Racing Point entry from Belgium 2018 are the same works.", f"{F1DB} {WP}Force_India")],
    "racing-point": [(2019, 2020, "silverstone", False, "Force India bought out of administration by the Stroll consortium; same factory and staff.", f"{F1DB} {WP}Racing_Point_F1_Team")],
    "aston-martin": [(1959, 1960, "aston-martin-1959", False, "David Brown's Aston Martin works team (Feltham); unrelated to the 2021 team.", f"{WP}Aston_Martin_in_Formula_One"),
                     (2021, 2025, "silverstone", False, "Racing Point renamed; same Silverstone factory.", f"{F1DB} {WP}Aston_Martin_in_Formula_One")],
    # Tyrrell, then Brackley: BAR -> Honda -> Brawn -> Mercedes
    "tyrrell": [(1970, 1998, "tyrrell", False, "Tyrrell Racing (Ockham). Raced 1998 under BAR ownership.", f"{WP}Tyrrell_Racing")],
    "bar": [(1999, 2005, "brackley", True, "Break from Tyrrell (judgement): BAR bought Tyrrell's entry but built a new factory at Brackley with a new design team (Reynard, Oastler); F1DB chains them.", f"{F1DB} {WP}British_American_Racing")],
    "honda": [(1964, 1968, "honda-1964", False, "Honda's 1960s works team; unrelated to the Brackley team that raced as Honda 2006-08.", f"{WP}Honda_in_Formula_One"),
              (2006, 2008, "brackley", False, "BAR renamed after Honda's full purchase; same Brackley factory.", f"{F1DB} {WP}Honda_Racing_F1")],
    "brawn": [(2009, 2009, "brackley", False, "Honda team bought by its management; same factory and staff.", f"{F1DB} {WP}Brawn_GP")],
    "mercedes": [(1954, 1955, "mercedes-1954", False, "Daimler-Benz works team (Stuttgart); unrelated to the Brackley team.", f"{WP}Mercedes-Benz_in_Formula_One"),
                 (2010, 2025, "brackley", False, "Brawn bought by Mercedes; same Brackley factory.", f"{F1DB} {WP}Mercedes-Benz_in_Formula_One")],
    # Hinwil: Sauber -> BMW Sauber -> Sauber -> Alfa Romeo -> Kick Sauber
    "sauber": [(1993, 2005, "hinwil", False, "Sauber (Hinwil).", f"{F1DB} {WP}Sauber_Motorsport"),
               (2010, 2018, "hinwil", False, "BMW sold the team back to Peter Sauber; same factory.", f"{F1DB} {WP}Sauber_Motorsport")],
    "bmw-sauber": [(2006, 2009, "hinwil", False, "Sauber bought by BMW; same factory.", f"{F1DB} {WP}BMW_Sauber")],
    "alfa-romeo": [(1950, 1951, "alfa-romeo-1950", False, "Alfa Corse works team (Portello).", f"{WP}Alfa_Romeo_in_Formula_One"),
                   (1979, 1985, "alfa-romeo-1979", False, "Alfa Romeo works chassis: Autodelta 1979-82, run by Euroracing 1983-85.", f"{WP}Alfa_Romeo_in_Formula_One"),
                   (2019, 2023, "hinwil", False, "Sauber under Alfa Romeo title sponsorship; same Hinwil factory.", f"{F1DB} {WP}Alfa_Romeo_in_Formula_One")],
    "kick-sauber": [(2024, 2025, "hinwil", False, "Sauber renamed for sponsorship; same factory.", f"{F1DB} {WP}Sauber_Motorsport")],
    # Faenza: Minardi -> Toro Rosso -> AlphaTauri -> RB -> Racing Bulls
    "minardi": [(1985, 2005, "faenza", False, "Minardi (Faenza).", f"{F1DB} {WP}Minardi")],
    "toro-rosso": [(2006, 2019, "faenza", False, "Minardi bought by Red Bull; same Faenza factory.", f"{F1DB} {WP}Scuderia_Toro_Rosso")],
    "alphatauri": [(2020, 2023, "faenza", False, "Toro Rosso renamed; same factory.", f"{F1DB} {WP}Scuderia_AlphaTauri")],
    "rb": [(2024, 2024, "faenza", False, "AlphaTauri renamed; same factory.", f"{F1DB} {WP}Racing_Bulls")],
    "racing-bulls": [(2025, 2025, "faenza", False, "RB renamed; same factory.", f"{F1DB} {WP}Racing_Bulls")],
    # Milton Keynes: Stewart -> Jaguar -> Red Bull
    "stewart": [(1997, 1999, "milton-keynes", False, "Stewart Grand Prix (Milton Keynes).", f"{F1DB} {WP}Stewart_Grand_Prix")],
    "jaguar": [(2000, 2004, "milton-keynes", False, "Stewart bought by Ford, renamed Jaguar; same factory.", f"{F1DB} {WP}Jaguar_Racing")],
    "red-bull": [(2005, 2025, "milton-keynes", False, "Jaguar bought by Red Bull; same factory.", f"{F1DB} {WP}Red_Bull_Racing")],
    # Others chained in F1DB
    "lotus-racing": [(2010, 2011, "caterham", False, "Lotus Racing (Fernandes), new team in 2010; unrelated to Team Lotus. Renamed Caterham 2012 (moved Hingham -> Leafield).", f"{F1DB} {WP}Caterham_F1")],
    "caterham": [(2012, 2014, "caterham", False, "Lotus Racing renamed, same owner and staff.", f"{F1DB} {WP}Caterham_F1")],
    "virgin": [(2010, 2011, "manor", False, "Virgin Racing, run by Manor Grand Prix (Dinnington).", f"{F1DB} {WP}Virgin_Racing")],
    "marussia": [(2012, 2015, "manor", False, "Virgin renamed after Marussia's purchase; same works (Banbury from 2011).", f"{F1DB} {WP}Marussia_F1")],
    "manor": [(2016, 2016, "manor", False, "Marussia renamed after the 2015 administration; same works.", f"{F1DB} {WP}Manor_Racing")],
    "ligier": [(1976, 1996, "ligier", False, "Equipe Ligier (Vichy, Magny-Cours from 1989).", f"{F1DB} {WP}Ligier")],
    "prost": [(1997, 2001, "ligier", True, "Ligier bought by Alain Prost; same team, though it moved to Guyancourt in 1998 (judgement: same works).", f"{F1DB} {WP}Prost_Grand_Prix")],
    "osella": [(1980, 1990, "osella", False, "Osella Squadra Corse.", f"{F1DB} {WP}Osella")],
    "fondmetal": [(1991, 1992, "osella", False, "Osella renamed after Gabriele Rumi's takeover; same team.", f"{F1DB} {WP}Fondmetal")],
    "coloni": [(1987, 1989, "coloni", False, "Coloni.", f"{F1DB} {WP}Coloni")],
    "andrea-moda": [(1992, 1992, "andrea-moda", True, "Break from Coloni (judgement): Andrea Sassetti bought Coloni's entry but ran new Simtek-designed cars with a new team; F1DB chains them.", f"{F1DB} {WP}Andrea_Moda_Formula")],
    "footwork": [(1991, 1996, "arrows", False, "Arrows renamed for its owner Footwork; same team (Milton Keynes).", f"{F1DB} {WP}Arrows_Grand_Prix_International")],
    "arrows": [(1978, 1990, "arrows", False, "Arrows Grand Prix International.", f"{F1DB} {WP}Arrows_Grand_Prix_International"),
               (1997, 2002, "arrows", False, "Footwork renamed back after TWR's purchase; moved to Leafield.", f"{F1DB} {WP}Arrows_Grand_Prix_International")],
    "politoys": [(1972, 1972, "fwrc", False, "Frank Williams Racing Cars' own chassis (Politoys sponsor).", f"{F1DB} {WP}Frank_Williams_Racing_Cars")],
    "iso-marlboro": [(1973, 1974, "fwrc", False, "Frank Williams Racing Cars (Iso-Marlboro name).", f"{F1DB} {WP}Frank_Williams_Racing_Cars")],
    "frank-williams-racing-cars": [(1975, 1975, "fwrc", False, "Frank Williams Racing Cars.", f"{F1DB} {WP}Frank_Williams_Racing_Cars")],
    "wolf-williams": [(1976, 1976, "fwrc", False, "FWRC after Walter Wolf's majority purchase; same team (Reading).", f"{F1DB} {WP}Walter_Wolf_Racing")],
    "wolf": [(1977, 1979, "fwrc", False, "Walter Wolf Racing: FWRC after Frank Williams left; same works.", f"{F1DB} {WP}Walter_Wolf_Racing")],
    "williams": [(1978, 2025, "williams", False, "Williams Grand Prix Engineering, founded 1977 after Frank Williams left Wolf: a new works.", f"{WP}Williams_Racing")],
    "march": [(1970, 1977, "march", False, "March Engineering (Bicester).", f"{F1DB} {WP}March_Engineering"),
              (1981, 1982, "march", True, "March's 1981-82 return (run with RAM's John Macdonald); same March Engineering design base (judgement).", f"{F1DB} {WP}March_Engineering"),
              (1987, 1989, "march", True, "March's 1987 return (March Racing, Bicester), Leyton House-backed; F1DB chains it (judgement: same works across the gap).", f"{F1DB} {WP}March_Engineering"),
              (1992, 1992, "march", False, "Leyton House renamed back to March after its sale; same team.", f"{F1DB} {WP}Leyton_House")],
    "leyton-house": [(1990, 1991, "march", False, "March renamed for its owner Leyton House (Akagi); same team.", f"{F1DB} {WP}Leyton_House")],
    "theodore": [(1978, 1982, "theodore", False, "Theodore Racing (Teddy Yip).", "https://8w.forix.com/theodore.html"),
                 (1983, 1983, "ensign", True, "Ensign merged into Theodore for 1983 and raced the Ensign-designed N183 as the Theodore N183 (judgement: Ensign's works).", "https://8w.forix.com/theodore.html https://en.wikipedia.org/wiki/Ensign_Racing")],
    "venturi": [(1992, 1992, "larrousse", False, "Larrousse under the Venturi name; same team.", f"{WP}Larrousse")],
    "larrousse": [(1993, 1994, "larrousse", False, "Larrousse with its own chassis (its 1987-91 cars were Lola chassis, constructor lola).", f"{WP}Larrousse")],
}

counts = collections.defaultdict(set)
for r in csv.DictReader(open("reference/cars/entries.csv")):
    counts[r["f1db_constructor"]].add(int(r["season"]))

rows = []
for c, years in sorted(counts.items()):
    spec = SPECIAL.get(c)
    if spec is None:
        rows.append(dict(f1db_constructor=c, first_year=min(years), last_year=max(years), works_id=c,
                         uncertain="", notes="own works", sources="https://github.com/f1db/f1db"))
        continue
    for a, b, w, unc, notes, src in spec:
        rows.append(dict(f1db_constructor=c, first_year=a, last_year=b, works_id=w,
                         uncertain="uncertain" if unc else "", notes=notes, sources=src))
    for y in years:     # every season of the constructor in the data is covered
        assert any(a <= y <= b for a, b, *_ in spec), (c, y)

with open("reference/cars/works.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "rows,", len({r['works_id'] for r in rows}), "works")
