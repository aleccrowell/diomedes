# Car effects against the constructors' championship, a result neither model is fitted to (#120).
#
#   julia --project scripts/external_check.jl [race=<cars.csv>] [laps=<tag>]
#
# Final constructors' standings: 1958–2020 from the Ergast dump (data/), 2021–2025 from
# the Jolpica API (cached in output/standings_<season>.json). Per season, the Spearman
# correlation between each car-season's effect and its championship position, over the cars in
# both. Points also reward reliability, strategy and the drivers, so agreement is expected
# to be high but not perfect; the check is in the seasons and cars where they part most.
# Race model: the site's cars.csv (scripts/export_site.jl race), whose `mean` is the car's total
# effect. gap_fit.jl's machine_effects.csv is only the residual part once a fit has chassis and
# engine walks, and against the championship it scores 0.36 for that reason. Lap model:
# output/<laps tag>_cars.csv.
using CSV, DataFrames, Statistics, Printf, JSON3, Downloads

opt(k, d) = (i = findfirst(a -> startswith(a, k * "="), ARGS); i === nothing ? d : String(split(ARGS[i], "="; limit = 2)[2]))
race_csv = opt("race", "docs/src/assets/data/race/cars.csv")
laps_tag = opt("laps", "laps_2011-2026_recorded_slopes_lead_tyremgmt_starts_age_circ_track_dinc_big_rwalk_nostartdrv_traffic_fpace_short_keepslow")

# final standings: (season, constructor id) => position
races = CSV.read("data/races.csv", DataFrame)
cons = CSV.read("data/constructors.csv", DataFrame)
cs = CSV.read("data/constructor_standings.csv", DataFrame)
last_race = combine(groupby(races, :year), [:round, :raceId] => ((r, id) -> id[argmax(r)]) => :raceId)
fin = innerjoin(innerjoin(cs, last_race; on = :raceId), select(cons, :constructorId, :constructorRef); on = :constructorId)
st = DataFrame(season = fin.year, id = String.(fin.constructorRef), position = fin.position)
for yr in 2021:2025
    f = "output/standings_$(yr).json"
    isfile(f) || Downloads.download("https://api.jolpi.ca/ergast/f1/$(yr)/constructorstandings.json", f)
    s = JSON3.read(read(f, String)).MRData.StandingsTable.StandingsLists[1]
    for c in s.ConstructorStandings
        push!(st, (yr, String(c.Constructor.constructorId), parse(Int, c.position)))
    end
end

ranks(x) = (o = sortperm(x); r = similar(x, Float64); for (i, j) in enumerate(o); r[j] = i; end;
            for v in unique(x); k = findall(==(v), x); r[k] .= mean(r[k]); end; r)
spearman(a, b) = cor(ranks(a), ranks(b))

function check(lbl, eff)        # eff: machine label "<id>_<season>", mean
    m = match.(r"^(.*)_(\d{4})$", eff.label)
    eff.id = [x[1] for x in m]; eff.season = [parse(Int, x[2]) for x in m]
    j = innerjoin(eff, st; on = [:season, :id])
    per = combine(groupby(j, :season), nrow => :cars, [:mean, :position] => spearman => :rho)
    per = per[per.cars .>= 5, :]
    @printf("\n== %s: %d seasons with ≥ 5 matched cars, median Spearman %.2f (car effect, negative = faster, against position)\n",
            lbl, nrow(per), median(per.rho))
    per.decade = 10 .* (per.season .÷ 10)
    for g in groupby(per, :decade)
        @printf("  %ds: %2d seasons, median %.2f, min %.2f (%d)\n", g.decade[1], nrow(g), median(g.rho), minimum(g.rho), g.season[argmin(g.rho)])
    end
    # cars whose effect rank in their season is furthest from their championship position
    j.eff_rank = zeros(nrow(j)); j.pos_rank = zeros(nrow(j))
    for g in groupby(j, :season); g.eff_rank .= ranks(g.mean); g.pos_rank .= ranks(Float64.(g.position)); end
    j.gap = j.pos_rank .- j.eff_rank
    recent = sort(j[j.season .>= 2011, :], :gap)
    println("  2011 on, faster by the model than the championship says (rank gap ≤ −3): ",
            join(["$(r.id) $(r.season) (P$(r.position), effect rank $(Int(r.eff_rank)))" for r in eachrow(recent) if r.gap <= -3], ", "))
    println("  2011 on, slower by the model than the championship says (rank gap ≥ 3): ",
            join(["$(r.id) $(r.season) (P$(r.position), effect rank $(Int(r.eff_rank)))" for r in eachrow(reverse(recent)) if r.gap >= 3], ", "))
    return per
end
rc = CSV.read(race_csv, DataFrame); rename!(rc, :id => :label)
pr = check("race model", rc)
lc = CSV.read("output/$(laps_tag)_cars.csv", DataFrame); rename!(lc, :machine => :label)
pl = check("lap model", lc)
both = innerjoin(rename(pr, :rho => :race), rename(pl, :rho => :laps); on = :season, makeunique = true)
@printf("\nseasons in both (%d): median Spearman race %.2f, laps %.2f; lap model closer to the championship in %d\n",
        nrow(both), median(both.race), median(both.laps), count(both.laps .> both.race))
