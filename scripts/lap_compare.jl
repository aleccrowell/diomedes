# Compare lap-model effects (#12).
#
#   julia --project scripts/lap_compare.jl <tagA> <tagB>
#       two lap-model fits, e.g. laps_2011-2019_recorded laps_2011-2019_inferred
#       (the test of inferred pit stops: do the effects agree?)
#   julia --project scripts/lap_compare.jl <tag> race [race_tag]
#       a lap-model fit against the race-time model's effects (default race_tag:
#       gap_all_pl_dur_rw_kappa_age_big_slope_dev_fgam) for the same drivers and
#       car-seasons
#
# LAP_SEASONS=<first>-<last> restricts the car-season comparison to those seasons.
# Reads output/<tag>_{drivers,cars}.csv (scripts/lap_fit.jl combine) and, for the
# race model, output/<race_tag>_{competitor,machine}_effects.csv.

using CSV, DataFrames, Statistics

# LAP_SEASONS=<first>-<last> keeps only car-seasons from those seasons (labels end in _<year>)
const SEASONS = let m = match(r"^(\d{4})-(\d{4})$", get(ENV, "LAP_SEASONS", ""))
    m === nothing ? nothing : parse(Int, m[1]):parse(Int, m[2])
end
in_seasons(label) = SEASONS === nothing || parse(Int, last(split(label, "_"))) in SEASONS
read2(tag) = (CSV.read("output/$(tag)_drivers.csv", DataFrame),
              filter(:machine => in_seasons, CSV.read("output/$(tag)_cars.csv", DataFrame)))

function report(name, x, sx, y, sy, labels; n_show = 8)
    println("\n$name: n = $(length(x)), correlation ", round(cor(x, y); digits = 3),
            ", Spearman ", round(cor(sortperm(sortperm(x)), sortperm(sortperm(y))); digits = 3),
            "; slope of B on A ", round(cov(x, y) / var(x); digits = 3))
    if sx !== nothing
        z = (y .- x) ./ sqrt.(sx .^ 2 .+ sy .^ 2)
        println("  |difference| / combined sd: mean ", round(mean(abs.(z)); digits = 2), ", max ",
                round(maximum(abs.(z)); digits = 2), ", share > 2: ", round(mean(abs.(z) .> 2); digits = 3))
        o = sortperm(abs.(z); rev = true)[1:min(n_show, length(z))]
        show(stdout, MIME"text/plain"(), DataFrame(level = labels[o], A = round.(x[o]; digits = 3),
                                                     B = round.(y[o]; digits = 3), z = round.(z[o]; digits = 2)))
        println()
    end
end

tagA = ARGS[1]
if length(ARGS) >= 2 && ARGS[2] != "race"
    (da, ca), (db, cb) = read2(tagA), read2(ARGS[2])
    println("A = $tagA, B = $(ARGS[2])")
    dj = innerjoin(da, db; on = :driver, makeunique = true)
    dj = dj[(dj.laps .>= 500) .& (dj.laps_1 .>= 500), :]
    report("drivers (≥ 500 laps)", dj.mean, dj.sd, dj.mean_1, dj.sd_1, dj.driver)
    cj = innerjoin(ca, cb; on = :machine, makeunique = true)
    cj = cj[(cj.laps .>= 500) .& (cj.laps_1 .>= 500), :]
    report("car-seasons (≥ 500 laps)", cj.mean, cj.sd, cj.mean_1, cj.sd_1, cj.machine)
else
    race_tag = get(ARGS, 3, "gap_all_pl_dur_rw_kappa_age_big_slope_dev_fgam")
    da, ca = read2(tagA)
    rd = CSV.read("output/$(race_tag)_competitor_effects.csv", DataFrame)
    rc = CSV.read("output/$(race_tag)_machine_effects.csv", DataFrame)
    println("A = $race_tag (race-time model, career effects), B = $tagA (lap model)")
    dj = innerjoin(rename(rd[:, [:label, :mean, :sd]], :label => :driver, :mean => :race, :sd => :race_sd),
                   da; on = :driver)
    dj = dj[dj.laps .>= 500, :]
    # race-model driver effects span whole careers; only the ranking and scale are comparable
    report("drivers (≥ 500 laps)", dj.race, nothing, dj.mean, nothing, dj.driver)
    cj = innerjoin(rename(rc[:, [:label, :mean, :sd]], :label => :machine, :mean => :race, :sd => :race_sd),
                   ca; on = :machine)
    cj = cj[cj.laps .>= 500, :]
    report("car-seasons (≥ 500 laps)", cj.race, cj.race_sd, cj.mean, cj.sd, cj.machine)
end
