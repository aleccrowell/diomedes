using Diomedes
using DataFrames
using JSON3
using CSV
using Random
using Statistics
using Test
using Serialization: serialize, deserialize
using Dates: Date, Day
using Turing: logjoint
using Distributions: Normal, cdf, ccdf, logpdf
using ADTypes: AutoForwardDiff, AutoReverseDiff
import SpecialFunctions
import Turing
import Distributions

const FIXTURES = joinpath(@__DIR__, "fixtures")

# A circuit table (#81) for the races of `g`, in the columns of `circuit_table()`:
# five layouts shared across rounds and seasons, conditions cycling dry, dry, mixed, wet.
function synthetic_circuits(g, rng)
    evs = unique(g.rows.event_id)
    rnd = [parse(Int, last(e, 2)) for e in evs]
    return DataFrame(event_id = evs, season = [parse(Int, first(e, 4)) for e in evs],
                     circuit_id = ["c$(r % 5)" for r in rnd], length_km = [4.0 + 0.3 * (r % 5) for r in rnd],
                     street = rand(rng, length(evs)), W = rand(rng, 0:2, length(evs)), S = rand(rng, 0:2, length(evs)),
                     temp_c = 15 .+ 15 .* rand(rng, length(evs)), condition = ["dry", "dry", "mixed", "wet"][mod1.(rnd, 4)])
end

# Simulated circuit-racing results for the retirement model (#13): per season,
# `n_car` cars with two drivers each race `n_race` races of `L` laps. Retirement
# hazards per race distance: mechanical exp(a[1] + car), incident exp(a[2] + driver),
# collision exp(a[3] + β_col·driver), plus first-lap jumps exp(b1). Returns the
# results table (and the true effects as a second value with `truth = true`).
function sim_retirements(rng; n_season = 3, n_race = 6, n_car = 6, L = 50, a = [-1.0, -2.0, -2.5],
                         b1 = [-3.0, -2.0, -1.5], σ_car = 0.6, σ_drv = 0.5, β_col = 0.5, truth = false)
    n_drv = 2n_car + 4
    drv = σ_drv .* randn(rng, n_drv)
    car = Dict{String,Float64}()
    rows = DataFrame()
    for s in 1:n_season
        seats = shuffle(rng, 1:n_drv)[1:2n_car]        # drivers change seats between seasons
        for c in 1:n_car
            car["c$(c)_$(2000 + s)"] = σ_car * randn(rng)
        end
        for r in 1:n_race, (j, dv) in enumerate(seats)
            c = cld(j, 2)
            η = [a[1] + car["c$(c)_$(2000 + s)"], a[2] + drv[dv], a[3] + β_col * drv[dv]]
            n, status = L, "Finished"
            for lap in 1:L
                h = exp.(η) .* (exp.(b1) .* (lap == 1) .+ 1 / L)
                if rand(rng) < 1 - exp(-sum(h))
                    k = findfirst(cumsum(h) ./ sum(h) .> rand(rng))
                    n, status = lap - 1, ("Engine", "Accident", "Collision")[k]
                    break
                end
            end
            push!(rows, (series = "sim", season = 2000 + s, event_id = "$(2000 + s)-$r", stage_id = "race",
                         competitor_id = "d$dv", machine_id = "c$c", laps = n, status = status))
        end
    end
    return truth ? (rows, (; drv, car)) : rows
end

# log(Φ(b) - Φ(a)) at high precision, using whichever tail keeps the difference representable.
refdiff(a, b) = setprecision(BigFloat, 2048) do
    Float64(a > 0 ? log(ccdf(Normal(), big(a)) - ccdf(Normal(), big(b))) :
                    log(cdf(Normal(), big(b)) - cdf(Normal(), big(a))))
end

# Opt-in test groups (set the variable to "1"):
#   DIOMEDES_SLOW_TESTS     - model fitting with NUTS (minutes on small hardware)
#   DIOMEDES_NETWORK_TESTS  - hit the live APIs
opt_in(name) = get(ENV, name, "") == "1"

# Build a Jolpica cache directory holding one synthetic page for the 2019 season
# (rounds 1-2 only), so `JolpicaF1` can be exercised end-to-end without network.
function jolpica_fixture_cache()
    dir = mktempdir()
    races = [JSON3.read(read(joinpath(FIXTURES, "jolpica", "2019_$r.json"), String)).MRData.RaceTable.Races[1]
             for r in 1:2]
    total = sum(length(r.Results) for r in races)
    page = Dict("MRData" => Dict("total" => string(total), "RaceTable" => Dict("Races" => races)))
    src = JolpicaF1(; cache_dir = dir)
    url = "$(src.base_url)/2019/results.json?limit=$(Diomedes.JOLPICA_PAGE)&offset=0"
    write(Diomedes.cache_path(dir, url), JSON3.write(page))
    return src
end

@testset "Diomedes" begin
    @testset "schema" begin
        @test validate_results(empty_results()) isa DataFrame
        @test_throws ArgumentError validate_results(select(empty_results(), Not(:machine_id)))
    end

    @testset "Jolpica rows: only lead-lap finishers get a race time (#55)" begin
        # from 2023 Jolpica gives lapped cars their own (shorter-distance) time
        race = JSON3.read("""{"season": "2024", "round": "8", "raceName": "Monaco Grand Prix", "date": "2024-05-26",
            "Results": [
              {"number": "16", "position": "1", "positionText": "1", "points": "25", "grid": "1", "laps": "78",
               "status": "Finished", "Time": {"millis": "8595548", "time": "2:23:15.554"},
               "Driver": {"driverId": "leclerc", "givenName": "Charles", "familyName": "Leclerc", "dateOfBirth": "1997-10-16"},
               "Constructor": {"constructorId": "ferrari"}},
              {"number": "14", "position": "11", "positionText": "11", "points": "0", "grid": "11", "laps": "76",
               "status": "Lapped", "Time": {"millis": "8599412", "time": "+2 Laps"},
               "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso", "dateOfBirth": "1981-07-29"},
               "Constructor": {"constructorId": "aston_martin"}},
              {"number": "4", "position": "19", "positionText": "R", "points": "0", "grid": "4", "laps": "64",
               "status": "Retired", "Time": {"millis": "7300000", "time": "2:01:40.000"},
               "Driver": {"driverId": "norris", "givenName": "Lando", "familyName": "Norris", "dateOfBirth": "1999-11-13"},
               "Constructor": {"constructorId": "mclaren"}}]}""")
        rows = [Diomedes.jolpica_row(race, r) for r in race.Results]
        @test rows[1].time_ms == 8595548.0
        @test ismissing(rows[2].time_ms) && rows[2].laps == 76 && rows[2].status == "Lapped"
        # a retired car's time when it stopped is its own elapsed time, not a race time
        # (as a race time it would beat the winner's)
        @test ismissing(rows[3].time_ms) && rows[3].own_time_ms == 7300000.0 && rows[3].laps == 64
        @test rows[2].own_time_ms == 8599412.0 && ismissing(rows[1].own_time_ms)
        # hours dropped from lap time strings (Bahrain 2020: a 1:22:30 lap read as 22:30)
        laps = [90_000.0, 22.5 * 60_000, 90_000.0]
        @test Diomedes.lap_hours_lost(sum(laps) + 3_600_000, laps) == 3_600_000
        @test Diomedes.lap_hours_lost(sum(laps) + 1_500, laps) == 0 && Diomedes.lap_hours_lost(sum(laps) - 9_000, laps) == 0
    end

    @testset "standardise_times!" begin
        df = DataFrame(series = "s", event_id = ["a", "a", "a", "b", "b"], stage_id = "x",
                       time_ms = [1.0, 2.0, missing, 5.0, 5.0])
        standardise_times!(df)
        @test df.z[1:2] ≈ [-1.0, 1.0]           # population sd, as sklearn.scale
        @test ismissing(df.z[3])
        @test df.z[4:5] == [0.0, 0.0]            # zero-variance stage
    end

    @testset "sources agree: ErgastCSV vs JolpicaF1" begin
        csv = fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019)
        api = fetch_results(jolpica_fixture_cache(), 2019)
        @test nrow(csv) == nrow(api) == 40
        key = [:event_id, :competitor_id]
        j = innerjoin(csv, api; on = key, makeunique = true)
        @test nrow(j) == 40
        @test j.machine_id == j.machine_id_1
        @test isequal(j.time_ms, j.time_ms_1)
        @test isequal(j.position, j.position_1)
        @test isequal(j.laps, j.laps_1) && !any(ismissing, j.laps)
        @test j.classified == j.classified_1
        @test j.status == j.status_1
        @test all(==(0.0), api.suspended_ms)          # fastest laps known, no suspect race
        @test isequal(j.event_date, j.event_date_1) && !any(ismissing, j.event_date)
        @test isequal(j.competitor_birth, j.competitor_birth_1) && !any(ismissing, j.competitor_birth)
    end

    @testset "Indy 500 excluded by default" begin
        dir = mktempdir()
        for f in readdir(joinpath(FIXTURES, "ergast"))
            cp(joinpath(FIXTURES, "ergast", f), joinpath(dir, f))
        end
        races = read(joinpath(dir, "races.csv"), String)
        write(joinpath(dir, "races.csv"), replace(races, "Bahrain Grand Prix" => "Indianapolis 500"))
        @test Set(fetch_results(ErgastCSV(dir), 2019).event_id) == Set(["2019-01"])
        @test Set(fetch_results(ErgastCSV(dir; include_indy500 = true), 2019).event_id) ==
              Set(["2019-01", "2019-02"])
    end

    @testset "WRC vehicle model normalisation" begin
        nm = Diomedes.normalise_model
        @test nm(" i20 Coupé  WRC ") == nm("I20 COUPE WRC") == "I20 COUPE WRC"
        @test nm("Puma Rally 1") == "PUMA RALLY1"
        @test nm("Fiesta Rally 3") == nm("FIESTA RALLY3")
        @test nm("Rally 12x") == "RALLY 12X"  # only a single-digit class number is joined
        # top two tiers, predecessor class names included
        @test wrc_tier("WRC") == wrc_tier("Rally1") == "Rally1" && wrc_tier("R5") == wrc_tier("Rally2") == "Rally2"
        @test ismissing(wrc_tier("Rally3")) && ismissing(wrc_tier(missing))
        df = DataFrame(season = [2021, 2022, 2022, 2022, 2019], class = ["WRC", "WRC", "Rally1", "Rally3", "R5"],
                       manufacturer = ["Ford", "Ford", "Toyota", "Ford", "Skoda"])
        top = wrc_top_tiers(df)
        @test top.class == ["WRC", "Rally1", "R5"]
        @test wrc_machine_key.(eachrow(top)) == ["Ford Rally1_2021", "Toyota Rally1_2022", "Skoda Rally2_2019"]
    end
    @testset "red-flag suspensions" begin
        @test Diomedes.laptime_ms("1:16.956") ≈ 76_956
        @test Diomedes.laptime_ms("16.956") ≈ 16_956
        @test Diomedes.laptime_ms("2:03:10.5") ≈ 7_390_500
        @test Diomedes.suspect_suspension(14_679_537, 70, 76_956)         # Canada 2011
        @test !Diomedes.suspect_suspension(5_400_000, 58, 85_000)        # a normal race
        @test !Diomedes.suspect_suspension(missing, 58, 85_000)
        cars = [fill(90_000.0 + 500c, 60) for c in 1:20]                  # 20 cars × 60 laps
        @test Diomedes.suspension_ms(cars) == 0
        for c in 1:20
            cars[c][30] *= 1.5                                            # safety-car lap: not flagged
            cars[c][25 - c % 4] += 120 * 60_000                           # stoppage on laps 22–25:
        end                                                               # field spread over lap numbers
        cars[3][40] += 8 * 60_000                                         # one car's long repair stop
        @test Diomedes.suspension_ms(cars) ≈ 120 * 60_000 rtol = 0.01
        cars2 = deepcopy(cars); foreach(c -> c[50] += 20 * 60_000, cars2) # a second stoppage
        @test Diomedes.suspension_ms(cars2) ≈ 140 * 60_000 rtol = 0.01

        # Ergast dump with a 2-hour stoppage in race 1: official times include it (as
        # since 2005) and lap times show it; the correction must recover the original gaps
        dir = mktempdir()
        for f in readdir(joinpath(FIXTURES, "ergast"))
            cp(joinpath(FIXTURES, "ergast", f), joinpath(dir, f))
        end
        base = fetch_results(ErgastCSV(dir), 2019)
        @test all(ismissing, base.suspended_ms)                 # no lap_times.csv
        stop = 120 * 60_000
        rc = CSV.read(joinpath(dir, "results.csv"), DataFrame; missingstring = "\\N")
        rc.milliseconds = [r.raceId == 1010 && !ismissing(r.milliseconds) ? r.milliseconds + stop :
                           r.milliseconds for r in eachrow(rc)]
        CSV.write(joinpath(dir, "results.csv"), rc; missingstring = "\\N")
        inflated = fetch_results(ErgastCSV(dir), 2019)
        open(joinpath(dir, "lap_times.csv"), "w") do io
            println(io, "raceId,driverId,lap,position,time,milliseconds")
            for d in 1:20, l in 1:58               # stoppage on laps 23–25: field spread over lap numbers
                println(io, "1010,$d,$l,$d,\"x\",$(85_000 + 300d + (l == 25 - d % 3 ? stop : 0))")
            end
        end
        res = fetch_results(ErgastCSV(dir), 2019)
        r1 = res.event_id .== "2019-01"
        @test all(res.suspended_ms[r1] .≈ stop)
        @test all(ismissing, res.suspended_ms[.!r1])            # race 2 has no lap times
        gb, gi, gc = prepare_gaps(base), prepare_gaps(inflated), prepare_gaps(res)
        @test gc.y ≈ gb.y                                       # correction recovers the true gaps
        @test gc.race_minutes ≈ gb.race_minutes
        k = gi.rows.event_id[gi.rows.kind .== :timed] .== "2019-01"
        @test all(gi.y[k] .<= gb.y[k]) && any(gi.y[k] .< gb.y[k])   # uncorrected: gaps shrunk
        @test gi.lo ≈ gb.lo && gi.hi ≈ gb.hi                    # lapped intervals unaffected
        @test isequal(prepare(inflated).y, prepare(base).y) || prepare(inflated).y ≈ prepare(base).y   # z-scores: shift cancels
    end

    @testset "prepare" begin
        res = fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019)
        d = prepare(res)
        @test length(d) == count(!ismissing, res.time_ms)
        @test issorted(d.competitors)
        @test d.competitors[d.competitor] == d.rows.competitor_id
        @test all(endswith("_2019"), d.machines)
    end

    @testset "pair-statistics likelihood matches per-observation likelihood" begin
        rng = Xoshiro(3)
        n_comp, n_mach, n = 12, 7, 300
        comp, mach = rand(rng, 1:n_comp, n), rand(rng, 1:n_mach, n)
        y = randn(rng, n)
        rows = DataFrame(competitor_id = string.(comp), competitor_name = string.(comp))
        d = Diomedes.ModelData(y, comp, mach, string.(1:n_comp), string.(1:n_mach), rows)
        ps = Diomedes.PairStats(d)
        @test sum(ps.n) == n && length(ps.n) < n
        for σ_y in (1.0, nothing), intercept in (false, true), _ in 1:3
            θ = (; σ_comp = rand(rng) + 0.1, σ_mach = rand(rng) + 0.1,
                 z_comp = randn(rng, n_comp), z_mach = randn(rng, n_mach))
            σ_y === nothing && (θ = (; θ..., σ_y = rand(rng) + 0.2))
            intercept && (θ = (; θ..., α = randn(rng)))
            @test logjoint(crossed_effects(d; σ_y, intercept), θ) ≈
                  logjoint(Diomedes.crossed_effects_obs(d; σ_y, intercept), θ)
        end
    end

    @testset "prepare_gaps" begin
        res = fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019)
        g = prepare_gaps(res)
        timed = g.rows[g.rows.kind .== :timed, :]
        lapped = g.rows[g.rows.kind .== :lapped, :]
        @test nrow(timed) == count(!ismissing, res.time_ms)
        @test sort(timed[timed.y .== 0, :competitor_id]) == ["bottas", "hamilton"]  # the two winners
        @test all(>=(0), g.y)
        # every untimed car with a lapped status is an interval; retirements are not,
        # even when they carry a numeric position (as in Ergast)
        @test nrow(lapped) == count(r -> ismissing(r.time_ms) && Diomedes.is_lapped_status(r.status), eachrow(res))
        @test all(Diomedes.is_lapped_status, lapped.status)
        @test Diomedes.laps_down_agrees("+2 Laps", 2) && !Diomedes.laps_down_agrees("+2 Laps", 3) &&
              Diomedes.laps_down_agrees("Lapped", 5)
        @test Diomedes.is_lapped_status("+1 Lap") && Diomedes.is_lapped_status("+12 Laps") &&
              Diomedes.is_lapped_status("Lapped")
        @test !Diomedes.is_lapped_status("Suspension") && !Diomedes.is_lapped_status("Finished") &&
              !Diomedes.is_lapped_status("+1 Lapse")
        retired = copy(res)
        k = findfirst(r -> ismissing(r.time_ms) && Diomedes.is_lapped_status(r.status), eachrow(retired))
        retired.status[k] = "Suspension"; retired.classified[k] = true   # retirement with a position
        @test nrow(prepare_gaps(retired).rows) == nrow(g.rows) - 1
        # a timed car with fewer laps (a retirement timed when it stopped) is not the
        # winner, however fast its time
        early = copy(res); ev = first(early.event_id)
        j = findfirst(k -> early.event_id[k] == ev && ismissing(early.time_ms[k]) && !ismissing(early.laps[k]), eachindex(early.event_id))
        early.time_ms[j] = 1000.0
        ge = prepare_gaps(early)
        @test ge.rows.winner_ms[findfirst(==(ev), ge.rows.event_id)] == g.rows.winner_ms[findfirst(==(ev), g.rows.event_id)]
        # a race whose winner completed fewer than 3 laps (2021 Belgian GP) is dropped
        short = copy(res); ev = first(short.event_id)
        short.laps[short.event_id .== ev] .= 1
        @test length(prepare_gaps(short).races) == length(g.races) - 1
        r = first(lapped[lapped.laps .== 57, :])       # Australia 2019 (58 laps), 1 lap down
        @test r.lo ≈ 100 * log(58 / 57) && r.hi ≈ 100 * log(58 / 56)
        @test all(g.lo .< g.hi)
        @test nrow(prepare_gaps(res; include_lapped = false).rows) == nrow(timed)
        # ages: Hamilton, born 1985-01-07, Australia 2019-03-17 -> 34
        h = findfirst(r -> r.competitor_id == "hamilton" && r.event_id == "2019-01", eachrow(g.rows))
        @test g.age_years[g.rows.age_bin[h]] == 34
        @test sum(g.age_rows) == nrow(g.rows) && all(>(0), vcat(g.t_age, g.c_age))
        # prior scale (#30): MAD of all rows about their race median, lapped rows at interval midpoints
        ps = PriorScale(g)
        yall, rall = vcat(g.y, (g.lo .+ g.hi) ./ 2), vcat(g.t_race, g.c_race)
        med = [median(yall[rall .== r]) for r in eachindex(g.races)]
        @test ps.s ≈ median(abs.(yall .- med[rall])) && ps.m ≈ median(med)
        # it scales with the data's units: every gap × 3 gives s × 3 and m × 3
        g3 = GapData(3 .* g.y, g.t_comp, g.t_mach, g.t_race, 3 .* g.lo, 3 .* g.hi, g.c_comp, g.c_mach, g.c_race,
                     g.mach_season, g.race_season, g.race_minutes, g.t_age, g.c_age, g.age_years, g.age_rows,
                     g.competitors, g.machines, g.races, g.rows)
        @test PriorScale(g3).s ≈ 3ps.s && PriorScale(g3).m ≈ 3ps.m
        # the career curve has no constant or linear part (weighted by rows per bin)
        B = AgeCurveBasis(g)
        f = age_curve(0.3, randn(Xoshiro(2), length(g.age_years) - 1), B)
        @test abs(sum(g.age_rows .* f)) < 1e-9
        @test abs(sum(g.age_rows .* g.age_years .* f)) < 1e-7
        # career rows: each driver's centred ages average zero over their rows
        cr = CareerRows(g)
        comps = vcat(g.t_comp, g.c_comp)
        @test length(cr.agec) == length(g.y) + length(g.lo) && all(cr.known .== 1)
        @test all(abs(sum(cr.agec[comps .== d])) < 1e-9 for d in unique(comps))
        @test driver_offsets(cr, f, nothing) ≈ f[vcat(g.t_age, g.c_age)]
        s = career_slopes(0.4, randn(Xoshiro(3), length(g.competitors)))
        @test abs(sum(s)) < 1e-12
        @test driver_offsets(cr, f, s) ≈ f[vcat(g.t_age, g.c_age)] .+ s[comps] .* cr.agec
    end

    @testset "logdiffΦ" begin
        for (a, b) in ((-1.0, 1.0), (2.0, 3.0), (8.0, 8.5), (-9.0, -8.0), (30.0, 31.0), (-0.5, 40.0),
                       (-31.0, -30.0), (0.0, 1e-3))
            @test Diomedes.logdiffΦ(a, b) ≈ refdiff(a, b) rtol = 1e-8
        end
        # Student-t interval probabilities against the plain CDF difference in BigFloat
        for ν in (3.0, 4.0, 8.0), (a, b) in ((-1.0, 1.0), (2.0, 3.0), (-9.0, -8.0), (20.0, 21.0), (0.0, 1e-3))
            d = Distributions.TDist(ν)
            ref = setprecision(BigFloat, 256) do
                Float64(log(big(Distributions.ccdf(d, a)) - big(Distributions.ccdf(d, b))))
            end
            @test Diomedes.logdiffcdf_std(StudentTNoise(ν), a, b) ≈ ref rtol = 1e-6
        end
        for ν in (3.0, 4.0, 8.0), x in (-3.0, 0.0, 0.7, 12.0)
            @test Diomedes.logpdf_std(StudentTNoise(ν), x) ≈ Distributions.logpdf(Distributions.TDist(ν), x)
        end
    end

    @testset "pace + loss noise" begin
        # EMG log kernel: matches a 512-bit reference, and stays exact when λ ≪ σ
        # (the naive sum of three huge terms cancelled catastrophically there)
        ref(x, σ, λ) = setprecision(512) do
            X, S, L = big(x), big(σ), big(λ)
            Float64(S^2 / (2L^2) - X / L + log(SpecialFunctions.erfc(-(X / S - S / L) / sqrt(big(2))) / 2))
        end
        for (x, σ, λ) in ((0.3, 0.5, 2.0), (-1.0, 0.5, 2.0), (8.0, 0.4, 3.0), (0.2, 0.4, 0.01), (-2.0, 0.4, 1e-6))
            @test Diomedes.emg_logkernel(x, σ, λ) ≈ ref(x, σ, λ) rtol = 1e-10
        end
        # as λ → 0 the loss vanishes and the density is Gaussian; gradients stay finite
        for x in (-1.0, 0.0, 0.7, 13.0)
            @test Diomedes.paceloss_logpdf(x, 0.36, 0.2, 1e-11) ≈ logpdf(Normal(0, 0.36), x) rtol = 1e-8
            v, p = Diomedes.value_gradN((s, l) -> Diomedes.pacebig_loginterval(x, x + 1.5, s, 0.2, l, 0.01, 60.0), 0.36, 1e-11)
            @test isfinite(v) && all(isfinite, p)
        end
        # composite Simpson on [a, b] with n (even) intervals
        simpson(f, a, b, n) = (h = (b - a) / n; h / 3 * (f(a) + f(b) +
            4 * sum(f(a + (2k - 1) * h) for k in 1:(n ÷ 2)) + 2 * sum(f(a + 2k * h) for k in 1:(n ÷ 2 - 1))))
        for (σ, π, λ) in ((0.3, 0.2, 2.0), (0.5, 0.05, 0.8), (0.2, 0.5, 5.0))
            pdf(x) = exp(Diomedes.paceloss_logpdf(x, σ, π, λ))
            @test simpson(pdf, -10σ, 60λ, 200_000) ≈ 1 rtol = 1e-6
            # interval probabilities against numerical integration of the density:
            # central, straddling 0, right tail (lapped-car territory), far tail
            for (lo, hi) in ((-0.2, 0.1), (-1.0, 1.0), (1.5, 3.4), (8.0, 10.0), (30.0, 32.0),
                             (-2.5 * σ, -1.5 * σ), (-6σ, -5σ))      # left tail too
                ref = log(simpson(pdf, lo, hi, 20_000))
                @test Diomedes.paceloss_loginterval(lo, hi, σ, π, λ) ≈ ref rtol = 1e-6
            end
        end
        # extreme right tail: finite and ordered where the density underflows
        l1 = Diomedes.paceloss_loginterval(300.0, 305.0, 0.3, 0.2, 2.0)
        l2 = Diomedes.paceloss_loginterval(600.0, 605.0, 0.3, 0.2, 2.0)
        @test isfinite(l1) && isfinite(l2) && l2 < l1
        @test l1 ≈ log(0.2) - 300 / 2.0 + 0.3^2 / (2 * 2.0^2) + log1p(-exp(-5 / 2.0)) rtol = 1e-8
        # far left tail (where the EMG part cancels): finite, never a DomainError, and
        # close to the Gaussian component alone
        for (lo, hi) in ((-30.0, -29.0), (-12.0, -11.9), (-100.0, -99.0), (-9.0, -3.0))
            v = Diomedes.paceloss_loginterval(lo, hi, 0.3, 0.2, 2.0)
            @test isfinite(v)
            @test v ≈ log(0.8) + Diomedes.logdiffΦ(lo / 0.3, hi / 0.3) atol = 0.05
        end
    end

    @testset "two-component incident loss (#16)" begin
        simpson(f, a, b, n) = (h = (b - a) / n; h / 3 * (f(a) + f(b) +
            4 * sum(f(a + (2k - 1) * h) for k in 1:(n ÷ 2)) + 2 * sum(f(a + 2k * h) for k in 1:(n ÷ 2 - 1))))
        for (σ, π, λ, ρ, λ2) in ((0.3, 0.2, 2.0, 0.05, 30.0), (0.5, 0.4, 1.0, 0.2, 8.0))
            pdf(x) = exp(Diomedes.pacebig_logpdf(x, σ, π, λ, ρ, λ2))
            @test simpson(pdf, -10σ, 60λ2, 400_000) ≈ 1 rtol = 1e-5
            for (lo, hi) in ((-1.0, 1.0), (1.5, 3.4), (8.0, 10.0), (100.0, 104.0), (-3σ, -2σ))
                @test Diomedes.pacebig_loginterval(lo, hi, σ, π, λ, ρ, λ2) ≈ log(simpson(pdf, lo, hi, 20_000)) rtol = 1e-6
            end
        end
        # ρ → 0 recovers the single-component model
        @test Diomedes.pacebig_logpdf(2.0, 0.3, 0.2, 2.0, 1e-14, 30.0) ≈ Diomedes.paceloss_logpdf(2.0, 0.3, 0.2, 2.0)
        @test Diomedes.pacebig_loginterval(1.5, 3.4, 0.3, 0.2, 2.0, 1e-14, 30.0) ≈
              Diomedes.paceloss_loginterval(1.5, 3.4, 0.3, 0.2, 2.0)
        # a result many laps down is far more likely under the big-loss component
        @test Diomedes.pacebig_loginterval(300.0, 305.0, 0.5, 0.2, 2.0, 0.05, 40.0) >
              Diomedes.paceloss_loginterval(300.0, 305.0, 0.5, 0.2, 2.0) + 50
    end

    @testset "gap_effects log density and gradient" begin
        rng = Xoshiro(5)
        res = fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019)
        g = prepare_gaps(res)
        nc, nm, nr = length(g.competitors), length(g.machines), length(g.races)
        θ = (; σ_comp = 0.7, σ_mach = 1.1, σ_y = 0.8, z_comp = randn(rng, nc),
             z_mach = randn(rng, nm), γ = randn(rng, nr))
        # reference: plain loops with Distributions
        a = θ.σ_comp .* (θ.z_comp .- mean(θ.z_comp))
        b = θ.σ_mach .* (θ.z_mach .- mean(θ.z_mach))     # one season in the fixture
        ps = PriorScale(g)                              # priors in the dataset's gap units (#30)
        H = Turing.truncated(Normal(0, 2ps.s); lower = 0)
        lp = logpdf(H, θ.σ_comp) + logpdf(H, θ.σ_mach) + logpdf(H, θ.σ_y) +
             sum(logpdf.(Normal(), θ.z_comp)) + sum(logpdf.(Normal(), θ.z_mach)) +
             sum(logpdf.(Normal(ps.m, 4ps.s), θ.γ))
        # race intercepts are relative to the field mean of a[comp] + b[mach] over the race's rows
        races = vcat(g.t_race, g.c_race)
        cs = vcat(a[g.t_comp] .+ b[g.t_mach], a[g.c_comp] .+ b[g.c_mach])
        cbar = [mean(cs[races .== r]) for r in 1:nr]
        for i in eachindex(g.y)
            r = g.t_race[i]
            lp += logpdf(Normal(θ.γ[r] + a[g.t_comp[i]] + b[g.t_mach[i]] - cbar[r], θ.σ_y), g.y[i])
        end
        for i in eachindex(g.lo)
            r = g.c_race[i]
            μ = θ.γ[r] + a[g.c_comp[i]] + b[g.c_mach[i]] - cbar[r]
            lp += refdiff((g.lo[i] - μ) / θ.σ_y, (g.hi[i] - μ) / θ.σ_y)
        end
        model = gap_effects(g; noise = NormalNoise())   # reference below is Gaussian
        @test logjoint(model, θ) ≈ lp
        # The sampler's gradient (uncompiled ReverseDiff through the hand-written rule
        # for gap_loglik) matches ForwardDiff through the plain primal, at several points.
        LDF = Turing.DynamicPPL.LogDensityFunction
        LDP = Turing.DynamicPPL.LogDensityProblems
        rd = LDF(model; adtype = Diomedes.gap_sampler().adtype)
        fd = LDF(model; adtype = AutoForwardDiff())
        for _ in 1:5
            # LogDensityFunction evaluates in constrained space here: σ's (first 3) must be positive
            x = [0.3 .+ rand(rng, 3); 0.5 .* randn(rng, LDP.dimension(rd) - 3)]
            @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
        end
        # Student-t: the rule against central finite differences of the primal
        # (ForwardDiff cannot differentiate the t CDF)
        sc = Diomedes.season_counts(g)
        for ν in (3.0, 4.0, 8.0)
            noise = StudentTNoise(ν)
            p0 = (randn(rng, nr), randn(rng, nc), 0.7, randn(rng, nm), 1.1, 0.9)
            f(p) = Diomedes.gap_loglik(p..., g, sc, noise)
            v, pb = Diomedes.ChainRulesCore.rrule(Diomedes.gap_loglik, p0..., g, sc, noise)
            @test v ≈ f(p0)
            grads = pb(1.0)[2:7]
            for k in 1:6, j in (p0[k] isa Real ? (1:1) : (1:min(3, length(p0[k]))))
                h = 1e-6
                bump(δ) = ntuple(i -> i != k ? p0[i] : p0[i] isa Real ? p0[i] + δ :
                                     (x = copy(p0[i]); x[j] += δ; x), 6)
                fdiff = (f(bump(h)) - f(bump(-h))) / 2h
                @test grads[k][j] ≈ fdiff rtol = 1e-5 atol = 1e-6
            end
        end
        # a compiled tape would replay stale gradients from the custom rule, so it is refused
        @test_throws ArgumentError fit_gaps(g; sampler = Diomedes.default_sampler(), n_samples = 10)
    end

    @testset "paceloss_effects log density and gradient" begin
        rng = Xoshiro(21)
        # two seasons (2019 and a relabelled copy as 2020), so the random walk has a
        # step and car effects are centred within each season
        res = fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019)
        res2 = copy(res); res2.season .= 2020; res2.event_id .= replace.(res2.event_id, "2019" => "2020")
        g = prepare_gaps(vcat(res, res2))
        @test LossCovariates(g).n_seasons == 2
        nc, nm, nr = length(g.competitors), length(g.machines), length(g.races)
        cov = LossCovariates(g)
        LDF = Turing.DynamicPPL.LogDensityFunction
        LDP = Turing.DynamicPPL.LogDensityProblems
        H(s) = Turing.truncated(Normal(0, s); lower = 0)
        combos = [[(d, e, false, false, false, false, false) for d in (false, true) for e in (:none, :decade, :regime, :rw)];
                  (true, :rw, true, false, false, false, false); (false, :none, true, false, false, false, false);
                  (true, :rw, true, true, false, false, false); (true, :rw, true, false, true, false, false);
                  (true, :rw, true, true, true, false, false); (true, :rw, true, false, false, true, false);
                  (true, :rw, true, true, true, true, false); (true, :rw, true, false, false, false, true);
                  (true, :rw, true, true, true, true, true)]
        # centred driver effects (#28): with and without the pace scale, and with every term on
        # and Student-t(3) drivers, centred and not (tdrv)
        cases = [[(c, false, false) for c in combos]; ((true, :rw, false, false, false, false, false), true, false);
                 ((true, :rw, true, false, false, false, false), true, false); ((true, :rw, true, true, true, true, true), true, false);
                 ((true, :rw, true, false, false, false, false), true, true); ((true, :rw, true, false, false, false, false), false, true);
                 ((true, :rw, true, true, true, true, true), true, true)]
        # hierarchical race intercepts (#32): with every term on, and on the plain pace + loss model
        cases = [[(c..., false) for c in cases]; ((true, :rw, true, true, true, true, true), false, false, true);
                 ((false, :none, false, false, false, false, false), false, false, true)]
        # learned common race level only (race_mean, #32)
        cases = [[(c..., false) for c in cases]; ((true, :rw, true, true, true, true, true), false, false, false, true)]
        # circuit slopes and track terms (#81): each alone, both with every term on, track on the plain model
        ctab = synthetic_circuits(g, rng)
        rc = RaceCircuits(g, ctab)
        cases = [[(c..., false, false) for c in cases];
                 ((true, :rw, true, false, false, false, false), false, false, false, true, true, false);
                 ((true, :rw, true, false, false, false, false), false, false, false, true, false, true);
                 ((true, :rw, true, true, true, true, true), false, false, false, true, true, true);
                 ((false, :none, false, false, false, false, false), false, false, false, true, false, true)]
        for ((dur, era, kap, age, big, slp, dev), cdrv, tdrv, hg, mg, circ, trk) in cases
            θ = (; σ_comp = 0.7, σ_mach = 1.1, σ = 0.4, z_comp = randn(rng, nc), z_mach = randn(rng, nm),
                 γ = randn(rng, nr), a_π = -1.2, a_λ = 0.6)
            dur && (θ = (; θ..., β_dur_π = 0.25, β_dur_λ = 0.3))
            k = era === :decade ? cov.n_decades : cov.n_regimes
            era in (:decade, :regime) && (θ = (; θ..., τ_π_era = 0.3, τ_λ_era = 0.2,
                                               z_π_era = randn(rng, k), z_λ_era = randn(rng, k)))
            era === :rw && (θ = (; θ..., τ_π_rw = 0.1, τ_λ_rw = 0.2, e_π_rw = randn(rng, cov.n_seasons - 1),
                                 e_λ_rw = randn(rng, cov.n_seasons - 1)))
            trk && (θ = (; θ..., β_track_π = randn(rng, 3), β_track_λ = randn(rng, 3)))
            # pace scale (#15): no z_mach; τ_κ, e_κ and centred b_mach come last, in sampling order
            kap && (θ = (; (k => v for (k, v) in pairs(θ) if k !== :z_mach)...,
                         τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm)))
            age && (θ = (; θ..., τ_age = 0.2, e_age = randn(rng, length(g.age_years) - 1)))
            big && (θ = (; θ..., a_ρ = -2.5, δ_λ2 = 2.0))
            slp && (θ = (; θ..., τ_slope = 0.3, u_slope = randn(rng, nc)))
            dev && (θ = (; θ..., τ_dev = 0.4, u_dev = randn(rng, nm)))
            circ && (θ = (; θ..., τ_cx = [0.3, 0.5, 0.2], u_cx = randn(rng, 3nm)))
            # centred drivers: a_comp (in % units) replaces z_comp, in the same sampling position
            cdrv && (θ = NamedTuple(k === :z_comp ? (:a_comp => θ.σ_comp .* v) : (k => v) for (k, v) in pairs(θ)))
            # hierarchical race intercepts: μ_γ and τ_γ sampled just before γ
            hg && (θ = NamedTuple(Iterators.flatten(k === :γ ? ((:μ_γ => 1.8), (:τ_γ => 1.3), (k => v)) : ((k => v),)
                                                    for (k, v) in pairs(θ))))
            mg && (θ = NamedTuple(Iterators.flatten(k === :γ ? ((:μ_γ => 1.8), (k => v)) : ((k => v),) for (k, v) in pairs(θ))))
            # reference: priors from Distributions, per-race loss written out by hand,
            # per-row terms in plain loops
            ps = PriorScale(g)                          # priors in the dataset's gap units (#30)
            lp = logpdf(H(2ps.s), θ.σ_comp) + logpdf(H(2ps.s), θ.σ_mach) + logpdf(H(ps.s), θ.σ) +
                 (cdrv ? sum(logpdf.(tdrv ? θ.σ_comp * Distributions.TDist(3) : Normal(0, θ.σ_comp), θ.a_comp)) :
                         sum(logpdf.(tdrv ? Distributions.TDist(3) : Normal(), θ.z_comp))) +
                 (kap ? 0.0 : sum(logpdf.(Normal(), θ.z_mach))) +
                 (hg ? logpdf(Normal(ps.m, 4ps.s), θ.μ_γ) + logpdf(H(4ps.s), θ.τ_γ) + sum(logpdf.(Normal(θ.μ_γ, θ.τ_γ), θ.γ)) :
                  mg ? logpdf(Normal(ps.m, 4ps.s), θ.μ_γ) + sum(logpdf.(Normal(θ.μ_γ, 4ps.s), θ.γ)) :
                       sum(logpdf.(Normal(ps.m, 4ps.s), θ.γ))) + logpdf(Normal(-1.5, 1), θ.a_π) +
                 logpdf(Normal(log(2ps.s), 1), θ.a_λ)
            logitπ, logλ = fill(θ.a_π, nr), fill(θ.a_λ, nr)
            if dur
                lp += logpdf(Normal(0, 1), θ.β_dur_π) + logpdf(Normal(0, 1), θ.β_dur_λ)
                logitπ .+= θ.β_dur_π .* cov.log_duration
                logλ .+= θ.β_dur_λ .* cov.log_duration
            end
            if era in (:decade, :regime)
                idx = era === :decade ? cov.decade : cov.regime
                lp += logpdf(H(0.5), θ.τ_π_era) + logpdf(H(0.5), θ.τ_λ_era) +
                      sum(logpdf.(Normal(), θ.z_π_era)) + sum(logpdf.(Normal(), θ.z_λ_era))
                logitπ .+= θ.τ_π_era .* (θ.z_π_era .- mean(θ.z_π_era))[idx]
                logλ .+= θ.τ_λ_era .* (θ.z_λ_era .- mean(θ.z_λ_era))[idx]
            elseif era === :rw
                lp += logpdf(H(0.25), θ.τ_π_rw) + logpdf(H(0.25), θ.τ_λ_rw) +
                      sum(logpdf.(Distributions.TDist(3), θ.e_π_rw)) + sum(logpdf.(Distributions.TDist(3), θ.e_λ_rw))
                path(steps) = (p = [0.0; cumsum(steps)]; p .- mean(p))
                logitπ .+= path(θ.τ_π_rw .* θ.e_π_rw)[cov.season]
                logλ .+= path(θ.τ_λ_rw .* θ.e_λ_rw)[cov.season]
            end
            if trk
                # tight (W/2), mixed and wet per race: slopes on logit π and log λ
                lp += sum(logpdf.(Normal(), θ.β_track_π)) + sum(logpdf.(Normal(), θ.β_track_λ))
                for r in 1:nr, j in 1:3
                    logitπ[r] += θ.β_track_π[j] * rc.track[r, j]
                    logλ[r] += θ.β_track_λ[j] * rc.track[r, j]
                end
            end
            π, λ = 1 ./ (1 .+ exp.(-logitπ)), exp.(logλ)
            if big
                lp += logpdf(Normal(log(0.05 / 0.95), 1), θ.a_ρ) +
                      logpdf(Turing.truncated(Normal(log(10), 1); lower = 0), θ.δ_λ2)
                ρ, λ2 = 1 / (1 + exp(-θ.a_ρ)), exp(θ.a_λ + θ.δ_λ2)
            end
            rowpdf(x, r) = big ? Diomedes.pacebig_logpdf(x, θ.σ, π[r], λ[r], ρ, λ2) :
                                 Diomedes.paceloss_logpdf(x, θ.σ, π[r], λ[r])
            rowint(lo, hi, r) = big ? Diomedes.pacebig_loginterval(lo, hi, θ.σ, π[r], λ[r], ρ, λ2) :
                                      Diomedes.paceloss_loginterval(lo, hi, θ.σ, π[r], λ[r])
            κ = ones(nr)
            a = cdrv ? θ.a_comp .- mean(θ.a_comp) : θ.σ_comp .* (θ.z_comp .- mean(θ.z_comp))
            seasonmean(v) = [mean(v[g.mach_season .== g.mach_season[j]]) for j in 1:nm]
            if kap
                # κ scales drivers in the likelihood; car spread via the centred car prior
                lp += logpdf(H(0.25), θ.τ_κ) + sum(logpdf.(Distributions.TDist(3), θ.e_κ))
                lκ = [0.0; cumsum(θ.τ_κ .* θ.e_κ)]
                κs = exp.(lκ .- mean(lκ))
                lp += sum(logpdf.(Normal.(0, θ.σ_mach .* κs[cov.mach_season]), θ.b_mach))
                κ = κs[cov.season]
                b = θ.b_mach .- seasonmean(θ.b_mach)
                kb = ones(nr)
            else
                b = θ.σ_mach .* (θ.z_mach .- seasonmean(θ.z_mach))   # centred within season
                kb = κ
            end
            at, ac = a[g.t_comp], a[g.c_comp]
            if age
                # career curve: random walk over age bins, constant and linear parts
                # removed by weighted least squares (solved on √w-scaled rows)
                lp += logpdf(H(0.2ps.s), θ.τ_age) + sum(logpdf.(Normal(), θ.e_age))
                fr = [0.0; cumsum(θ.τ_age .* θ.e_age)]
                X = hcat(ones(length(fr)), Float64.(g.age_years)); w = sqrt.(g.age_rows)
                fage = fr .- X * ((w .* X) \ (w .* fr))
                at = at .+ fage[g.t_age]; ac = ac .+ fage[g.c_age]
            end
            if slp
                # per-driver slopes, centred across drivers, times (age - driver's mean age) in decades
                lp += logpdf(H(0.5ps.s), θ.τ_slope) + sum(logpdf.(Normal(), θ.u_slope))
                s = θ.τ_slope .* (θ.u_slope .- mean(θ.u_slope))
                ages, comps = g.age_years[vcat(g.t_age, g.c_age)], vcat(g.t_comp, g.c_comp)
                m = [any(comps .== d) ? mean(ages[comps .== d]) : 0.0 for d in 1:nc]
                at = at .+ s[g.t_comp] .* (g.age_years[g.t_age] .- m[g.t_comp]) ./ 10
                ac = ac .+ s[g.c_comp] .* (g.age_years[g.c_age] .- m[g.c_comp]) ./ 10
            end
            races = vcat(g.t_race, g.c_race)
            cs = vcat(κ[g.t_race] .* at .+ kb[g.t_race] .* b[g.t_mach],
                      κ[g.c_race] .* ac .+ kb[g.c_race] .* b[g.c_mach])
            if dev
                # in-season development: trend per car (centred within season) times the race's
                # season position (0 = first round, 1 = last) minus the car's mean position; not scaled by κ
                lp += logpdf(H(ps.s), θ.τ_dev) + sum(logpdf.(Normal(), θ.u_dev))
                t = θ.τ_dev .* (θ.u_dev .- seasonmean(θ.u_dev))
                rnd = Dict(r.race_key => r.round for r in eachrow(g.rows))
                pos = zeros(nr)
                for s in unique(g.race_season)
                    idx = findall(==(s), g.race_season)
                    rs = sort(unique([rnd[g.races[r]] for r in idx]))
                    for r in idx; pos[r] = (findfirst(==(rnd[g.races[r]]), rs) - 1) / (length(rs) - 1); end
                end
                machs = vcat(g.t_mach, g.c_mach)
                pm = [mean(pos[races[machs .== j]]) for j in 1:nm]
                cs = cs .+ t[machs] .* (pos[races] .- pm[machs])
            end
            if circ
                # circuit slopes: per car and covariate, centred within season, times the race's
                # covariate minus the car's mean over its rows; not scaled by κ
                lp += sum(logpdf.(H(0.5ps.s), θ.τ_cx)) + sum(logpdf.(Normal(), θ.u_cx))
                machs = vcat(g.t_mach, g.c_mach)
                for k in 1:3
                    u = θ.u_cx[((k - 1) * nm + 1):(k * nm)]
                    sl = θ.τ_cx[k] .* (u .- seasonmean(u))
                    xm = [mean(rc.x[races[machs .== j], k]) for j in 1:nm]
                    cs = cs .+ sl[machs] .* (rc.x[races, k] .- xm[machs])
                end
            end
            cbar = [mean(cs[races .== r]) for r in 1:nr]
            nt = length(g.y)
            for i in eachindex(g.y)
                r = g.t_race[i]
                lp += rowpdf(g.y[i] - (θ.γ[r] + cs[i] - cbar[r]), r)
            end
            for i in eachindex(g.lo)
                r = g.c_race[i]; μ = θ.γ[r] + cs[nt + i] - cbar[r]
                lp += rowint(g.lo[i] - μ, g.hi[i] - μ, r)
            end
            model = paceloss_effects(g; loss_duration = dur, era, pace_scale = kap, age, big_loss = big, slopes = slp, dev,
                                     centred_drivers = cdrv, driver_ν = tdrv ? 3 : nothing, race_hier = hg, race_mean = mg,
                                     circ, track = trk, circuits = ctab)
            @test logjoint(model, θ) ≈ lp
            # sampler gradient (uncompiled ReverseDiff through the fused rule) vs ForwardDiff,
            # in constrained space: positive-valued parameters are set positive by name
            rd = LDF(model; adtype = Diomedes.gap_sampler().adtype)
            fd = LDF(model; adtype = AutoForwardDiff())
            for _ in 1:2
                θx = map(v -> v isa AbstractVector ? 0.5 .* randn(rng, length(v)) : 0.5 * randn(rng), θ)
                θx = merge(θx, NamedTuple(k => (θ[k] isa AbstractVector ? 0.2 .+ rand(rng, length(θ[k])) : 0.2 + rand(rng))
                                          for k in keys(θ) if startswith(string(k), "σ") || startswith(string(k), "τ")))
                x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θx)])
                @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
            end
            # the likelihood must reach ReverseDiff through the fused rule: if a call
            # matches no binding, ReverseDiff traces the per-row loop instead (correct
            # gradients but ~30× slower; the tape here grew from ~450 to ~4,000-6,000)
            tape = Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(model), z),
                                                     reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θ)]))
            @test length(tape.tape) < 1000
        end
        @test_throws ArgumentError paceloss_effects(g; era = :century)
        @test_throws ArgumentError paceloss_effects(prepare_gaps(res); era = :rw)   # one season
        @test_throws ArgumentError fit_paceloss(g; sampler = Diomedes.default_sampler(), n_samples = 10)
        @test let s = model_spec("pl_dur_rw"), e = (; family = :pl, loss_duration = true, era = :rw, pace_scale = false, age = false,
                                          big_loss = false, slopes = false, dev = false,
                                          centred_drivers = false, driver_ν = nothing, race_hier = false, race_mean = true,
                                          circ = false, track = false, walk = false, walk1 = false, works = false, dinc = false)
            s[keys(e)] == e && !any(s[k] for k in setdiff(keys(s), keys(e)))
        end
        @test let s = model_spec("pl_dur_rw_kappa"), e = (; family = :pl, loss_duration = true, era = :rw, pace_scale = true, age = false,
                                                big_loss = false, slopes = false, dev = false,
                                          centred_drivers = false, driver_ν = nothing, race_hier = false, race_mean = true,
                                          circ = false, track = false, walk = false, walk1 = false, works = false, dinc = false)
            s[keys(e)] == e && !any(s[k] for k in setdiff(keys(s), keys(e)))
        end
        @test model_spec("pl_dur_rw_kappa_age_big_slope").slopes
        @test model_spec("pl_dur_rw_kappa_age_big_slope_dev").dev
        let s = model_spec("pl_dur_rw_kappa_age_big_slope_dev_circ_track")
            @test s.circ && s.track && s.dev && s.era === :rw
            @test Diomedes.model_param_names(s)[(end - 1):end] == (:τ_cx, :u_cx)
            @test (:β_track_π, :β_track_λ) ⊆ Diomedes.model_param_names(s)
        end
        @test !model_spec("pl_dur_rw").circ && !model_spec("pl_dur_rw").track
        @test_throws ArgumentError paceloss_effects(g; circ = true, circuits = ctab)       # needs the pace scale
        @test model_spec("pl_dur_rw_kappa_cdrv").centred_drivers
        @test model_spec("pl_dur_rw_kappa_cdrv_tdrv").driver_ν == 3
        @test Diomedes.model_param_names(model_spec("pl_dur_rw_kappa_cdrv"))[4] === :a_comp
        @test model_spec("pl_dur_rw_kappa_age_big_slope_dev_hgam").race_hier
        @test model_spec("pl_dur_rw_kappa_age_big_slope_dev_hgam").era === :rw
        @test Diomedes.model_param_names(model_spec("pl_hgam"))[6:8] == (:μ_γ, :τ_γ, :γ)
        @test Diomedes.model_param_names(model_spec("pl_mgam"))[6:7] == (:μ_γ, :γ)
        @test_throws ArgumentError paceloss_effects(g; race_hier = true, race_mean = true)
        # the learned common race level is the default (#36); _fgam is the fixed prior, _mgam an alias
        @test !model_spec("pl_dur_rw_fgam").race_mean && !model_spec("pl_dur_rw_fgam").race_hier
        @test model_spec("pl_dur_rw_mgam") == model_spec("pl_dur_rw")
        @test !model_spec("pl_hgam").race_mean
        @test Diomedes.model_param_names(model_spec("pl_fgam"))[6] === :γ
        @test_throws ArgumentError model_spec("pl_mgam_fgam")
        @test_throws ArgumentError model_spec("pl_hgam_fgam")
        @test paceloss_effects(g).args.race_mean && !paceloss_effects(g; race_hier = true).args.race_mean
        @test !paceloss_effects(g; race_mean = false).args.race_mean
        @test_throws ArgumentError paceloss_effects(g; dev = true)             # needs pace_scale
        @test_throws ArgumentError paceloss_effects(g; slopes = true)          # needs pace_scale
        @test model_spec("pl_dur_rw_kappa_age").age
        @test model_spec("pl_dur_rw_kappa_big").big_loss
        @test_throws ArgumentError paceloss_effects(g; big_loss = true)        # needs pace_scale
        @test_throws ArgumentError paceloss_effects(g; age = true)             # needs pace_scale
        @test model_spec("t4").family === :t4
        @test_throws ArgumentError model_spec("pl_rw_decade")
        @test cov.regime == [searchsortedlast(REGIME_STARTS, s) for s in g.race_season]
    end

    @testset "pointwise log-likelihoods sum to the fused likelihood" begin
        rng = Xoshiro(41)
        g = prepare_gaps(fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019))
        nc, nm, nr = length(g.competitors), length(g.machines), length(g.races)
        sc, cov = Diomedes.season_counts(g), LossCovariates(g)
        base = (; σ_comp = 0.7, σ_mach = 1.1, z_comp = randn(rng, nc), z_mach = randn(rng, nm),
                γ = randn(rng, nr))
        θt = (; base..., σ_y = 0.6)
        rows = Diomedes.gap_rows(θt, g)
        @test length(rows) == length(g.y) + length(g.lo)
        @test sum(rows) ≈ Diomedes.gap_loglik(θt.γ, θt.z_comp, θt.σ_comp, θt.z_mach, θt.σ_mach, θt.σ_y,
                                              g, sc, StudentTNoise(4))
        cr = CareerRows(g)
        for (dur, era, kap, age, slp) in ((false, :none, false, false, false), (true, :regime, false, false, false),
                                          (false, :rw, false, false, false), (true, :rw, false, false, false),
                                          (true, :rw, true, false, false), (false, :none, true, false, false),
                                          (true, :rw, true, true, false), (true, :rw, true, false, true),
                                          (true, :rw, true, true, true))
            θ = (; base..., σ = 0.4, a_π = -1.2, a_λ = 0.6, β_dur_π = 0.25, β_dur_λ = 0.3,
                 (kap ? (; τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm)) : (;))...,
                 (age ? (; τ_age = 0.2, e_age = randn(rng, length(g.age_years) - 1)) : (;))...,
                 (slp ? (; τ_slope = 0.3, u_slope = randn(rng, nc)) : (;))...,
                 τ_π_era = 0.3, τ_λ_era = 0.2, z_π_era = randn(rng, cov.n_regimes), z_λ_era = randn(rng, cov.n_regimes),
                 τ_π_rw = 0.1, τ_λ_rw = 0.2, e_π_rw = randn(rng, cov.n_seasons - 1), e_λ_rw = randn(rng, cov.n_seasons - 1))
            πr, λr = Diomedes.race_loss(θ, cov, dur, era)
            @test sum(Diomedes.paceloss_rows(θ, g; loss_duration = dur, era)) ≈ (kap ?
                  Diomedes.paceloss_loglik_cc(θ.γ, θ.z_comp, θ.σ_comp, θ.b_mach, θ.σ, πr, λr,
                                              race_pace_scale(θ, cov),
                                              (age || slp) ? driver_offsets(cr, age ? age_curve(θ.τ_age, θ.e_age, AgeCurveBasis(g)) : nothing,
                                                                            slp ? career_slopes(θ.τ_slope, θ.u_slope) : nothing) :
                                                             nothing, g, sc) :
                  Diomedes.paceloss_loglik(θ.γ, θ.z_comp, θ.σ_comp, θ.z_mach, θ.σ_mach, θ.σ, πr, λr, nothing, g, sc))
        end
        # in-season development (#14): rows vs the fused rule, the car offset passed as offset/κ
        θd = (; base..., σ = 0.4, a_π = -1.2, a_λ = 0.6, β_dur_π = 0.25, β_dur_λ = 0.3,
              τ_π_rw = 0.1, τ_λ_rw = 0.2, e_π_rw = randn(rng, cov.n_seasons - 1), e_λ_rw = randn(rng, cov.n_seasons - 1),
              τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm), τ_dev = 0.5, u_dev = randn(rng, nm))
        πr, λr = Diomedes.race_loss(θd, cov, true, :rw)
        κr = race_pace_scale(θd, cov)
        dr = DevRows(g)
        hd = dev_trends(θd.τ_dev, θd.u_dev, g, sc)[dr.mach] .* dr.posc ./ κr[vcat(g.t_race, g.c_race)]
        @test sum(Diomedes.paceloss_rows(θd, g; loss_duration = true, era = :rw)) ≈
              Diomedes.paceloss_loglik_cc(θd.γ, θd.z_comp, θd.σ_comp, θd.b_mach, θd.σ, πr, λr, κr, hd, g, sc)
        @test all(abs(sum(dev_trends(1.0, randn(rng, nm), g, sc)[g.mach_season .== s])) < 1e-10 for s in 1:maximum(g.mach_season))
        # two-component incident loss (#16)
        θb = (; base..., σ = 0.4, a_π = -1.2, a_λ = 0.6, β_dur_π = 0.25, β_dur_λ = 0.3,
              τ_π_rw = 0.1, τ_λ_rw = 0.2, e_π_rw = randn(rng, cov.n_seasons - 1), e_λ_rw = randn(rng, cov.n_seasons - 1),
              τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm), a_ρ = -2.5, δ_λ2 = 2.0)
        πr, λr = Diomedes.race_loss(θb, cov, true, :rw)
        @test sum(Diomedes.paceloss_rows(θb, g; loss_duration = true, era = :rw)) ≈
              Diomedes.paceloss_loglik_cc(θb.γ, θb.z_comp, θb.σ_comp, θb.b_mach, θb.σ, πr, λr,
                                          race_pace_scale(θb, cov), nothing,
                                          1 / (1 + exp(-θb.a_ρ)), exp(θb.a_λ + θb.δ_λ2), g, sc)
        # circuit slopes and track terms (#81): rows vs the fused rule, the slope term passed as offset/κ
        covc = LossCovariates(g; circuits = synthetic_circuits(g, rng))
        θc = (; base..., σ = 0.4, a_π = -1.2, a_λ = 0.6, β_track_π = randn(rng, 3), β_track_λ = randn(rng, 3),
              τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm), τ_cx = [0.3, 0.5, 0.2],
              u_cx = randn(rng, 3nm))
        πr, λr = Diomedes.race_loss(θc, covc, false, :none)
        κr = race_pace_scale(θc, covc)
        crw = CircuitRows(g, covc.circuits)
        hc = circuit_offsets(circuit_slopes(θc.τ_cx, θc.u_cx, g, sc), crw) ./ κr[vcat(g.t_race, g.c_race)]
        @test sum(Diomedes.paceloss_rows(θc, g; cov = covc)) ≈
              Diomedes.paceloss_loglik_cc(θc.γ, θc.z_comp, θc.σ_comp, θc.b_mach, θc.σ, πr, λr, κr, hc, g, sc)
        # the fused car part (#92): same values, and its rule's gradient matches ReverseDiff
        # through the indexing and broadcasting it replaced
        rs = covc.season[vcat(g.t_race, g.c_race)]
        κs = Diomedes.season_pace_scale(θc)
        cx = circuit_slopes(θc.τ_cx, θc.u_cx, g, sc)
        @test Diomedes.car_row_offsets(cx, crw.mach, crw.xc, κs, rs) ≈ hc
        w = randn(rng, length(hc))
        RD = Diomedes.ReverseDiff
        fused(c, k) = sum(w .* Diomedes.car_row_offsets(c, crw.mach, crw.xc, k, rs))
        plain(c, k) = sum(w .* (circuit_offsets(c, crw) ./ k[rs]))
        gf, gp = RD.gradient(fused, (cx, κs)), RD.gradient(plain, (cx, κs))
        @test gf[1] ≈ gp[1] && gf[2] ≈ gp[2]
        @test_throws ArgumentError Diomedes.race_loss(θc, cov, false, :none)       # track terms need circuits
    end

    @testset "chassis and engine walks (#95)" begin
        rng = Xoshiro(95)
        RD = Diomedes.ReverseDiff
        # the walk recursion against a dense lower-triangular product, values and gradients
        prev = [0, 1, 2, 0, 4, 0, 6, 7, 8]
        L = zeros(length(prev), length(prev))
        for k in eachindex(prev)
            j = k
            while j != 0; L[k, j] = 1; j = prev[j]; end
        end
        st, w = randn(rng, length(prev)), randn(rng, length(prev))
        @test Diomedes.walk_path(st, prev) ≈ L * st
        @test RD.gradient(s -> sum(w .* Diomedes.walk_path(s, prev)), st) ≈ RD.gradient(s -> sum(w .* (L * s)), st)
        # per-row offsets against plain indexing
        xc, xe = randn(rng, 5), randn(rng, 3)
        ic, ie = rand(rng, 1:5, 40), rand(rng, 1:3, 40)
        w = randn(rng, 40)
        @test Diomedes.walk_offsets(xc, ic, xe, ie) ≈ xc[ic] .+ xe[ie]
        gf = RD.gradient((a, b) -> sum(w .* Diomedes.walk_offsets(a, ic, b, ie)), (xc, xe))
        gp = RD.gradient((a, b) -> sum(w .* (a[ic] .+ b[ie])), (xc, xe))
        @test gf[1] ≈ gp[1] && gf[2] ≈ gp[2]

        # two seasons (2019 and a relabelled copy as 2020), as for the other race-model terms
        res = fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019)
        res2 = copy(res); res2.season .= 2020; res2.event_id .= replace.(res2.event_id, "2019" => "2020")
        g = prepare_gaps(vcat(res, res2))
        # the tables for the copy: 2019's entries relabelled 2020, and ranges ending in 2019 extended
        real = car_table()
        e19 = real.entries[real.entries.season .== 2019, :]
        e20 = copy(e19); e20.season .= 2020
        ext(t) = transform(t, :last_year => ByRow(y -> y == 2019 ? 2020 : y) => :last_year)
        cars = merge(real, (; entries = vcat(real.entries[real.entries.season .!= 2020, :], e20),
                            works = ext(real.works), engines = ext(real.engines)))
        cw = CarWalks(g, cars)
        nrow_ = length(g.y) + length(g.lo)
        @test length(cw.row_ch) == length(cw.row_en) == nrow_
        rows = vcat(g.rows[g.rows.kind .== :timed, :], g.rows[g.rows.kind .== :lapped, :]; cols = :union)
        # every row's states are in its own season; each walk starts once and steps once
        for ws in (cw.chassis, cw.engine)
            @test all(ws.first .== (ws.prev .== 0))
            @test all(ws.season[ws.prev[.!ws.first]] .== ws.season[.!ws.first] .- 1)
            @test sum(ws.counts) == length(ws.prev)
        end
        @test cw.chassis.season[cw.row_ch] == rows.season .- 2018
        @test cw.engine.season[cw.row_en] == rows.season .- 2018
        @test any(startswith("milton-keynes 2019"), cw.chassis.names) && any(startswith("honda 2019"), cw.engine.names)
        # the 2020 steps: no ownership change; no regulation change between 2019 and 2020
        @test all(cw.chassis.own .== 0) && all(cw.chassis.reg .== 0) && all(cw.engine.reg .== 0)
        # flags from the tables: Racing Point's ownership change (2019), the 2017 and 2014 rules
        ws, _ = Diomedes._walk_states([("silverstone", 2017), ("silverstone", 2018), ("silverstone", 2019),
                                       ("mercedes", 2013), ("mercedes", 2015)], 2013, 7,
                                      Set([("silverstone", 2019)]), [2014, 2017])
        @test ws.names == ["mercedes 2013", "mercedes 2015", "silverstone 2017", "silverstone 2018", "silverstone 2019"]
        @test ws.prev == [0, 1, 0, 3, 4] && ws.own == [0, 0, 0, 0, 1] && ws.reg == [0, 1, 0, 0, 0]
        @test ws.rootgap ≈ [1, sqrt(2), 1, 1, 1]

        # the model: rows vs the fused rule with the walk offsets (and the works term, #100)
        # in the κ-scaled driver offset, and per-driver incident offsets on logit π (#101)
        nc, nm, nr = length(g.competitors), length(g.machines), length(g.races)
        cov = LossCovariates(g; cars)
        sc = Diomedes.season_counts(g)
        @test sum(cw.row_works) > 0 && sum(cw.row_works) < length(cw.row_works)     # Ferrari, Mercedes, Renault rows
        @test all(cw.row_works[findall(==("ferrari"), first.(split.(cw.chassis.names[cw.row_ch])))] .== 1)
        θ = (; σ_comp = 0.7, σ_mach = 0.5, σ = 0.4, z_comp = randn(rng, nc), μ_γ = 1.8, γ = randn(rng, nr),
             a_π = -1.2, a_λ = 0.6, β_dur_π = 0.25, β_dur_λ = 0.3,
             τ_π_rw = 0.1, τ_λ_rw = 0.2, e_π_rw = randn(rng, cov.n_seasons - 1), e_λ_rw = randn(rng, cov.n_seasons - 1),
             τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm), a_ρ = -2.5, δ_λ2 = 2.0,
             τ_ch = 0.4, σ0_ch = 0.9, z_ch = randn(rng, length(cw.chassis.prev)),
             τ_en = 0.3, σ0_en = 0.7, z_en = randn(rng, length(cw.engine.prev)),
             δ_own = 0.5, δ_reg = 0.4, β_works = -0.3, τ_dinc = 0.6, u_dinc = randn(rng, nc))
        πr, λr = Diomedes.race_loss(θ, cov, true, :rw)
        xc, xe = car_walk_states(θ, cov.cars)
        hw = Diomedes.walk_offsets(xc, cw.row_ch, xe, cw.row_en) .+ θ.β_works .* cw.row_works
        πd = Diomedes.driver_incident_offsets(θ)
        @test abs(sum(πd)) < 1e-10
        ρ, λ2 = 1 / (1 + exp(-θ.a_ρ)), exp(θ.a_λ + θ.δ_λ2)
        @test sum(Diomedes.paceloss_rows(θ, g; loss_duration = true, era = :rw, cov)) ≈
              Diomedes.paceloss_loglik_cc(θ.γ, θ.z_comp, θ.σ_comp, θ.b_mach, θ.σ, πr, λr, race_pace_scale(θ, cov), hw,
                                          ρ, λ2, πd, g, sc)
        # with all offsets zero, the driver-offset method equals the race-level one
        @test Diomedes.paceloss_loglik_cc(θ.γ, θ.z_comp, θ.σ_comp, θ.b_mach, θ.σ, πr, λr, race_pace_scale(θ, cov), hw,
                                          ρ, λ2, zeros(nc), g, sc) ≈
              Diomedes.paceloss_loglik_cc(θ.γ, θ.z_comp, θ.σ_comp, θ.b_mach, θ.σ, πr, λr, race_pace_scale(θ, cov), hw,
                                          ρ, λ2, g, sc)
        # the driver-offset rule's gradient against ReverseDiff through the plain loop
        w8 = randn(rng, nc)
        fr(o) = Diomedes.paceloss_loglik_cc(θ.γ, θ.z_comp, θ.σ_comp, θ.b_mach, θ.σ, πr, λr, race_pace_scale(θ, cov), hw,
                                            ρ, λ2, o, g, sc)
        fp(o) = Diomedes._loglik_cc(θ.γ, θ.z_comp, θ.σ_comp, θ.b_mach, θ.σ, πr, λr, race_pace_scale(θ, cov), hw,
                                    [ρ, λ2], o, g, sc)
        @test RD.gradient(fr, πd) ≈ RD.gradient(fp, πd)
        # states centred within season
        @test all(abs(sum(xc[cw.chassis.season .== s])) < 1e-10 for s in 1:2)
        @test all(abs(sum(xe[cw.engine.season .== s])) < 1e-10 for s in 1:2)
        # sampler gradient vs ForwardDiff, and every term reaches ReverseDiff through the fused rules,
        # for the combined model and for the #95 model (three step multipliers)
        LDF = Turing.DynamicPPL.LogDensityFunction
        LDP = Turing.DynamicPPL.LogDensityProblems
        θ3 = (; (k => v for (k, v) in pairs(θ) if k ∉ (:a_ρ, :δ_λ2, :δ_reg, :β_works, :τ_dinc, :u_dinc))...,
              δ_reg_ch = 0.4, δ_reg_en = 0.6)
        # non-centred car effects (#102): z_res in b_mach's place
        θn = NamedTuple(k === :b_mach ? (:z_res => v) : (k => v) for (k, v) in pairs(θ))
        @test sum(Diomedes.paceloss_rows(θn, g; loss_duration = true, era = :rw, cov)) ≈
              sum(Diomedes.paceloss_rows(merge(θ, (; b_mach = Diomedes.car_residuals(θn, cov))), g;
                                         loss_duration = true, era = :rw, cov))
        for (θm, kw) in ((θ, (; big_loss = true, walk1 = true, works = true, dinc = true)), (θ3, (;)),
                         (θn, (; big_loss = true, walk1 = true, works = true, dinc = true, ncres = true)))
            model = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, cars, kw...)
            rd = LDF(model; adtype = Diomedes.gap_sampler().adtype)
            fd = LDF(model; adtype = AutoForwardDiff())
            for _ in 1:2
                θx = map(v -> v isa AbstractVector ? 0.5 .* randn(rng, length(v)) : 0.5 * randn(rng), θm)
                θx = merge(θx, NamedTuple(k => 0.2 + rand(rng) for k in keys(θm)
                                          if any(p -> startswith(string(k), p), ("σ", "τ", "δ"))))
                x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θx)])
                @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
            end
            tape = RD.GradientTape(z -> LDP.logdensity(LDF(model), z), reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θm)]))
            @test length(tape.tape) < 1000
        end
        # level parametrisation (#102): same log joint at the mapped point; rule vs plain AD
        let m3 = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, cars),
            ml = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, wlev = true, cars)
            θl = Diomedes.walk_to_levels(θ3, cw)
            # same posterior; densities over different coordinates differ by the Jacobian of
            # z_first → level (σ0 per walk), the rest of the change of variables being unit
            nwc, nwe = count(cw.chassis.first), count(cw.engine.first)
            @test logjoint(ml, θl) ≈ logjoint(m3, θ3) - nwc * log(θ3.σ0_ch) - nwe * log(θ3.σ0_en)
            @test all(car_walk_states(θl, cw) .≈ car_walk_states(θ3, cw))
            vl = Diomedes.walk_level_inputs(θl.τ_ch, θl.δ_own, θl.δ_reg_ch, θl.zl_ch, cw.chassis)
            wv = randn(rng, length(vl))
            # dense equivalent of walk_level_path: x = A v
            n = length(vl); A = zeros(n, n)
            for j in 1:n
                e = zeros(n); e[j] = 1
                A[:, j] = Diomedes.walk_level_path(e, cw.chassis.prev)
            end
            @test Diomedes.walk_level_path(vl, cw.chassis.prev) ≈ A * vl
            @test RD.gradient(v -> sum(wv .* Diomedes.walk_level_path(v, cw.chassis.prev)), vl) ≈ A' * wv
            rd = LDF(ml; adtype = Diomedes.gap_sampler().adtype); fd = LDF(ml; adtype = AutoForwardDiff())
            x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θl)])
            @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
            @test length(RD.GradientTape(z -> LDP.logdensity(LDF(ml), z), x).tape) < 1000
        end
        # non-centred engine levels (#102): z_first → level/σ0_en has unit Jacobian
        let m3 = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, cars),
            mn = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, wlevn = true, cars)
            θl = Diomedes.walk_to_levels(θ3, cw; nc_en = true)
            @test logjoint(mn, θl) ≈ logjoint(m3, θ3) - count(cw.chassis.first) * log(θ3.σ0_ch)
            @test all(car_walk_states(θl, cw) .≈ car_walk_states(θ3, cw))
            @test haskey(θl, :zl_ch) && haskey(θl, :zn_en) && !haskey(θl, :zl_en)
            @test model_spec("pl_dur_rw_kappa_walk_wlevn").wlevn && !model_spec("pl_dur_rw_kappa_walk_wlevn").wlev
            rd = LDF(mn; adtype = Diomedes.gap_sampler().adtype); fd = LDF(mn; adtype = AutoForwardDiff())
            x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θl)])
            @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
        end
        # centred pace-scale steps (#102): same posterior; Jacobian of e_κ → s_κ = τ_κ·e_κ
        let m3 = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, cars),
            mk = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, kc = true, cars)
            θk = NamedTuple(k === :e_κ ? (:s_κ => θ3.τ_κ .* v) : (k => v) for (k, v) in pairs(θ3))
            @test Diomedes.season_pace_scale(θk) ≈ Diomedes.season_pace_scale(θ3)
            @test logjoint(mk, θk) ≈ logjoint(m3, θ3) - length(θ3.e_κ) * log(θ3.τ_κ)
        end
        # walk states and the κ path sampled directly (#102): same posterior; Jacobians of the maps
        let m3 = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, cars),
            ms = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, walk = true, wst = true,
                                  kst = true, cars)
            xc, xe = Diomedes.car_walk_raw(θ3, cw)
            lκ = Diomedes.rw_path(θ3.τ_κ .* θ3.e_κ) .+ 0.01
            θs = NamedTuple(k === :z_ch ? (:xs_ch => xc) : k === :z_en ? (:xs_en => xe) : k === :e_κ ? (:l_κ => lκ) : (k => v)
                            for (k, v) in pairs(θ3))
            @test all(car_walk_states(θs, cw) .≈ car_walk_states(θ3, cw))
            @test Diomedes.season_pace_scale(θs) ≈ Diomedes.season_pace_scale(θ3)
            sdc = Diomedes.walk_steps(θ3.τ_ch, θ3.σ0_ch, θ3.δ_own, θ3.δ_reg_ch, ones(length(xc)), cw.chassis)
            sde = Diomedes.walk_steps(θ3.τ_en, θ3.σ0_en, 0.0, θ3.δ_reg_en, ones(length(xe)), cw.engine)
            # z → x has Jacobian ∏ sd; e_κ → l_κ adds the mean coordinate (prior N(0, 0.05)) and ∏ τ_κ
            @test logjoint(ms, θs) ≈ logjoint(m3, θ3) - sum(log, sdc) - sum(log, sde) -
                                     length(θ3.e_κ) * log(θ3.τ_κ) + logpdf(Normal(0, 0.05), mean(lκ))
            # walk_diffs inverts walk_path; its rule vs plain AD
            st = randn(rng, length(xc))
            @test Diomedes.walk_diffs(Diomedes.walk_path(st, cw.chassis.prev), cw.chassis.prev) ≈ st
            wv = randn(rng, length(xc))
            @test RD.gradient(x -> sum(wv .* Diomedes.walk_diffs(x, cw.chassis.prev)), xc) ≈
                  RD.gradient(x -> sum(wv .* [p == 0 ? x[k] : x[k] - x[p] for (k, p) in enumerate(cw.chassis.prev)]), xc)
            rd = LDF(ms; adtype = Diomedes.gap_sampler().adtype); fd = LDF(ms; adtype = AutoForwardDiff())
            x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θs)])
            @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
            @test length(RD.GradientTape(z -> LDP.logdensity(LDF(ms), z), x).tape) < 1000
        end
        # every option together: ReverseDiff vs ForwardDiff and tape size
        let mall = paceloss_effects(g; loss_duration = true, era = :rw, pace_scale = true, big_loss = true, walk = true,
                                    walk1 = true, works = true, dinc = true, ncres = true, wlev = true, kc = true, cars)
            θa = Diomedes.walk_to_levels(θn, cw)
            θa = NamedTuple(k === :e_κ ? (:s_κ => θa.τ_κ .* v) : (k => v) for (k, v) in pairs(θa))
            @test Tuple(keys(θa)) == Diomedes.model_param_names(model_spec(
                "pl_dur_rw_kappa_big_walk1_works_dinc_ncres_wlev_kc"))
            rd = LDF(mall; adtype = Diomedes.gap_sampler().adtype); fd = LDF(mall; adtype = AutoForwardDiff())
            x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θa)])
            @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
            @test length(RD.GradientTape(z -> LDP.logdensity(LDF(mall), z), x).tape) < 1000
        end
        s = model_spec("pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk")
        @test s.walk && !s.walk1 && s.circ && Diomedes.model_param_names(s)[(end - 8):end] == Diomedes.walk_param_names(true)
        s1 = model_spec("pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc")
        @test s1.walk && s1.walk1 && s1.works && s1.dinc
        @test Diomedes.model_param_names(s1)[(end - 10):end] ==
              (Diomedes.walk_param_names(true, true)..., :β_works, :τ_dinc, :u_dinc)
        @test !model_spec("pl_dur_rw_kappa").walk
        @test_throws ArgumentError paceloss_effects(g; pace_scale = true, works = true, cars)     # works needs the walks
        @test_throws ArgumentError paceloss_effects(g; pace_scale = true, dinc = true)            # dinc needs big losses
        @test_throws ArgumentError paceloss_effects(g; walk = true, cars)        # needs the pace scale
    end

    @testset "race circuit covariates (#81)" begin
        rng = Xoshiro(61)
        g = prepare_gaps(fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019))
        ctab = synthetic_circuits(g, rng)
        rc = RaceCircuits(g, ctab)
        nr, nm = length(g.races), length(g.machines)
        ev = [first(g.rows.event_id[g.rows.race_key .== k]) for k in g.races]
        row(e) = ctab[findfirst(==(e), ctab.event_id), :]
        # speed: layout length × the winner's laps ÷ the winner's time, averaged over the layout's dry races
        L = [maximum(skipmissing(g.rows.laps[g.rows.race_key .== k])) for k in g.races]
        raw = [row(e).length_km * L[r] / (g.race_minutes[r] / 60) for (r, e) in enumerate(ev)]
        @test 150 < median(raw) < 250
        for r in 1:nr
            same = [i for i in 1:nr if row(ev[i]).circuit_id == row(ev[r]).circuit_id]
            dry = filter(i -> row(ev[i]).condition == "dry", same)
            @test rc.x_raw[r, 1] ≈ mean(raw[isempty(dry) ? same : dry])
            @test rc.x_raw[r, 2] == row(ev[r]).S && rc.x_raw[r, 3] == row(ev[r]).temp_c
            @test rc.track[r, :] == [row(ev[r]).W / 2, row(ev[r]).condition == "mixed", row(ev[r]).condition == "wet"]
        end
        @test all(abs.(mean(rc.x; dims = 1)) .< 1e-12) && all(std(rc.x; dims = 1) .≈ 1)
        @test rc.x_names == [:speed, :rough, :temp] && rc.track_names == [:tight, :mixed, :wet]
        # a covariate with no variation across the races is 0, not NaN
        @test all(RaceCircuits(g, transform(ctab, :S => (s -> fill(1, length(s))) => :S)).x[:, 2] .== 0)
        # covariates centred over each car-season's rows; slopes centred within season
        cr = CircuitRows(g, rc)
        @test all(maximum(abs.(sum(cr.xc[cr.mach .== m, :]; dims = 1))) < 1e-10 for m in 1:nm if any(cr.mach .== m))
        sc = Diomedes.season_counts(g)
        cx = circuit_slopes([0.3, 0.5, 0.2], randn(rng, 3nm), g, sc)
        @test all(abs(sum(cx[((k - 1) * nm + 1):(k * nm)][g.mach_season .== s])) < 1e-10 for k in 1:3, s in 1:maximum(g.mach_season))
        @test circuit_offsets(cx, cr) ≈ [sum(cx[(k - 1) * nm + cr.mach[i]] * cr.xc[i, k] for k in 1:3) for i in eachindex(cr.mach)]
        @test_throws ArgumentError RaceCircuits(g, ctab[2:end, :])
        # the reference tables: one row per race 1950–2026 (17 of 2026 run), with walls, surface and conditions
        ct = circuit_table()
        @test nrow(ct) == 1166 && Set(ct.condition) == Set(["dry", "mixed", "wet"])
        @test count(==("wet"), ct.condition) == 63 && count(==("mixed"), ct.condition) == 107
        @test all(in(0:2), ct.W) && all(in(0:2), ct.S)
    end

    @testset "K-fold CV (#18)" begin
        rng = Xoshiro(51)
        g = prepare_gaps(fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019))
        nc, nm, nr = length(g.competitors), length(g.machines), length(g.races)
        n = length(g.y) + length(g.lo)
        K = 5
        folds = kfold_folds(g, K; rng)
        @test length(folds) == n && all(in(1:K), folds)
        races = vcat(g.t_race, g.c_race)
        @test all(count(==(k), folds[races .== r]) <= cld(count(==(r), races), K) for r in 1:nr, k in 1:K)
        # keeping every row reproduces g
        gall = subset_gaps(g, trues(n))
        @test gall.y == g.y && gall.lo == g.lo && gall.age_rows == g.age_rows && nrow(gall.rows) == nrow(g.rows)
        train = folds .!= 1
        gt = subset_gaps(g, train)
        @test length(gt.y) + length(gt.lo) == count(train) == nrow(gt.rows)
        @test sum(gt.age_rows) == count(train)
        @test gt.y == g.y[train[1:length(g.y)]]
        # rows in the training subset score the same whether evaluated on the subset
        # or on the full data with the training mask, so held-out rows get the
        # predictions of the model fitted to the subset
        cov = LossCovariates(g)
        θ = (; σ_comp = 0.7, σ_mach = 1.1, σ = 0.4, z_comp = randn(rng, nc), γ = randn(rng, nr),
             a_π = -1.2, a_λ = 0.6, β_dur_π = 0.25, β_dur_λ = 0.3,
             τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm),
             τ_age = 0.2, e_age = randn(rng, length(g.age_years) - 1), a_ρ = -2.5, δ_λ2 = 2.0,
             τ_slope = 0.3, u_slope = randn(rng, nc))
        sub = Diomedes.paceloss_rows(θ, gt; loss_duration = true)
        full = Diomedes.paceloss_rows(θ, g; loss_duration = true, train, age_basis = AgeCurveBasis(gt))
        @test full[train] ≈ sub
        # held-out rows differ from an evaluation that also centres on them
        @test !(full[.!train] ≈ Diomedes.paceloss_rows(θ, g; loss_duration = true)[.!train])
        # elpd per row: log mean likelihood over draws
        ll = randn(rng, 7, 3, 4)
        @test elpd_rows(ll) ≈ [log(mean(exp.(ll[:, :, j]))) for j in 1:4]
    end

    @testset "retirement model (#13)" begin
        @test retirement_cause("Finished", 58) === :finish
        @test retirement_cause("+2 Laps", 56) === :finish
        @test retirement_cause("Not classified", 40) === :finish
        @test retirement_cause("Engine", 12) === :mech
        @test retirement_cause("Spun off", 3) === :incident
        @test retirement_cause("Collision damage", 0) === :collision
        @test retirement_cause("Did not qualify", 0) === :nonstart
        @test retirement_cause("Withdrew", 0) === :nonstart
        @test retirement_cause("Withdrew", 10) === :other
        @test retirement_cause("Disqualified", 58) === :excluded
        @test retirement_cause("Out of fuel", 50) === :other
        @test retirement_cause("Retired", 30) === :unknown     # cause not recorded (Jolpica from 2023, #55)
        # an unknown-cause retirement is the sum of the three cause-specific likelihoods
        for (n, L, η, b) in ((0, 5, (-1.0, -2.0, -2.5), (-1.0, -0.5, 0.3)), (33, 70, (0.4, -1.0, -3.0), (-3.0, -2.0, -1.0)))
            @test Diomedes.retire_row(η..., b..., n, L, -1) ≈
                  log(sum(exp(Diomedes.retire_row(η..., b..., n, L, k)) for k in 1:3))
        end
        # every outcome of one start (retire in lap 1..L from each cause, or finish) sums to 1
        for (L, η, b) in ((5, (-1.0, -2.0, -2.5), (-1.0, -0.5, 0.3)), (70, (0.4, -1.0, -3.0), (-3.0, -2.0, -1.0)))
            p = exp(Diomedes.retire_row(η..., b..., L, L, 0))
            for n in 0:(L - 1), k in 1:3
                p += exp(Diomedes.retire_row(η..., b..., n, L, k))
            end
            @test p ≈ 1
        end
        # fixture: 2019 rounds 1-2
        d1 = prepare_retirements(fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019))
        @test length(d1.n) == 40 && length(d1.races) == 2
        @test [count(==(k), d1.cause) for k in 1:3] == [3, 0, 2]      # Engine, Wheel; Damage, Collision damage
        @test all(d1.n .< d1.L .|| d1.cause .== 0)
        @test_throws ArgumentError retirement_effects(d1)              # one season
        # log density and fused gradient on a small simulated table (3 seasons)
        rng = Xoshiro(13)
        d = prepare_retirements(sim_retirements(rng; n_season = 3, n_race = 3))
        mc, rc = Diomedes.retire_counts(d)
        S, nm, nc, nr = length(d.seasons), length(d.machines), length(d.competitors), length(d.races)
        θ = (; a = [-1.0, -2.0, -2.5], τ_rw = [0.2, 0.1, 0.3], e_m = randn(rng, S - 1), e_i = randn(rng, S - 1),
             e_c = randn(rng, S - 1), b1 = [-2.0, -1.0, -0.5], σ_car = 0.6, σ_drv = 0.4, β_col = 0.7,
             σ_race = [0.2, 0.3, 0.4], z_car = randn(rng, nm), z_drv = randn(rng, nc), u_m = randn(rng, nr),
             u_i = randn(rng, nr), u_c = randn(rng, nr))
        car = θ.σ_car .* Diomedes.sum_to_zero_by(θ.z_car, d.mach_season, mc)
        drv = θ.σ_drv .* (θ.z_drv .- mean(θ.z_drv))
        path(τ, e) = (p = [0; cumsum(τ .* e)]; p .- mean(p))
        α = [θ.a[1] .+ path(θ.τ_rw[1], θ.e_m), θ.a[2] .+ path(θ.τ_rw[2], θ.e_i), θ.a[3] .+ path(θ.τ_rw[3], θ.e_c)]
        race = [θ.σ_race[k] .* Diomedes.sum_to_zero_by(u, d.race_season, rc) for (k, u) in enumerate((θ.u_m, θ.u_i, θ.u_c))]
        ll = sum(Diomedes.retire_row(α[1][d.season[j]] + car[d.mach[j]] + race[1][d.race[j]],
                                     α[2][d.season[j]] + drv[d.comp[j]] + race[2][d.race[j]],
                                     α[3][d.season[j]] + θ.β_col * drv[d.comp[j]] + race[3][d.race[j]],
                                     θ.b1..., d.n[j], d.L[j], d.cause[j]) for j in eachindex(d.n))
        model = retirement_effects(d)
        @test Turing.DynamicPPL.loglikelihood(model, θ) ≈ ll
        LDF = Turing.DynamicPPL.LogDensityFunction
        LDP = Turing.DynamicPPL.LogDensityProblems
        rd = LDF(model; adtype = Diomedes.gap_sampler().adtype)
        fd = LDF(model; adtype = AutoForwardDiff())
        pos = (:τ_rw, :σ_car, :σ_drv, :σ_race)
        for _ in 1:3
            x = reduce(vcat, [k in pos ? 0.1 .+ rand(rng, length(v)) : 0.5 .* randn(rng, length(v)) for (k, v) in pairs(θ)])
            @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
        end
        # the likelihood reaches ReverseDiff through the fused rule (not a traced per-start loop)
        x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θ)])
        @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(model), z), x).tape) < 1000
        # checkpointed sampling: a run stopped after 2 blocks and resumed gives the same draws as one run
        dir = mktempdir()
        fit(prefix, n) = fit_retirement(d; rng = Xoshiro(5), checkpoint = joinpath(dir, prefix), seed = 7,
                                        n_samples = n, block = 5, n_adapts = 10)
        fit("a", 10)
        @test isfile(joinpath(dir, "a_block2.jls")) && !isfile(joinpath(dir, "a_block3.jls"))
        resumed, whole = fit("a", 15), fit("b", 15)
        @test size(resumed[:σ_car]) == (15, 1)
        @test resumed[:σ_car] == whole[:σ_car] && resumed[:z_drv] == whole[:z_drv]
        @test all(==(resumed[:step_size][6]), resumed[:step_size][6:end])     # adaptation off after block 1
        @test_throws ArgumentError fit_retirement(d; checkpoint = joinpath(dir, "c"), n_chains = 2)
        # a compiled ReverseDiff tape gives wrong gradients through the fused rule (#62)
        @test_throws ArgumentError fit_retirement(d; sampler = Diomedes.NUTS(0.8; adtype = Diomedes.AutoReverseDiff(; compile = true)))
    end

    @testset "partial-distance gaps of retirements (#13)" begin
        # one 4-lap race: winner w, finishers f1 (losing time late) and f2, retirements
        cum = Dict(("2000-01", "w") => [100.0, 200, 300, 400], ("2000-01", "f1") => [101.0, 202, 302, 402],
                   ("2000-01", "f2") => [102.0, 204, 306, 408], ("2000-01", "r") => [103.0, 206],
                   ("2000-01", "early") => [110.0], ("2000-01", "dq") => [100.0, 201])
        rows = DataFrame(series = "f1", season = 2000, event_id = "2000-01", stage_id = "race",
                         competitor_id = ["w", "f1", "f2", "r", "early", "dq"], machine_id = ["m1", "m1", "m2", "m2", "m3", "m3"],
                         time_ms = Union{Missing,Float64}[400, 402, 408, missing, missing, missing],
                         suspended_ms = Union{Missing,Float64}[missing for _ in 1:6],
                         laps = Union{Missing,Int}[4, 4, 4, 2, 1, 2],
                         status = ["Finished", "Finished", "Finished", "Engine", "Accident", "Disqualified"])
        pg = partial_gaps(rows, cum)
        @test pg.competitor_id == ["r"]      # "early" did 1 lap (< 2), the disqualification is dropped
        y(C, n) = 100 * log(C[n] / cum[("2000-01", "w")][n])
        f1, f2, r = cum[("2000-01", "f1")], cum[("2000-01", "f2")], cum[("2000-01", "r")]
        shift = median([y(f1, 4) - y(f1, 2), y(f2, 4) - y(f2, 2)])
        @test only(pg.y_n) ≈ y(r, 2) && only(pg.shift) ≈ shift && only(pg.y_hat) ≈ y(r, 2) + shift
        @test only(pg.n_ref) == 2
        # trim: the gap taken k laps earlier; r's 2 laps leave too few after trimming 1
        @test isempty(partial_gaps(rows, cum; trim = 1))
        # as a timed row at y_hat; finishers and the winner unchanged, even if the partial row is faster
        g = prepare_gaps(with_partial_rows(rows, pg))
        m = partial_mask(g)
        @test count(m) == 1 && g.y[findfirst(m)] ≈ only(pg.y_hat)
        @test sort(g.y[.!m[1:length(g.y)]]) ≈ sort(prepare_gaps(rows).y)
        fast = copy(pg); fast.y_hat .= -1.0
        g2 = prepare_gaps(with_partial_rows(rows, fast))
        @test minimum(g2.y[.!partial_mask(g2)[1:length(g2.y)]]) == 0.0 && g2.y[findfirst(partial_mask(g2))] ≈ -1.0
    end

    @testset "Jolpica-era data: own times of lapped cars, unknown causes, cumulative laps (#55)" begin
        # a lapped car with its own time over its l laps gets the exact constant-pace gap,
        # inside its interval; one without keeps the interval
        rows = DataFrame(series = "f1", season = 2024, event_id = "2024-01", stage_id = "race",
                         competitor_id = ["w", "a", "b"], machine_id = ["m1", "m2", "m3"],
                         time_ms = Union{Missing,Float64}[6000.0, missing, missing],
                         own_time_ms = Union{Missing,Float64}[missing, 6050.0, missing],   # takes the flag just after the winner
                         suspended_ms = Union{Missing,Float64}[0.0, 0.0, 0.0],
                         laps = Union{Missing,Int}[60, 58, 58], status = ["Finished", "Lapped", "Lapped"])
        g = prepare_gaps(rows)
        ex = findfirst(g.rows.lapped_exact)
        @test g.rows.competitor_id[ex] == "a" && length(g.y) == 2 && length(g.lo) == 1
        ya = 100 * log((6050.0 / 58) / (6000.0 / 60))
        @test g.y[findfirst(==("a"), g.rows.competitor_id[g.rows.kind .== :timed])] ≈ ya
        @test 100 * log(60 / 58) <= ya < 100 * log(60 / 57)
        @test g.lo[1] ≈ 100 * log(60 / 58)
        # a red-flag suspension comes off the lapped car's own time too
        r2 = copy(rows); r2.time_ms[1] += 500.0; r2.own_time_ms[2] += 500.0; r2.suspended_ms .= 500.0
        @test prepare_gaps(r2).y ≈ g.y
        # partial gaps include retirements of unknown cause ("Retired")
        cum = Dict(("2000-01", "w") => [100.0, 200, 300, 400], ("2000-01", "f") => [101.0, 202, 302, 402],
                   ("2000-01", "r") => [103.0, 206])
        res = DataFrame(series = "f1", season = 2000, event_id = "2000-01", stage_id = "race",
                        competitor_id = ["w", "f", "r"], machine_id = ["m1", "m1", "m2"],
                        time_ms = Union{Missing,Float64}[400, 402, missing], suspended_ms = Union{Missing,Float64}[missing, missing, missing],
                        laps = Union{Missing,Int}[4, 4, 2], status = ["Finished", "Finished", "Retired"])
        pg = partial_gaps(res, cum)
        @test pg.competitor_id == ["r"] && only(pg.cause) === :unknown
        # cumulative laps from a LAP_SCHEMA table; a car with a gap in its laps is left out
        lt = DataFrame(series = "f1", season = 2024, event_id = "2024-01", competitor_id = ["a", "a", "a", "b", "b"],
                       lap = [2, 1, 3, 1, 3], time_ms = [91.0, 90.0, 92.0, 95.0, 95.0], position = Union{Missing,Int}[missing for _ in 1:5])
        c = cumulative_laps(lt)
        @test c[("2024-01", "a")] == [90.0, 181.0, 273.0] && !haskey(c, ("2024-01", "b"))
        # Jolpica's stray timings after a gap (#55): each car keeps laps 1–k up to its first gap
        lt2 = sort(vcat(lt, DataFrame(series = "f1", season = 2024, event_id = "2024-01", competitor_id = ["c", "c", "c"],
                                      lap = [1, 2, 26], time_ms = [97.0, 96.0, 97.0], position = Union{Missing,Int}[3, 3, missing])),
                   [:event_id, :competitor_id, :lap])
        kept = Diomedes.contiguous_laps(lt2)
        @test [(r.competitor_id, r.lap) for r in eachrow(kept)] == [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("c", 1), ("c", 2)]
    end

    @testset "step-size tuning after warm-up (#62)" begin
        Turing.@model function toy(y)
            m ~ Normal(0, 10)
            s ~ Distributions.truncated(Normal(0, 5); lower = 0)
            y .~ Normal(m, s)
        end
        model = toy(randn(Xoshiro(3), 200) .+ 2)
        dir = mktempdir()
        ch = Diomedes.run_nuts_checkpointed(model, Turing.InitFromParams((; m = 0.0, s = 1.0)), joinpath(dir, "t");
                                            n_samples = 40, n_adapts = 100, block = 20, tune_step = true)
        @test size(ch[:m]) == (40, 1)
        # every iteration, warm-up included, traced as it is sampled
        tr = readlines(joinpath(dir, "t_trace.tsv"))
        hd = split(tr[1], '\t')
        @test hd[1:4] == ["block", "iter", "n_steps", "tree_depth"] && "m" in hd && "s" in hd
        # (warm-up and draws; a block's starting point may add a row)
        @test count(l -> startswith(l, "1\t"), tr) >= 100 + 20 - 1 && count(l -> startswith(l, "2\t"), tr) >= 20
        @test parse(Float64, split(tr[end], '\t')[findfirst(==("m"), hd)]) == vec(ch[:m])[end]
        probe = readlines(joinpath(dir, "t_steptune.txt"))
        @test startswith(probe[1], "factor") && startswith(probe[end], "chosen")
        b1, b2 = Diomedes.read_block(joinpath(dir, "t_block1.jls")), Diomedes.read_block(joinpath(dir, "t_block2.jls"))
        @test b2.ϵ == parse(Float64, split(probe[end], "\t")[2]) && b2.ϵ >= b1.ϵ && b2.Minv == b1.Minv
        @test Diomedes.checkpoint_adapted(joinpath(dir, "t")) == (; Minv = b2.Minv, ϵ = b2.ϵ)
        # blocks hold no sampler state (it embeds the model and its data, #66)
        @test !occursin("TuringState", String(read(joinpath(dir, "t_block1.jls")))) &&
              !occursin("HMCState", String(read(joinpath(dir, "t_block2.jls"))))
        # a run resumed after block 1 gives the same draws as one run
        dir2 = mktempdir()
        cp(joinpath(dir, "t_block1.jls"), joinpath(dir2, "t_block1.jls"))
        ch2 = Diomedes.run_nuts_checkpointed(model, Turing.InitFromParams((; m = 0.0, s = 1.0)), joinpath(dir2, "t");
                                             n_samples = 40, n_adapts = 100, block = 20, tune_step = true)
        @test vec(ch2[:m]) == vec(ch[:m])
        # resumed with a fixed step size: later blocks use it, saved ones keep theirs (#66)
        dir3 = mktempdir()
        cp(joinpath(dir, "t_block1.jls"), joinpath(dir3, "t_block1.jls"))
        ch3 = Diomedes.run_nuts_checkpointed(model, Turing.InitFromParams((; m = 0.0, s = 1.0)), joinpath(dir3, "t");
                                             n_samples = 40, n_adapts = 100, block = 20, tune_step = true,
                                             fixed_step = 0.5b1.ϵ)
        b3 = Diomedes.read_block(joinpath(dir3, "t_block2.jls"))
        @test b3.ϵ == 0.5b1.ϵ && b3.Minv == b1.Minv && vec(ch3[:m])[1:20] == vec(ch[:m])[1:20]
        @test all(vec(b3.chain[:step_size]) .== 0.5b1.ϵ) && !isfile(joinpath(dir3, "t_steptune.txt"))
        # combines read the draws alone
        @test vec(Diomedes.block_chain(joinpath(dir, "t_block2.jls"))[:m]) == vec(b2.chain[:m]) == vec(ch[:m])[21:40]
        # with_step_size replaces only the step size
        st = only(Turing.loadstate(Diomedes.run_nuts(model, [Turing.InitFromParams((; m = 0.0, s = 1.0))]; n_samples = 5,
                                                      n_chains = 1, ensemble = Turing.MCMCSerial(),
                                                      sampler = Turing.NUTS(10, 0.8), rng = Xoshiro(1), progress = false,
                                                      progress_log = nothing, log_every = 100, save_state = true)))
        @test Diomedes.step_size(Diomedes.with_step_size(st, 0.123)) == 0.123
    end

    @testset "lap-time model (#12)" begin
        # pit inference: 90 s laps, a stop with in-lap +5 s and out-lap +18 s at laps 20/21, a spin (+8 s) at 40
        t = fill(90_000.0, 60); t[20] += 5_000; t[21] += 18_000; t[40] += 8_000
        @test infer_pit_stops(t, Set{Int}()) == [20]
        @test infer_pit_stops(t, Set([20, 21])) == Int[]          # race-wide slow laps are skipped
        # prepare_laps on a small race: 2 cars × 10 laps, a stop for car "a" at lap 4,
        # a safety car on lap 7 (every car +40%), lap 1 slow from the start
        rows = DataFrame(series = "f1", season = 2019, event_id = "2019-01", competitor_id = ["a", "b"],
                         machine_id = ["m1", "m2"])
        lt = DataFrame(series = String[], season = Int[], event_id = String[], competitor_id = String[], lap = Int[],
                       time_ms = Float64[], position = Union{Missing,Int}[])
        for (c, base) in (("a", 90_000.0), ("b", 91_000.0)), l in 1:10
            x = base * (l == 1 ? 1.1 : l == 7 ? 1.4 : 1.0) + (c == "a" && l == 4 ? 5_000 : c == "a" && l == 5 ? 18_000 : 0)
            push!(lt, ("f1", 2019, "2019-01", c, l, x, missing))
        end
        ps = DataFrame(series = "f1", season = 2019, event_id = "2019-01", competitor_id = "a", lap = 4,
                       duration_ms = Union{Missing,Float64}[23_000])
        for mode in (:recorded, :inferred)
            d = prepare_laps(lt, rows; pits = mode, pit_stops = ps)
            @test d.excluded[:lap1] == 2 && d.excluded[:slow_race] == 4 && d.excluded[:pit] == 2   # laps 7, 8
            ra = d.rows[d.rows.competitor_id .== "a", :]
            @test ra.lap == [2, 3, 6, 9, 10] && ra.stint == [1, 2, 1, 4, 5]
            @test d.y ≈ 100 .* log.(d.rows.time_ms ./ 91_000)      # median clean lap: 5 laps of a at 90 s, 7 of b at 91 s
        end
        @test_throws ArgumentError prepare_laps(lt, rows; pits = :recorded)
        # traffic and field pace (#113): three cars at 90, 91 and 121 s a lap, every car 3%
        # slower on lap 4, so c is caught by a and b. Crossings (laps done): …, c 242 (2),
        # a 270 (3), b 273 (3), a 362.7 (4), c 363 (3), b 366.73 (4), a 452.7 (5),
        # b 457.73 (5), c 487.63 (4), …, c 608.63 (5)
        rowst = DataFrame(series = "f1", season = 2019, event_id = "2019-01", competitor_id = ["a", "b", "c"],
                          machine_id = ["m1", "m1", "m2"])
        ltt = DataFrame(series = String[], season = Int[], event_id = String[], competitor_id = String[], lap = Int[],
                        time_ms = Float64[], position = Union{Missing,Int}[])
        for (c, t) in (("a", 90_000.0), ("b", 91_000.0), ("c", 121_000.0)), l in 1:6
            push!(ltt, ("f1", 2019, "2019-01", c, l, t * (l == 4 ? 1.03 : 1.0), missing))
        end
        dt = prepare_laps(ltt, rowst; pits = :inferred)
        tr = LapTraffic(dt, ltt)
        row(c, l) = findfirst(i -> dt.rows.competitor_id[i] == c && dt.rows.lap[i] == l, 1:nrow(dt.rows))
        let i = row("a", 4)            # start of a's lap 4: its lap-3 crossing at 270
            @test tr.ahead[i] ≈ 270 - 242 && tr.ahead_laps[i] == -1      # c, a lap down
            @test tr.behind[i] ≈ 3 && tr.behind_laps[i] == 0             # b
            @test isinf(tr.front[i]) && tr.back[i] ≈ 3                   # leading
        end
        let i = row("b", 6)            # start of b's lap 6: its lap-5 crossing at 457.73
            @test tr.ahead[i] ≈ 457.73 - 452.7 && tr.ahead_laps[i] == 0     # a, same lap
            @test tr.behind[i] ≈ 487.63 - 457.73 && tr.behind_laps[i] == -1 # c, a lap down
            @test tr.front[i] ≈ 457.73 - 452.7 && tr.back[i] ≈ 608.63 - 457.73
        end
        let i = row("c", 5)            # start of c's lap 5: its lap-4 crossing at 487.63
            @test tr.ahead[i] ≈ 487.63 - 457.73 && tr.ahead_laps[i] == 1    # b, a lap up
            @test tr.behind[i] ≈ 542.7 - 487.63 && tr.behind_laps[i] == 2   # a, two laps up
            @test tr.front[i] ≈ 487.63 - 366.73 && isinf(tr.back[i])
        end
        @test isinf(tr.ahead[row("a", 2)]) && tr.front[row("c", 2)] ≈ 121 - 91
        fp = field_pace(dt; window = 0, min_cars = 3)
        @test all(fp[dt.rows.lap .== 4] .> 1) && all(fp[dt.rows.lap .!= 4] .< 1)
        @test sum(fp) ≈ 0 atol = 1e-9                                   # detrended within the race
        @test all(iszero, field_pace(dt; window = 0, min_cars = 4))      # no lap has 4 cars
        # log density against a direct sum, and the fused gradient against ForwardDiff
        rng = Xoshiro(12)
        lt2 = DataFrame(series = String[], season = Int[], event_id = String[], competitor_id = String[], lap = Int[],
                        time_ms = Float64[], position = Union{Missing,Int}[])
        rows2 = DataFrame(series = String[], season = Int[], event_id = String[], competitor_id = String[], machine_id = String[])
        for s in 2018:2019, r in 1:2, (j, c) in enumerate(("a", "b", "c", "e"))
            push!(rows2, ("f1", s, "$s-0$r", c, "m$(cld(j, 2) + (s == 2019 && j > 2 ? 0 : 0))"))
            for l in 1:12
                push!(lt2, ("f1", s, "$s-0$r", c, l, 90_000 * (1 + 0.01 * randn(rng)) + 300j, missing))
            end
        end
        d = prepare_laps(lt2, rows2; pits = :inferred)
        nc, nm, nr = length(d.competitors), length(d.machines), length(d.races)
        θ = (; σ_comp = 0.5, σ_mach = 0.7, σ = 0.4, a_comp = randn(rng, nc), b_mach = randn(rng, nm), μ_γ = 0.1,
             γ = 0.2 .* randn(rng, nr), μ_δ = -1.0, τ_δ = 0.3, δ = randn(rng, nr), β = [0.02, 0.01], a_π = -1.2, a_λ = 0.3)
        a = θ.a_comp .- mean(θ.a_comp)
        b = Diomedes.sum_to_zero_by(θ.b_mach, d.mach_season, Diomedes.lap_mach_counts(d))
        h = [a[d.comp[j]] + b[d.mach[j]] + θ.β[1] * d.stint[j] / 10 + θ.β[2] * (d.stint[j] / 10)^2 for j in eachindex(d.y)]
        hbar = [mean(h[d.race .== r]) for r in 1:nr]
        fbar = [mean(d.frac[d.race .== r]) for r in 1:nr]
        ll = sum(Diomedes.paceloss_logpdf(d.y[j] - (θ.γ[d.race[j]] + θ.δ[d.race[j]] * (d.frac[j] - fbar[d.race[j]]) +
                                                     h[j] - hbar[d.race[j]]),
                                          θ.σ, Diomedes.logistic(θ.a_π), exp(θ.a_λ)) for j in eachindex(d.y))
        model = lap_effects(d)
        @test Turing.DynamicPPL.loglikelihood(model, θ) ≈ ll
        LDF = Turing.DynamicPPL.LogDensityFunction
        LDP = Turing.DynamicPPL.LogDensityProblems
        rd = LDF(model; adtype = Diomedes.gap_sampler().adtype)
        fd = LDF(model; adtype = AutoForwardDiff())
        pos = (:σ_comp, :σ_mach, :σ, :τ_δ)
        for _ in 1:3
            x = reduce(vcat, [k in pos ? [0.2 + rand(rng)] : v isa AbstractVector ? 0.5 .* randn(rng, length(v)) : [0.5 * randn(rng)]
                              for (k, v) in pairs(θ)])
            @test LDP.logdensity_and_gradient(rd, x)[2] ≈ LDP.logdensity_and_gradient(fd, x)[2]
        end
        x = reduce(vcat, [v isa AbstractVector ? v : [v] for v in values(θ)])
        @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(model), z), x).tape) < 1000
        # the warm-start layout matches the sampler's parameter vector
        ranges = NamedTuple(getfield(LDF(model), :_varname_ranges))
        let o = 0
            for (name, (n, _)) in Diomedes.lap_layout(d)
                @test ranges[name].range == (o + 1):(o + n)
                o += n
            end
            @test o == LDP.dimension(LDF(model))
        end
        # warm start: shared levels keep their variance, new levels get the group median,
        # scalars scale with the lap counts; a 2018-only fit warm-starts the 2018-19 one
        d_old = prepare_laps(lt2[lt2.season .== 2018, :], rows2[rows2.season .== 2018, :]; pits = :inferred)
        dim_old = sum(first(last(p)) for p in Diomedes.lap_layout(d_old))
        fake = (; hamiltonian = (; metric = (; M⁻¹ = collect(1.0:dim_old))), kernel = (; τ = (; integrator = (; ϵ = 0.04))))
        ws = lap_warm_start(fake, d_old, d)
        @test length(ws.Minv) == LDP.dimension(LDF(model))
        r = length(d_old.y) / length(d.y)
        @test ws.Minv[1:3] ≈ [1.0, 2.0, 3.0] .* r && ws.ϵ ≈ 0.04 * sqrt(r)
        γo, γn = NamedTuple(getfield(LDF(lap_effects(d_old)), :_varname_ranges)).γ.range, ranges.γ.range
        @test ws.Minv[γn][1:2] == collect(1.0:dim_old)[γo]                 # 2018 races, in the same order
        @test all(==(median(collect(1.0:dim_old)[γo])), ws.Minv[γn][3:4])  # 2019 races are new
        @test_throws DimensionMismatch lap_warm_start(fake, d, d)
        # the target acceptance rate reaches the sampler, cold and warm-started (#62)
        let dir = mktempdir()
            c1 = fit_laps(d; n_samples = 10, n_adapts = 10, checkpoint = joinpath(dir, "cold"), δ = 0.7)
            c2 = fit_laps(d; n_samples = 10, n_adapts = 10, checkpoint = joinpath(dir, "warm"), δ = 0.7, warm_start = ws)
            @test size(c1[:σ]) == (10, 1) && size(c2[:σ]) == (10, 1)
            @test Diomedes.lap_warm_sampler(ws.Minv, ws.ϵ; δ = 0.7).sampler.adaptor.ssa.δ == 0.7
        end
        # without the tyre terms: β = 0, the fused rule still applies, and the layout and
        # warm start (from a fit with them) drop β
        θ0 = Base.structdiff(θ, (; β = nothing))
        h0 = [a[d.comp[j]] + b[d.mach[j]] for j in eachindex(d.y)]
        h0bar = [mean(h0[d.race .== r]) for r in 1:nr]
        ll0 = sum(Diomedes.paceloss_logpdf(d.y[j] - (θ.γ[d.race[j]] + θ.δ[d.race[j]] * (d.frac[j] - fbar[d.race[j]]) +
                                                      h0[j] - h0bar[d.race[j]]),
                                           θ.σ, Diomedes.logistic(θ.a_π), exp(θ.a_λ)) for j in eachindex(d.y))
        model0 = lap_effects(d; tyres = false)
        @test Turing.DynamicPPL.loglikelihood(model0, θ0) ≈ ll0
        rd0, fd0 = LDF(model0; adtype = Diomedes.gap_sampler().adtype), LDF(model0; adtype = AutoForwardDiff())
        x0 = reduce(vcat, [k in pos ? [0.2 + rand(rng)] : v isa AbstractVector ? 0.5 .* randn(rng, length(v)) : [0.5 * randn(rng)]
                           for (k, v) in pairs(θ0)])
        @test LDP.logdensity_and_gradient(rd0, x0)[2] ≈ LDP.logdensity_and_gradient(fd0, x0)[2]
        @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(model0), z), x0).tape) < 1000
        ranges0 = NamedTuple(getfield(LDF(model0), :_varname_ranges))
        @test !haskey(ranges0, :β)
        let o = 0
            for (name, (n, _)) in Diomedes.lap_layout(d; tyres = false)
                @test ranges0[name].range == (o + 1):(o + n)
                o += n
            end
            @test o == LDP.dimension(rd0)
        end
        ws0 = lap_warm_start(fake, d_old, d; tyres_new = false)
        @test length(ws0.Minv) == LDP.dimension(rd0)
        @test ws0.Minv == ws.Minv[setdiff(eachindex(ws.Minv), ranges.β.range)]
        # career slopes: ages from the results (fractional years at the race), centred
        # per driver; unknown ages give 0
        rows3 = copy(rows2)
        rows3.event_date = [Date(r.season, 3, 1) + Day(14 * parse(Int, last(r.event_id))) for r in eachrow(rows3)]
        births = Dict("a" => Date(1990, 1, 1), "b" => Date(1985, 6, 1), "c" => Date(1995, 1, 1), "e" => missing)
        rows3.competitor_birth = [births[c] for c in rows3.competitor_id]
        d3 = prepare_laps(lt2, rows3; pits = :inferred)
        @test d3.y == d.y && d3.comp == d.comp
        for k in eachindex(d3.competitors)
            @test abs(mean(d3.agec[d3.comp .== k])) < 1e-12
        end
        @test all(iszero, d3.agec[d3.comp .== findfirst(==("e"), d3.competitors)])
        # ages are per season (the mean over the driver's laps that season): two seasons
        # a year apart give two values one year apart, whatever the laps per season
        let ag = d3.agec[d3.comp .== findfirst(==("a"), d3.competitors)]
            @test length(unique(ag)) == 2
            @test (maximum(ag) - minimum(ag)) * 10 ≈ 365 / 365.2425
        end
        # in-season position: with 2 rounds per season, round 1 is 0 and round 2 is 1,
        # centred on each car's mean over its laps
        let rp = [parse(Int, last(e)) - 1.0 for e in d3.rows.event_id]
            cm = [mean(rp[d3.mach .== d3.mach[j]]) for j in eachindex(d3.y)]
            @test d3.posc ≈ rp .- cm
        end
        model3 = lap_effects(d3; slopes = true)
        θ3 = merge(θ, (; τ_slope = 0.4, u_slope = randn(rng, nc)))
        sl = 0.4 .* (θ3.u_slope .- mean(θ3.u_slope))
        h3 = [a[d3.comp[j]] + sl[d3.comp[j]] * d3.agec[j] + b[d3.mach[j]] + θ.β[1] * d3.stint[j] / 10 +
              θ.β[2] * (d3.stint[j] / 10)^2 for j in eachindex(d3.y)]
        h3bar = [mean(h3[d3.race .== r]) for r in 1:nr]
        ll3 = sum(Diomedes.paceloss_logpdf(d3.y[j] - (θ.γ[d3.race[j]] + θ.δ[d3.race[j]] * (d3.frac[j] - fbar[d3.race[j]]) +
                                                       h3[j] - h3bar[d3.race[j]]),
                                           θ.σ, Diomedes.logistic(θ.a_π), exp(θ.a_λ)) for j in eachindex(d3.y))
        @test Turing.DynamicPPL.loglikelihood(model3, θ3) ≈ ll3
        # car development: dv = τ_dev·(u - season mean u) on the centred season position
        model4 = lap_effects(d3; slopes = true, dev = true)
        nm3 = length(d3.machines)
        θ4 = merge(θ3, (; τ_dev = 0.3, u_dev = randn(rng, nm3)))
        dvv = 0.3 .* Diomedes.sum_to_zero_by(θ4.u_dev, d3.mach_season, Diomedes.lap_mach_counts(d3))
        h4 = h3 .+ [dvv[d3.mach[j]] * d3.posc[j] for j in eachindex(d3.y)]
        h4bar = [mean(h4[d3.race .== r]) for r in 1:nr]
        ll4 = sum(Diomedes.paceloss_logpdf(d3.y[j] - (θ.γ[d3.race[j]] + θ.δ[d3.race[j]] * (d3.frac[j] - fbar[d3.race[j]]) +
                                                       h4[j] - h4bar[d3.race[j]]),
                                           θ.σ, Diomedes.logistic(θ.a_π), exp(θ.a_λ)) for j in eachindex(d3.y))
        @test Turing.DynamicPPL.loglikelihood(model4, θ4) ≈ ll4
        for (m, tyr, dv) in ((model3, true, false), (lap_effects(d3; tyres = false, slopes = true), false, false),
                             (model4, true, true), (lap_effects(d3; tyres = false, slopes = true, dev = true), false, true),
                             (lap_effects(d3; tyres = false, dev = true), false, true))
            rdm, fdm = LDF(m; adtype = Diomedes.gap_sampler().adtype), LDF(m; adtype = AutoForwardDiff())
            rg = NamedTuple(getfield(rdm, :_varname_ranges))
            x3 = 0.5 .* randn(rng, LDP.dimension(rdm))
            for k in (:σ_comp, :σ_mach, :σ, :τ_δ, :τ_slope, :τ_dev)
                haskey(rg, k) && (x3[rg[k].range] .= 0.2 + rand(rng))
            end
            @test LDP.logdensity_and_gradient(rdm, x3)[2] ≈ LDP.logdensity_and_gradient(fdm, x3)[2]
            @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(m), z), x3).tape) < 1000
            let o = 0
                for (name, (n, _)) in Diomedes.lap_layout(d3; tyres = tyr, slopes = haskey(rg, :τ_slope), dev = dv)
                    @test rg[name].range == (o + 1):(o + n)
                    o += n
                end
                @test o == LDP.dimension(rdm)
            end
        end
        # lead laps and tyre management (#66): the lead indicator comes from the lap's
        # position; η·lead and (td + tm)·s enter h, with td centred across drivers and
        # tm within season
        lt5 = copy(lt2); lt5.position = Vector{Union{Missing,Int}}(missing, nrow(lt5))
        lt5.position[lt5.competitor_id .== "a"] .= 1
        d5 = prepare_laps(lt5, rows3; pits = :inferred)
        @test d5.y == d3.y && d5.lead == [c == "a" ? 1.0 : 0.0 for c in d5.rows.competitor_id] && all(iszero, d3.lead)
        model5 = lap_effects(d5; slopes = true, dev = true, lead = true, tyre_mgmt = true)
        θ5 = merge(θ4, (; η = [0.4], τ_td = 0.2, u_td = randn(rng, nc), τ_tm = 0.3, u_tm = randn(rng, nm3)))
        tdv = 0.2 .* (θ5.u_td .- mean(θ5.u_td))
        tmv = 0.3 .* Diomedes.sum_to_zero_by(θ5.u_tm, d5.mach_season, Diomedes.lap_mach_counts(d5))
        s10 = d5.stint ./ 10
        scd = s10 .- [mean(s10[d5.comp .== d5.comp[j]]) for j in eachindex(d5.y)]
        scm = s10 .- [mean(s10[d5.mach .== d5.mach[j]]) for j in eachindex(d5.y)]
        @test d5.stint_cd ≈ scd && d5.stint_cm ≈ scm
        h5 = h4 .+ [0.4 * d5.lead[j] + tdv[d5.comp[j]] * scd[j] + tmv[d5.mach[j]] * scm[j] for j in eachindex(d5.y)]
        h5bar = [mean(h5[d5.race .== r]) for r in 1:nr]
        ll5 = sum(Diomedes.paceloss_logpdf(d5.y[j] - (θ.γ[d5.race[j]] + θ.δ[d5.race[j]] * (d5.frac[j] - fbar[d5.race[j]]) +
                                                       h5[j] - h5bar[d5.race[j]]),
                                           θ.σ, Diomedes.logistic(θ.a_π), exp(θ.a_λ)) for j in eachindex(d5.y))
        @test Turing.DynamicPPL.loglikelihood(model5, θ5) ≈ ll5
        for (m, flags) in ((model5, (; slopes = true, dev = true, lead = true, tyre_mgmt = true)),
                           (lap_effects(d5; tyres = false, lead = true), (; tyres = false, lead = true)),
                           (lap_effects(d5; tyre_mgmt = true), (; tyre_mgmt = true)))
            rdm, fdm = LDF(m; adtype = Diomedes.gap_sampler().adtype), LDF(m; adtype = AutoForwardDiff())
            rg = NamedTuple(getfield(rdm, :_varname_ranges))
            x5 = 0.5 .* randn(rng, LDP.dimension(rdm))
            for k in (:σ_comp, :σ_mach, :σ, :τ_δ, :τ_slope, :τ_dev, :τ_td, :τ_tm)
                haskey(rg, k) && (x5[rg[k].range] .= 0.2 + rand(rng))
            end
            @test LDP.logdensity_and_gradient(rdm, x5)[2] ≈ LDP.logdensity_and_gradient(fdm, x5)[2]
            @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(m), z), x5).tape) < 1000
            let o = 0
                for (name, (n, _)) in Diomedes.lap_layout(d5; flags...)
                    @test rg[name].range == (o + 1):(o + n)
                    o += n
                end
                @test o == LDP.dimension(rdm)
            end
        end
        # one shared stint curve per era: a refuelling season (2008) and a later one
        re08(e) = replace(e, "2018-" => "2008-")
        lt6 = copy(lt5); lt6.season = [s == 2018 ? 2008 : s for s in lt6.season]; lt6.event_id = re08.(lt6.event_id)
        rows6 = copy(rows3); rows6.season = [s == 2018 ? 2008 : s for s in rows6.season]; rows6.event_id = re08.(rows6.event_id)
        rows6.event_date = [Date(r.season, 3, 1) + Day(14 * parse(Int, last(r.event_id))) for r in eachrow(rows6)]
        d6 = prepare_laps(lt6, rows6; pits = :inferred)
        @test d6.eras == [:refuel, :norefuel] && d6.era == [s <= 2009 ? 1 : 2 for s in d6.rows.season]
        @test length(d3.eras) == 1 && Diomedes.lap_layout(d6)[findfirst(p -> first(p) === :β, Diomedes.lap_layout(d6))][2][1] == 4
        model6 = lap_effects(d6)
        β6 = [0.03, 0.01, 0.02, -0.005]
        θ6 = merge(θ, (; β = β6))
        b6 = Diomedes.sum_to_zero_by(θ.b_mach, d6.mach_season, Diomedes.lap_mach_counts(d6))
        h6 = [a[d6.comp[j]] + b6[d6.mach[j]] + β6[2d6.era[j] - 1] * d6.stint[j] / 10 + β6[2d6.era[j]] * (d6.stint[j] / 10)^2
              for j in eachindex(d6.y)]
        h6bar = [mean(h6[d6.race .== r]) for r in 1:nr]
        f6bar = [mean(d6.frac[d6.race .== r]) for r in 1:nr]
        ll6 = sum(Diomedes.paceloss_logpdf(d6.y[j] - (θ.γ[d6.race[j]] + θ.δ[d6.race[j]] * (d6.frac[j] - f6bar[d6.race[j]]) +
                                                       h6[j] - h6bar[d6.race[j]]),
                                           θ.σ, Diomedes.logistic(θ.a_π), exp(θ.a_λ)) for j in eachindex(d6.y))
        @test Turing.DynamicPPL.loglikelihood(model6, θ6) ≈ ll6
        let m = lap_effects(d6; slopes = true, tyre_mgmt = true)
            rdm, fdm = LDF(m; adtype = Diomedes.gap_sampler().adtype), LDF(m; adtype = AutoForwardDiff())
            rg = NamedTuple(getfield(rdm, :_varname_ranges))
            x6 = 0.5 .* randn(rng, LDP.dimension(rdm))
            for k in (:σ_comp, :σ_mach, :σ, :τ_δ, :τ_slope, :τ_td, :τ_tm)
                haskey(rg, k) && (x6[rg[k].range] .= 0.2 + rand(rng))
            end
            @test LDP.logdensity_and_gradient(rdm, x6)[2] ≈ LDP.logdensity_and_gradient(fdm, x6)[2]
            @test length(rg.β.range) == 4
        end
        # starts (#66): lap 1 with the grid slot; pit-lane starts (grid 0) left out
        rows7 = copy(rows3); rows7.grid = Union{Missing,Int}[c == "e" ? 0 : findfirst(==(c), ["a", "b", "c", "e"]) for c in rows7.competitor_id]
        d7 = prepare_laps(lt5, rows7; pits = :inferred)
        @test length(d7.s_y) == 4 * 3 && all(!=(findfirst(==("e"), d7.competitors)), d7.s_comp)
        for r in unique(d7.s_race)
            @test median(d7.s_y[d7.s_race .== r]) ≈ 0 atol = 1e-12
        end
        @test sort(unique(d7.s_grid)) ≈ [0.0, 0.1, 0.2] && isempty(d5.s_y)
        model7 = lap_effects(d7; starts = true)
        θ7 = merge(θ, (; γ1 = 0.1 .* randn(rng, nr), τ_sd = 0.3, u_sd = randn(rng, nc), τ_sc = 0.2, u_sc = randn(rng, nm),
                       g1 = [0.5, -0.1], σ1 = 0.8, a_π1 = -1.0, a_λ1 = 0.5))
        sdv = 0.3 .* (θ7.u_sd .- mean(θ7.u_sd))
        b7 = Diomedes.sum_to_zero_by(θ.b_mach, d7.mach_season, Diomedes.lap_mach_counts(d7))
        scv = 0.2 .* Diomedes.sum_to_zero_by(θ7.u_sc, d7.mach_season, Diomedes.lap_mach_counts(d7))
        h7 = [a[d7.s_comp[j]] + b7[d7.s_mach[j]] + sdv[d7.s_comp[j]] + scv[d7.s_mach[j]] + 0.5 * d7.s_grid[j] - 0.1 * d7.s_grid[j]^2
              for j in eachindex(d7.s_y)]
        h7bar = [mean(h7[d7.s_race .== r]) for r in 1:nr]
        ll7s = sum(Diomedes.paceloss_logpdf(d7.s_y[j] - (θ7.γ1[d7.s_race[j]] + h7[j] - h7bar[d7.s_race[j]]), 0.8,
                                            Diomedes.logistic(-1.0), exp(0.5)) for j in eachindex(d7.s_y))
        @test Turing.DynamicPPL.loglikelihood(model7, θ7) ≈ Turing.DynamicPPL.loglikelihood(lap_effects(d7), θ) + ll7s
        let m = lap_effects(d7; slopes = true, starts = true)
            rdm, fdm = LDF(m; adtype = Diomedes.gap_sampler().adtype), LDF(m; adtype = AutoForwardDiff())
            rg = NamedTuple(getfield(rdm, :_varname_ranges))
            x7 = 0.5 .* randn(rng, LDP.dimension(rdm))
            for k in (:σ_comp, :σ_mach, :σ, :τ_δ, :τ_slope, :τ_sd, :τ_sc, :σ1)
                haskey(rg, k) && (x7[rg[k].range] .= 0.2 + rand(rng))
            end
            @test LDP.logdensity_and_gradient(rdm, x7)[2] ≈ LDP.logdensity_and_gradient(fdm, x7)[2]
            @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(m), z), x7).tape) < 1000
            let o = 0
                for (name, (n, _)) in Diomedes.lap_layout(d7; slopes = true, starts = true)
                    @test rg[name].range == (o + 1):(o + n)
                    o += n
                end
                @test o == LDP.dimension(rdm)
            end
        end
        # population age curve (#12) and circuit slopes (#14): one-year bins of the
        # season age; per race the speed (layout length / median clean lap), street
        # score and temperature, standardised across races; both enter h on clean
        # laps and lap 1
        circ = DataFrame(event_id = d7.races, length_km = [4.0 + 0.5r for r in eachindex(d7.races)],
                         street = [isodd(r) ? 1.0 : 0.0 for r in eachindex(d7.races)],
                         temp_c = [15.0 + 3r for r in eachindex(d7.races)])
        d8 = prepare_laps(lt5, rows7; pits = :inferred, circuits = circ)
        @test d8.y == d7.y && d8.race_x_names == [:speed, :street, :temp]
        medms = [median(d8.rows.time_ms[d8.race .== r]) for r in 1:nr]
        @test d8.race_x_raw[:, 1] ≈ circ.length_km ./ (medms ./ 3.6e6)
        @test all(abs.(mean(d8.race_x; dims = 1)) .< 1e-12) && all(std(d8.race_x; dims = 1) .≈ 1)
        @test_throws ArgumentError prepare_laps(lt5, rows7; pits = :inferred, circuits = circ[2:end, :])
        # with circuit ids, the speed is the mean over the races on the same layout
        circ2 = transform(circ, :event_id => ByRow(e -> "x") => :circuit_id); circ2.length_km .= 5.0
        d8l = prepare_laps(lt5, rows7; pits = :inferred, circuits = circ2)
        @test all(d8l.race_x_raw[:, 1] .≈ mean(5.0 ./ (medms ./ 3.6e6))) && all(iszero, d8l.race_x[:, 1])
        @test d8.age_years == sort(unique(d8.age_years)) && length(d8.age_years) == 6
        let ecomp = findfirst(==("e"), d8.competitors)
            @test all(iszero, d8.abin[d8.comp .== ecomp]) && all(>(0), d8.abin[d8.comp .!= ecomp])
        end
        @test d8.age_rows == [count(==(i), d8.abin) for i in eachindex(d8.age_years)]
        # a lap's bin is its driver's season age in whole years
        let j = findfirst(==("a"), d8.rows.competitor_id)
            sa = mean([(x - Date(1990, 1, 1)).value for x in
                       rows7.event_date[(rows7.competitor_id .== "a") .& (rows7.season .== d8.rows.season[j])]] ./ 365.2425)
            @test d8.age_years[d8.abin[j]] == floor(Int, sa)
        end
        @test all(d8.s_abin[d8.s_comp .== findfirst(==("a"), d8.competitors)] .> 0)
        @test isempty(d7.age_years) == false && size(d7.race_x) == (nr, 0)
        @test_throws ArgumentError lap_effects(d7; circuits = true)
        model8 = lap_effects(d8; starts = true, age = true, circuits = true)
        na, nk = length(d8.age_years), 3
        θ8 = merge(θ7, (; τ_age = 0.3, e_age = randn(rng, na - 1), τ_cx = [0.2, 0.1, 0.05], u_cx = randn(rng, nm * nk)))
        fage = age_curve(0.3, θ8.e_age, AgeCurveBasis(d8.age_years, d8.age_rows))
        cxm = reduce(hcat, [θ8.τ_cx[k] .* Diomedes.sum_to_zero_by(θ8.u_cx[((k - 1) * nm + 1):(k * nm)], d8.mach_season,
                                                                    Diomedes.lap_mach_counts(d8)) for k in 1:nk])
        cterm(m, r) = sum(cxm[m, k] * d8.race_x[r, k] for k in 1:nk)
        b8 = Diomedes.sum_to_zero_by(θ.b_mach, d8.mach_season, Diomedes.lap_mach_counts(d8))
        h8 = [a[d8.comp[j]] + b8[d8.mach[j]] + θ.β[1] * d8.stint[j] / 10 + θ.β[2] * (d8.stint[j] / 10)^2 +
              (d8.abin[j] > 0 ? fage[d8.abin[j]] : 0.0) + cterm(d8.mach[j], d8.race[j]) for j in eachindex(d8.y)]
        h8bar = [mean(h8[d8.race .== r]) for r in 1:nr]
        f8bar = [mean(d8.frac[d8.race .== r]) for r in 1:nr]
        ll8 = sum(Diomedes.paceloss_logpdf(d8.y[j] - (θ.γ[d8.race[j]] + θ.δ[d8.race[j]] * (d8.frac[j] - f8bar[d8.race[j]]) +
                                                       h8[j] - h8bar[d8.race[j]]),
                                           θ.σ, Diomedes.logistic(θ.a_π), exp(θ.a_λ)) for j in eachindex(d8.y))
        hs8 = [a[d8.s_comp[j]] + b8[d8.s_mach[j]] + sdv[d8.s_comp[j]] + scv[d8.s_mach[j]] + 0.5 * d8.s_grid[j] -
               0.1 * d8.s_grid[j]^2 + (d8.s_abin[j] > 0 ? fage[d8.s_abin[j]] : 0.0) + cterm(d8.s_mach[j], d8.s_race[j])
               for j in eachindex(d8.s_y)]
        hs8bar = [mean(hs8[d8.s_race .== r]) for r in 1:nr]
        ll8s = sum(Diomedes.paceloss_logpdf(d8.s_y[j] - (θ7.γ1[d8.s_race[j]] + hs8[j] - hs8bar[d8.s_race[j]]), 0.8,
                                            Diomedes.logistic(-1.0), exp(0.5)) for j in eachindex(d8.s_y))
        @test Turing.DynamicPPL.loglikelihood(model8, θ8) ≈ ll8 + ll8s
        for flags in ((; starts = true, age = true, circuits = true), (; age = true), (; tyres = false, circuits = true),
                      (; slopes = true, dev = true, lead = true, tyre_mgmt = true, starts = true, age = true, circuits = true))
            m = lap_effects(d8; flags...)
            rdm, fdm = LDF(m; adtype = Diomedes.gap_sampler().adtype), LDF(m; adtype = AutoForwardDiff())
            rg = NamedTuple(getfield(rdm, :_varname_ranges))
            x8 = 0.5 .* randn(rng, LDP.dimension(rdm))
            for k in (:σ_comp, :σ_mach, :σ, :τ_δ, :τ_slope, :τ_dev, :τ_td, :τ_tm, :τ_sd, :τ_sc, :σ1, :τ_age, :τ_cx)
                haskey(rg, k) && (x8[rg[k].range] .= 0.2 .+ rand(rng, length(rg[k].range)))
            end
            @test LDP.logdensity_and_gradient(rdm, x8)[2] ≈ LDP.logdensity_and_gradient(fdm, x8)[2]
            @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(m), z), x8).tape) < 1000
            let o = 0
                for (name, (n, _)) in Diomedes.lap_layout(d8; flags...)
                    @test rg[name].range == (o + 1):(o + n)
                    o += n
                end
                @test o == LDP.dimension(rdm)
            end
        end
        # no per-car size cut (#108): every lap the cut dropped is kept
        let dk = prepare_laps(lt5, rows7; pits = :inferred, stoppage = false)
            @test length(dk.y) >= length(d7.y) && dk.excluded[:stoppage] == 0
        end
        # track terms and driver incident proneness (#108)
        @test size(d8.race_track) == (nr, 0)
        @test_throws ArgumentError lap_effects(d8; track = true)
        circt = transform(circ, :event_id => ByRow(e -> 1) => :W)
        circt.W = [r % 3 for r in eachindex(d7.races)]
        circt.condition = [r == 1 ? "wet" : r == 2 ? "mixed" : "dry" for r in eachindex(d7.races)]
        d9 = prepare_laps(lt5, rows7; pits = :inferred, circuits = circt)
        @test d9.race_track_names == [:tight, :mixed, :wet] && d9.race_track[:, 1] == circt.W ./ 2 &&
              d9.race_track[1, 3] == 1 && d9.race_track[2, 2] == 1 && sum(d9.race_track[:, 2:3]) == 2
        let m9 = lap_effects(d9; track = true, dinc = true), nc = length(d9.competitors)
            θ9 = merge(θ, (; β_track_π = [0.3, 0.8, 1.5], β_track_λ = [0.1, 0.2, 0.3], τ_dinc = 0.7, u_dinc = randn(rng, nc)))
            πr = Diomedes.logistic.(θ.a_π .+ d9.race_track * θ9.β_track_π)
            λr = exp.(θ.a_λ .+ d9.race_track * θ9.β_track_λ)
            o = 0.7 .* (θ9.u_dinc .- mean(θ9.u_dinc))
            b9 = Diomedes.sum_to_zero_by(θ.b_mach, d9.mach_season, Diomedes.lap_mach_counts(d9))
            h9 = [a[d9.comp[j]] + b9[d9.mach[j]] + θ.β[1] * d9.stint[j] / 10 + θ.β[2] * (d9.stint[j] / 10)^2 for j in eachindex(d9.y)]
            h9bar = [mean(h9[d9.race .== r]) for r in 1:nr]
            f9bar = [mean(d9.frac[d9.race .== r]) for r in 1:nr]
            ll9 = sum(Diomedes.paceloss_logpdf(d9.y[j] - (θ.γ[d9.race[j]] + θ.δ[d9.race[j]] * (d9.frac[j] - f9bar[d9.race[j]]) +
                                                           h9[j] - h9bar[d9.race[j]]), θ.σ,
                                               Diomedes.logistic(Diomedes.logit(πr[d9.race[j]]) + o[d9.comp[j]]),
                                               λr[d9.race[j]]) for j in eachindex(d9.y))
            @test Turing.DynamicPPL.loglikelihood(m9, θ9) ≈ ll9
            # without the terms the likelihood is unchanged from the scalar-π form
            @test Turing.DynamicPPL.loglikelihood(lap_effects(d9), θ) ≈ Turing.DynamicPPL.loglikelihood(lap_effects(d7), θ)
            # big incident losses (#108): pacebig_logpdf per lap, the drivers' offsets on the big-loss share
            θ9b = merge(θ9, (; a_ρ = -2.0, δ_λ2 = 2.0))
            λ29 = exp(θ.a_λ + 2.0)
            ll9b = sum(Diomedes.pacebig_logpdf(d9.y[j] - (θ.γ[d9.race[j]] + θ.δ[d9.race[j]] * (d9.frac[j] - f9bar[d9.race[j]]) +
                                                          h9[j] - h9bar[d9.race[j]]), θ.σ, πr[d9.race[j]],
                                               λr[d9.race[j]], Diomedes.logistic(-2.0 + o[d9.comp[j]]), λ29) for j in eachindex(d9.y))
            @test Turing.DynamicPPL.loglikelihood(lap_effects(d9; track = true, dinc = true, big = true), θ9b) ≈ ll9b
        end
        for flags in ((; track = true), (; dinc = true), (; big = true), (; track = true, dinc = true, slopes = true, starts = true,
                                                                            age = true, circuits = true, big = true))
            m = lap_effects(d9; flags...)
            rdm, fdm = LDF(m; adtype = Diomedes.gap_sampler().adtype), LDF(m; adtype = AutoForwardDiff())
            rg = NamedTuple(getfield(rdm, :_varname_ranges))
            x9 = 0.5 .* randn(rng, LDP.dimension(rdm))
            for k in (:σ_comp, :σ_mach, :σ, :τ_δ, :τ_slope, :τ_sd, :τ_sc, :σ1, :τ_age, :τ_cx, :τ_dinc, :τ_dinc1, :δ_λ2)
                haskey(rg, k) && (x9[rg[k].range] .= 0.2 .+ rand(rng, length(rg[k].range)))
            end
            @test LDP.logdensity_and_gradient(rdm, x9)[2] ≈ LDP.logdensity_and_gradient(fdm, x9)[2]
            @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(m), z), x9).tape) < 1000
            let o = 0
                for (name, (n, _)) in Diomedes.lap_layout(d9; flags...)
                    @test rg[name].range == (o + 1):(o + n)
                    o += n
                end
                @test o == LDP.dimension(rdm)
            end
        end
        # chassis and engine walks with the works term (#108): car tables for the fixture's
        # machines, two engine builders, one works pairing
        let seas = [d9.seasons[s] for s in d9.mach_season]
            mids = [replace(m, Regex("_$(y)\$") => "") for (m, y) in zip(d9.machines, seas)]
            ucon = unique(mids)
            eng(c) = isodd(findfirst(==(c), ucon)) ? "e1" : "e2"
            fc = (; entries = DataFrame(season = seas, round = fill(1, length(seas)), driver_id = fill("x", length(seas)),
                                        machine_id = mids, f1db_constructor = mids, f1db_engine = eng.(mids)),
                  works = DataFrame(f1db_constructor = ucon, first_year = 1900, last_year = 2100, works_id = ucon),
                  engines = DataFrame(f1db_engine = ["e1", "e2"], first_year = 1900, last_year = 2100, builder_id = ["e1", "e2"]),
                  ownership = DataFrame(works_id = String[], season = Int[]),
                  regulations = DataFrame(season = Int[], chassis = Int[], engine = Int[]),
                  works_engines = DataFrame(builder_id = ["e1"], works_id = [ucon[1]], first_year = 1900, last_year = 2100))
            cwl = CarWalks(d9, fc)
            @test length(cwl.row_ch) == length(d9.machines) && sum(cwl.row_works) == count(==(ucon[1]), mids)
            mw = lap_effects(d9; walk = true, works = true, cars = fc)
            θw = merge(θ, (; τ_ch = 0.3, σ0_ch = 1.0, z_ch = zeros(length(cwl.chassis.prev)), τ_en = 0.2, σ0_en = 0.5,
                           z_en = zeros(length(cwl.engine.prev)), δ_own = 0.2, δ_reg = 0.2, β_works = 0.0))
            # zero walks and works term: the base model's likelihood
            @test Turing.DynamicPPL.loglikelihood(mw, θw) ≈ Turing.DynamicPPL.loglikelihood(lap_effects(d9), θ)
            θw2 = merge(θw, (; β_works = 0.4))
            @test !(Turing.DynamicPPL.loglikelihood(mw, θw2) ≈ Turing.DynamicPPL.loglikelihood(mw, θw))
            @test_throws ArgumentError lap_effects(d9; works = true, cars = fc)
            # race-level chassis walk (#108): states per works per race, chained in race order
            rwk = RaceWalk(d9, fc)
            @test length(rwk.lap) == length(d9.y) && length(rwk.start) == length(d9.s_y) && sum(rwk.counts) == length(rwk.prev)
            @test all(rwk.prev[k] == 0 || rwk.prev[k] == k - 1 for k in eachindex(rwk.prev))
            wk(k) = first(rsplit(rwk.names[k], " "; limit = 2))
            @test all(rwk.prev[k] == 0 || wk(rwk.prev[k]) == wk(k) for k in eachindex(rwk.prev))
            @test all(rwk.boundary[k] == (rwk.prev[k] > 0 && d9.race_season[rwk.race[k]] != d9.race_season[rwk.race[rwk.prev[k]]])
                      for k in eachindex(rwk.prev))
            @test all(rwk.race[rwk.lap[j]] == d9.race[j] for j in eachindex(d9.y))
            @test_throws ArgumentError lap_effects(d9; rwalk = true, dev = true, cars = fc)
            # zero walk steps: no car part, as the base model with zero car effects
            mr = lap_effects(d9; rwalk = true, cars = fc)
            θr = merge((; (k => v for (k, v) in pairs(θ) if k ∉ (:σ_mach, :b_mach))...),
                       (; τ_cr = 0.1, τ_co = 0.3, σ0_ch = 1.0, xs_cr = zeros(length(rwk.prev)), δ_own = 0.2, δ_reg = 0.2))
            @test Turing.DynamicPPL.loglikelihood(mr, θr) ≈
                  Turing.DynamicPPL.loglikelihood(lap_effects(d9), merge(θ, (; b_mach = zeros(length(d9.machines)))))
            @test !(Turing.DynamicPPL.loglikelihood(mr, merge(θr, (; xs_cr = randn(rng, length(rwk.prev))))) ≈
                    Turing.DynamicPPL.loglikelihood(mr, θr))
            # directly sampled states: the step form's prior, less the log Jacobian Σ log sd of steps → states
            sdr = Diomedes.race_walk_sd(0.1, 0.3, 1.0, 0.2, 0.2, rwk)
            zr = randn(rng, length(rwk.prev))
            @test Diomedes.walk_states_logpdf(Diomedes.walk_path(zr .* sdr, rwk.prev), sdr, rwk.prev) ≈
                  sum(Diomedes.normlogpdf.(zr)) - sum(log.(sdr))
            for flags in ((; walk = true), (; walk = true, works = true, track = true, dinc = true, starts = true, big = true),
                          (; rwalk = true), (; rwalk = true, track = true, dinc = true, starts = true, big = true, sdrv = false),
                          (; rwalk = true, dinc = true, starts = true, dinc1 = true))
                m = lap_effects(d9; flags..., cars = fc)
                rdm, fdm = LDF(m; adtype = Diomedes.gap_sampler().adtype), LDF(m; adtype = AutoForwardDiff())
                rg = NamedTuple(getfield(rdm, :_varname_ranges))
                xw = 0.5 .* randn(rng, LDP.dimension(rdm))
                for k in (:σ_comp, :σ_mach, :σ, :τ_δ, :τ_sd, :τ_sc, :σ1, :τ_dinc, :τ_dinc1, :τ_ch, :σ0_ch, :τ_en, :σ0_en, :δ_own,
                          :δ_reg, :δ_λ2, :τ_cr, :τ_co)
                    haskey(rg, k) && (xw[rg[k].range] .= 0.2 .+ rand(rng, length(rg[k].range)))
                end
                @test LDP.logdensity_and_gradient(rdm, xw)[2] ≈ LDP.logdensity_and_gradient(fdm, xw)[2]
                @test length(Diomedes.ReverseDiff.GradientTape(z -> LDP.logdensity(LDF(m), z), xw).tape) < 1000
                let o = 0
                    for (name, (n, _)) in Diomedes.lap_layout(d9; flags..., cw = cwl, rwk = get(flags, :rwalk, false) ? rwk : nothing)
                        @test rg[name].range == (o + 1):(o + n)
                        o += n
                    end
                    @test o == LDP.dimension(rdm)
                end
            end
        end
        # warm start from a fit without them: the new blocks get the defaults
        ws8 = lap_warm_start(fake, d_old, d8; starts_new = true, age_new = true, circuits_new = true)
        r8 = NamedTuple(getfield(LDF(model8), :_varname_ranges))
        @test length(ws8.Minv) == LDP.dimension(LDF(model8))
        @test all(==(Diomedes.LAP_WARM_DEFAULT[:u_cx]), ws8.Minv[r8.u_cx.range]) &&
              all(==(Diomedes.LAP_WARM_DEFAULT[:e_age]), ws8.Minv[r8.e_age.range])
        ws5 = lap_warm_start(fake, d_old, d5; slopes_new = true, dev_new = true, lead_new = true, tyre_mgmt_new = true)
        r5 = NamedTuple(getfield(LDF(model5), :_varname_ranges))
        @test length(ws5.Minv) == LDP.dimension(LDF(model5))
        @test ws5.Minv[r5.η.range] == [Diomedes.LAP_WARM_DEFAULT[:η]] && all(==(0.01), ws5.Minv[r5.u_tm.range])
        # warm start of a slopes fit from a fit without them: new blocks get the defaults
        ws3 = lap_warm_start(fake, d_old, d3; slopes_new = true)
        r3 = NamedTuple(getfield(LDF(model3), :_varname_ranges))
        @test length(ws3.Minv) == LDP.dimension(LDF(model3))
        @test all(==(Diomedes.LAP_WARM_DEFAULT[:u_slope]), ws3.Minv[r3.u_slope.range])
        @test ws3.Minv[r3.τ_slope.range] == [Diomedes.LAP_WARM_DEFAULT[:τ_slope]]
        @test ws3.Minv[r3.a_π.range] ≈ ws.Minv[ranges.a_π.range]
        ws4 = lap_warm_start(fake, d_old, d3; slopes_new = true, dev_new = true)
        r4 = NamedTuple(getfield(LDF(model4), :_varname_ranges))
        @test length(ws4.Minv) == LDP.dimension(LDF(model4))
        @test all(==(Diomedes.LAP_WARM_DEFAULT[:u_dev]), ws4.Minv[r4.u_dev.range])
        @test ws4.Minv[r4.a_λ.range] ≈ ws.Minv[ranges.a_λ.range]
    end

    opt_in("DIOMEDES_SLOW_TESTS") && @testset "paceloss_effects recovers simulated effects" begin
        rng = Xoshiro(31)
        n_comp, n_season, cars_per_season, n_race_per_season = 24, 3, 6, 8
        comp_eff = 0.8 .* randn(rng, n_comp)
        mach_eff = [1.0 .* randn(rng, cars_per_season) for _ in 1:n_season]
        π_true, λ_true, σ_true = 0.2, 3.0, 0.3
        rows = DataFrame()
        for s in 1:n_season, r in 1:n_race_per_season
            γ = 1.0 * randn(rng)
            for (k, dr) in enumerate(randperm(rng, n_comp)[1:12])
                car = mod1(k, cars_per_season)
                loss = rand(rng) < π_true ? -λ_true * log(rand(rng)) : 0.0
                gap = γ + comp_eff[dr] + mach_eff[s][car] + σ_true * randn(rng) + loss
                push!(rows, (; series = "sim", season = 2000 + s, round = r, event_id = "$s-$r",
                             event_name = "", stage_id = "race", competitor_id = "d$dr",
                             competitor_name = "d$dr", codriver_id = missing,
                             machine_id = "c$car", class = missing, gap,
                             position = missing, status = "", classified = true))
            end
        end
        L, lap = 60, 90_000.0
        rows.time_ms = Vector{Union{Missing,Float64}}(undef, nrow(rows))
        rows.laps = Vector{Union{Missing,Int}}(undef, nrow(rows))
        for grp in groupby(rows, [:event_id])
            T = L * lap .* exp.(grp.gap ./ 100)
            Tw = minimum(T)
            for i in eachindex(T)
                l = T[i] < Tw * L / (L - 1) ? L : floor(Int, L * Tw / T[i]) + 1
                grp.laps[i] = l
                grp.time_ms[i] = l == L ? T[i] : missing
                grp.status[i] = l == L ? "Finished" : "+$(L - l) Lap" * (L - l == 1 ? "" : "s")
            end
        end
        g = prepare_gaps(select(rows, Not(:gap)))
        @test length(g.lo) > 0.15 * nrow(rows)
        chain = fit_paceloss(g; n_samples = 300, n_chains = 2, rng, progress = false)
        est_π = mean(1 ./ (1 .+ exp.(-vec(chain[:a_π]))))
        est_λ = mean(exp.(vec(chain[:a_λ])))
        @test 0.1 < est_π < 0.35
        @test 1.5 < est_λ < 5.0
        # a chain read back from disk keeps working after its key index is rebuilt (#32)
        io = IOBuffer(); serialize(io, chain); seekstart(io)
        back = rehash_chain!(deserialize(io))
        @test vec(back[:a_π]) == vec(chain[:a_π])
        eff = gap_effects_table(chain, g)
        est = Dict(zip(eff.competitors.label, eff.competitors.mean))
        @test cor([est["d$i"] for i in 1:n_comp], comp_eff .- mean(comp_eff)) > 0.8
        conv = convergence_summary(chain)
        @test conv.max_rhat < 1.1 && conv.min_ess > 20
        # convergence of the identified quantities (#25): all groups present and mixing
        ic = identified_convergence(chain, g, "pl")
        @test Set(keys(ic)) ⊇ Set((:overall, :drivers, :cars, :races, :scalars))
        @test ic.overall.max_rhat < 1.1 && ic.drivers.min_ess > 20
        # one mode for a converged fit; two when chains sit in different places (#64)
        @test chain_modes(chain, [:σ, :a_π, :a_λ]) == ["A", "A"]
        let x = randn(MersenneTwister(1), 500, 4)
            two = Dict(:p => x .+ [0 3 3 0], :q => x)
            @test chain_modes(two, [:p, :q]) == ["A", "B", "B", "A"]
            @test chain_modes(Dict(:q => x), [:q]) == fill("A", 4)
        end
        # pointwise log-likelihoods from the chain: shape, and draw (1, 1) sums to the fused likelihood
        ll = pointwise_loglik(chain, g, "pl")
        @test size(ll) == (300, 2, length(g.y) + length(g.lo))
        θ = Diomedes.draw(chain, (:σ_comp, :σ_mach, :σ, :z_comp, :z_mach, :γ, :a_π, :a_λ), 1, 1)
        @test sum(ll[1, 1, :]) ≈ Diomedes.paceloss_loglik(θ.γ, θ.z_comp, θ.σ_comp, θ.z_mach, θ.σ_mach,
            θ.σ, fill(1 / (1 + exp(-θ.a_π)), length(g.races)), fill(exp(θ.a_λ), length(g.races)),
            nothing, g, Diomedes.season_counts(g))
    end

    opt_in("DIOMEDES_SLOW_TESTS") && @testset "gap_effects recovers simulated effects with censoring" begin
        rng = Xoshiro(11)
        n_comp, n_season, cars_per_season, n_race_per_season = 24, 3, 6, 8
        comp_eff = 0.8 .* randn(rng, n_comp)
        mach_eff = [1.2 .* randn(rng, cars_per_season) for _ in 1:n_season]
        rows = DataFrame()
        for s in 1:n_season, r in 1:n_race_per_season
            γ = 1.0 * randn(rng)
            drivers = randperm(rng, n_comp)[1:12]
            for (k, dr) in enumerate(drivers)
                car = mod1(k, cars_per_season)
                gap = γ + comp_eff[dr] + mach_eff[s][car] + 0.5 * randn(rng)
                push!(rows, (; series = "sim", season = s, round = r, event_id = "$s-$r",
                             event_name = "", stage_id = "race", competitor_id = "d$dr",
                             competitor_name = "d$dr", codriver_id = missing,
                             machine_id = "c$car", class = missing, gap,
                             position = missing, status = "", classified = true))
            end
        end
        # gaps are in % of race time; on a 60-lap race one lap is ~1.7%, so the
        # slower part of each field ends up lapped
        L, lap = 60, 90_000.0
        rows.time_ms = Vector{Union{Missing,Float64}}(undef, nrow(rows))
        rows.laps = Vector{Union{Missing,Int}}(undef, nrow(rows))
        for grp in groupby(rows, [:event_id])
            T = L * lap .* exp.(grp.gap ./ 100)
            Tw = minimum(T)
            for i in eachindex(T)
                # laps completed when the winner finishes: largest l with (l - 1)·T/L < T_w
                l = T[i] < Tw * L / (L - 1) ? L : floor(Int, L * Tw / T[i]) + 1
                grp.laps[i] = l
                grp.time_ms[i] = l == L ? T[i] : missing
                grp.status[i] = l == L ? "Finished" : "+$(L - l) Lap" * (L - l == 1 ? "" : "s")
            end
        end
        g = prepare_gaps(select(rows, Not(:gap)))
        @test length(g.lo) > 0.15 * nrow(rows)                 # meaningful censoring
        chain = fit_gaps(g; n_samples = 300, n_chains = 2, rng, progress = false)
        eff = gap_effects_table(chain, g)
        est = Dict(zip(eff.competitors.label, eff.competitors.mean))
        truth = comp_eff .- mean(comp_eff)
        @test cor([est["d$i"] for i in 1:n_comp], truth) > 0.8
        conv = convergence_summary(chain)
        @test conv.max_rhat < 1.1 && conv.min_ess > 20
    end

    opt_in("DIOMEDES_SLOW_TESTS") && @testset "retirement model recovers simulated effects (#13)" begin
        rng = Xoshiro(13)
        rows, truth = sim_retirements(rng; n_season = 4, n_race = 12, n_car = 8, σ_car = 0.8, σ_drv = 0.6,
                                      β_col = 0.8, truth = true)
        d = prepare_retirements(rows)
        chain = fit_retirement(d; n_samples = 500, n_chains = 1, rng, progress = false)
        mc, _ = Diomedes.retire_counts(d)
        draws(f) = reduce(hcat, [f(i) for i in 1:size(chain[:σ_car], 1)])
        car = vec(mean(draws(i -> chain[:σ_car][i, 1] .* Diomedes.sum_to_zero_by(chain[:z_car][i, 1], d.mach_season, mc)); dims = 2))
        drv = vec(mean(draws(i -> chain[:σ_drv][i, 1] .* Diomedes.sum_to_zero(chain[:z_drv][i, 1])); dims = 2))
        true_car = [truth.car[m] for m in d.machines]
        true_drv = [truth.drv[parse(Int, c[2:end])] for c in d.competitors]
        @test cor(car, true_car) > 0.6
        @test cor(drv, true_drv) > 0.5
        @test abs(mean(chain[:β_col]) - 0.8) < 0.6
        @test abs(mean(chain[:a][i, 1][1] for i in 1:500) - (-1.0)) < 0.4
    end

    opt_in("DIOMEDES_SLOW_TESTS") && @testset "crossed_effects recovers simulated effects" begin
        rng = Xoshiro(1)
        n_comp, n_mach, n = 30, 15, 900
        comp_eff = 0.6 .* randn(rng, n_comp)
        mach_eff = 0.8 .* randn(rng, n_mach)
        comp = rand(rng, 1:n_comp, n)
        mach = rand(rng, 1:n_mach, n)
        y = comp_eff[comp] .+ mach_eff[mach] .+ 0.5 .* randn(rng, n)
        rows = DataFrame(competitor_id = string.(comp), competitor_name = string.(comp))
        d = Diomedes.ModelData(y, comp, mach, string.(1:n_comp), string.(1:n_mach), rows)

        log = IOBuffer()
        chain = fit_effects(d; σ_y = nothing, n_samples = 300, n_chains = 2, rng,
                            progress = false, progress_log = log, log_every = 150)
        logged = String(take!(log))
        # warm-up (150 = 300 ÷ 2 adaptation iterations) is logged, then trimmed from the chain
        @test occursin("chain 1/2: iter 150/450 (warm-up)", logged)
        @test occursin(r"chain 2/2: iter 450/450, .*total ETA", logged)
        @test Turing.FlexiChains.niters(chain) == 300
        @test 0.35 < mean(chain[:σ_y]) < 0.65
        conv = convergence_summary(chain)
        @test conv.max_rhat < 1.1
        # Guard against catastrophic mixing only. With 2 × 300 draws, worst-parameter
        # ESS (usually σ_comp) ranged 28-66 across seeds and init strategies, so a
        # tighter threshold is flaky. Real-data convergence is checked in
        # scripts/legacy_parity.jl (4 × 1000 draws: max R-hat 1.006, min ESS 1025).
        @test conv.min_ess > 20
        eff = effects_table(chain, d)
        est = eff.competitors.mean[sortperm(parse.(Int, eff.competitors.label))]
        @test cor(est, comp_eff) > 0.85
        @test abs(mean(eff.competitors.mean)) < 1e-8   # sum-to-zero
    end

    opt_in("DIOMEDES_NETWORK_TESTS") && @testset "network: Jolpica red-flag suspension (Canada 2011)" begin
        res = fetch_results(JolpicaF1(; cache_dir = mktempdir()), 2011)
        can = res[res.event_name .== "Canadian Grand Prix", :]
        @test first(can.suspended_ms) ≈ 123 * 60_000 rtol = 0.02
        # the only other 2011 suspension: Monaco, ~20 min after Petrov's crash (#17)
        susp = filter(r -> coalesce(r.suspended_ms, 0.0) > 0, unique(res[:, [:event_name, :suspended_ms]]))
        @test sort(susp.event_name) == ["Canadian Grand Prix", "Monaco Grand Prix"]
        @test only(susp[susp.event_name .== "Monaco Grand Prix", :suspended_ms]) ≈ 20 * 60_000 rtol = 0.05
    end

    opt_in("DIOMEDES_NETWORK_TESTS") && @testset "network: WRCTiming" begin
        res = fetch_results(WRCTiming(; cache_dir = mktempdir()), 2025)
        @test nrow(res) > 1000
        @test length(unique(res.event_id)) > 10
        @test count(!ismissing, res.time_ms) > 0.8 * nrow(res)
    end
end
