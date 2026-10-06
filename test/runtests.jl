using Diomedes
using DataFrames
using JSON3
using CSV
using Random
using Statistics
using Test
using Serialization: serialize, deserialize
using Turing: logjoint
using Distributions: Normal, cdf, ccdf, logpdf
using ADTypes: AutoForwardDiff, AutoReverseDiff
import SpecialFunctions
import Turing
import Distributions

const FIXTURES = joinpath(@__DIR__, "fixtures")

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
        @test Diomedes.is_lapped_status("+1 Lap") && Diomedes.is_lapped_status("+12 Laps") &&
              Diomedes.is_lapped_status("Lapped")
        @test !Diomedes.is_lapped_status("Suspension") && !Diomedes.is_lapped_status("Finished") &&
              !Diomedes.is_lapped_status("+1 Lapse")
        retired = copy(res)
        k = findfirst(r -> ismissing(r.time_ms) && Diomedes.is_lapped_status(r.status), eachrow(retired))
        retired.status[k] = "Suspension"; retired.classified[k] = true   # retirement with a position
        @test nrow(prepare_gaps(retired).rows) == nrow(g.rows) - 1
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
        for ((dur, era, kap, age, big, slp, dev), cdrv, tdrv, hg, mg) in cases
            θ = (; σ_comp = 0.7, σ_mach = 1.1, σ = 0.4, z_comp = randn(rng, nc), z_mach = randn(rng, nm),
                 γ = randn(rng, nr), a_π = -1.2, a_λ = 0.6)
            dur && (θ = (; θ..., β_dur_π = 0.25, β_dur_λ = 0.3))
            k = era === :decade ? cov.n_decades : cov.n_regimes
            era in (:decade, :regime) && (θ = (; θ..., τ_π_era = 0.3, τ_λ_era = 0.2,
                                               z_π_era = randn(rng, k), z_λ_era = randn(rng, k)))
            era === :rw && (θ = (; θ..., τ_π_rw = 0.1, τ_λ_rw = 0.2, e_π_rw = randn(rng, cov.n_seasons - 1),
                                 e_λ_rw = randn(rng, cov.n_seasons - 1)))
            # pace scale (#15): no z_mach; τ_κ, e_κ and centred b_mach come last, in sampling order
            kap && (θ = (; (k => v for (k, v) in pairs(θ) if k !== :z_mach)...,
                         τ_κ = 0.3, e_κ = randn(rng, cov.n_seasons - 1), b_mach = randn(rng, nm)))
            age && (θ = (; θ..., τ_age = 0.2, e_age = randn(rng, length(g.age_years) - 1)))
            big && (θ = (; θ..., a_ρ = -2.5, δ_λ2 = 2.0))
            slp && (θ = (; θ..., τ_slope = 0.3, u_slope = randn(rng, nc)))
            dev && (θ = (; θ..., τ_dev = 0.4, u_dev = randn(rng, nm)))
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
                                     centred_drivers = cdrv, driver_ν = tdrv ? 3 : nothing, race_hier = hg, race_mean = mg)
            @test logjoint(model, θ) ≈ lp
            # sampler gradient (uncompiled ReverseDiff through the fused rule) vs ForwardDiff,
            # in constrained space: positive-valued parameters are set positive by name
            rd = LDF(model; adtype = Diomedes.gap_sampler().adtype)
            fd = LDF(model; adtype = AutoForwardDiff())
            for _ in 1:2
                θx = map(v -> v isa AbstractVector ? 0.5 .* randn(rng, length(v)) : 0.5 * randn(rng), θ)
                θx = merge(θx, NamedTuple(k => 0.2 + rand(rng) for k in keys(θ)
                                          if startswith(string(k), "σ") || startswith(string(k), "τ")))
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
        @test model_spec("pl_dur_rw") == (; family = :pl, loss_duration = true, era = :rw, pace_scale = false, age = false,
                                          big_loss = false, slopes = false, dev = false,
                                          centred_drivers = false, driver_ν = nothing, race_hier = false, race_mean = true)
        @test model_spec("pl_dur_rw_kappa") == (; family = :pl, loss_duration = true, era = :rw, pace_scale = true, age = false,
                                                big_loss = false, slopes = false, dev = false,
                                          centred_drivers = false, driver_ν = nothing, race_hier = false, race_mean = true)
        @test model_spec("pl_dur_rw_kappa_age_big_slope").slopes
        @test model_spec("pl_dur_rw_kappa_age_big_slope_dev").dev
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
