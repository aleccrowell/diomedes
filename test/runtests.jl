using Diomedes
using DataFrames
using JSON3
using Random
using Statistics
using Test

const FIXTURES = joinpath(@__DIR__, "fixtures")

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
        @test j.classified == j.classified_1
        @test j.status == j.status_1
    end

    @testset "prepare" begin
        res = fetch_results(ErgastCSV(joinpath(FIXTURES, "ergast")), 2019)
        d = prepare(res)
        @test length(d) == count(!ismissing, res.time_ms)
        @test issorted(d.competitors)
        @test d.competitors[d.competitor] == d.rows.competitor_id
        @test all(endswith("_2019"), d.machines)
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

        chain = fit_effects(d; σ_y = nothing, n_samples = 300, rng, progress = false)
        @test 0.35 < mean(chain[:σ_y]) < 0.65
        eff = effects_table(chain, d)
        est = eff.competitors.mean[sortperm(parse.(Int, eff.competitors.label))]
        @test cor(est, comp_eff) > 0.85
    end

    opt_in("DIOMEDES_NETWORK_TESTS") && @testset "network: WRCTiming" begin
        res = fetch_results(WRCTiming(; cache_dir = mktempdir()), 2025)
        @test nrow(res) > 1000
        @test length(unique(res.event_id)) > 10
        @test count(!ismissing, res.time_ms) > 0.8 * nrow(res)
    end
end
