using Diomedes
using DataFrames
using JSON3
using Random
using Statistics
using Test
using Turing: logjoint
using Distributions: Normal, cdf, ccdf, logpdf
using ADTypes: AutoForwardDiff, AutoReverseDiff
import Turing

const FIXTURES = joinpath(@__DIR__, "fixtures")

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
        # every classified, untimed car with 2 <= laps < winner's laps is an interval
        @test nrow(lapped) == count(r -> ismissing(r.time_ms) && r.classified, eachrow(res))
        r = first(lapped[lapped.laps .== 57, :])       # Australia 2019 (58 laps), 1 lap down
        @test r.lo ≈ 100 * log(58 / 57) && r.hi ≈ 100 * log(58 / 56)
        @test all(g.lo .< g.hi)
        @test nrow(prepare_gaps(res; include_lapped = false).rows) == nrow(timed)
    end

    @testset "logdiffΦ" begin
        for (a, b) in ((-1.0, 1.0), (2.0, 3.0), (8.0, 8.5), (-9.0, -8.0), (30.0, 31.0), (-0.5, 40.0),
                       (-31.0, -30.0), (0.0, 1e-3))
            @test Diomedes.logdiffΦ(a, b) ≈ refdiff(a, b) rtol = 1e-8
        end
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
        H = Turing.truncated(Normal(0, 2); lower = 0)
        lp = logpdf(H, θ.σ_comp) + logpdf(H, θ.σ_mach) + logpdf(H, θ.σ_y) +
             sum(logpdf.(Normal(), θ.z_comp)) + sum(logpdf.(Normal(), θ.z_mach)) +
             sum(logpdf.(Normal(0, 5), θ.γ))
        for i in eachindex(g.y)
            lp += logpdf(Normal(θ.γ[g.t_race[i]] + a[g.t_comp[i]] + b[g.t_mach[i]], θ.σ_y), g.y[i])
        end
        for i in eachindex(g.lo)
            μ = θ.γ[g.c_race[i]] + a[g.c_comp[i]] + b[g.c_mach[i]]
            lp += refdiff((g.lo[i] - μ) / θ.σ_y, (g.hi[i] - μ) / θ.σ_y)
        end
        model = gap_effects(g)
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
        # a compiled tape would replay stale gradients from the custom rule, so it is refused
        @test_throws ArgumentError fit_gaps(g; sampler = Diomedes.default_sampler(), n_samples = 10)
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
        # turn gaps into times on a 60-lap race; cars > ~1 lap behind become lapped
        L, lap = 60, 90_000.0
        rows.time_ms = Vector{Union{Missing,Float64}}(undef, nrow(rows))
        rows.laps = Vector{Union{Missing,Int}}(undef, nrow(rows))
        for grp in groupby(rows, [:event_id])
            T = L * lap .* exp.(grp.gap ./ 100 .* 40)       # amplify so some cars get lapped
            Tw = minimum(T)
            for i in eachindex(T)
                l = min(L, floor(Int, L * Tw / T[i]) + (T[i] == Tw ? 0 : 1))
                l = T[i] <= Tw * L / (L - 1) ? L : floor(Int, L * Tw / T[i]) + 1
                grp.laps[i] = l
                grp.time_ms[i] = l == L ? T[i] : missing
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
        @test occursin("iter 150/450", String(take!(log)))   # 300 kept + 150 adaptation
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

    opt_in("DIOMEDES_NETWORK_TESTS") && @testset "network: WRCTiming" begin
        res = fetch_results(WRCTiming(; cache_dir = mktempdir()), 2025)
        @test nrow(res) > 1000
        @test length(unique(res.event_id)) > 10
        @test count(!ismissing, res.time_ms) > 0.8 * nrow(res)
    end
end
