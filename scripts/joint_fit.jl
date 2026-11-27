# The joint race + lap model (#128): one chain, or a combine of saved chains.
#
#   julia --project scripts/joint_fit.jl chain <k> [draws=1000] [warmup=500] [block=25] [delta=0.8] [probe]
#   julia --project scripts/joint_fit.jl combine [<n>=4] [probe]
#   julia --project scripts/joint_fit.jl bench [n=10]
#
# The race part is the published race model (`durc`) on 1950–2020, so no race is in both
# parts. The lap part is the lap model of #159 on 2021–2026 (recorded pit stops, reconciled;
# `cond wetallg wetrace racedegc tyreage pitfix keepslow keepwet rwrel`, no age curve).
# TODO(#128 step 5): switch the lap part to every lap (`alllaps`), so both parts measure a
# share of total time.
#
# Chain k starts at the race model's starting values (`paceloss_init_values`; the published
# race fit covers other seasons) and the last draw of chain k of the lap fit (`joint_init`).
# It is saved in blocks (`run_nuts_checkpointed`) under output/joint_<tag>_chain<k>.
# `bench` times one log density and gradient of the joint model (cost and hard stops).
using Diomedes, Statistics, Printf, Random, Serialization, CSV, DataFrames
import Turing
import MCMCDiagnosticTools as MDT
include("gap_data.jl")

const RACE_MODE = "jolpica1950-2020+partial"
const RACE_MODEL = "pl_durc_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc"
const LAP_SEASONS = 2021:2026
const LAP_TAG = "laps_2021-2026_recorded_slopes_lead_tyremgmt_starts_circ_track_dinc_big_rwalk_nostartdrv_rwrel_traffic_fpace_short_cond_wetallg_wetrace_racedegc_tyreage_pitfix_keepslow_keepwet"

opt(k, d) = (i = findfirst(a -> startswith(a, k * "="), ARGS); i === nothing ? d : String(split(ARGS[i], "="; limit = 2)[2]))
args = filter(a -> !occursin('=', a) && a != "probe", ARGS)
action = get(args, 1, "bench")
probe = "probe" in ARGS
tag = "joint_" * RACE_MODEL * (probe ? "_probe" : "")

function race_model()
    _, g = gap_data(RACE_MODE; io = devnull)
    s = model_spec(RACE_MODEL)
    rm = paceloss_effects(g; loss_duration = s.loss_duration, era = s.era, pace_scale = s.pace_scale, age = s.age,
                          big_loss = s.big_loss, slopes = s.slopes, dev = s.dev, centred_drivers = s.centred_drivers,
                          driver_ν = s.driver_ν, race_hier = s.race_hier, race_mean = s.race_mean, circ = s.circ,
                          track = s.track, walk = s.walk, walk1 = s.walk1, works = s.works, dinc = s.dinc, durc = s.durc)
    return g, rm, s
end

function lap_model()
    jol = JolpicaF1()
    ss = collect(LAP_SEASONS)
    res, laps, pits = fetch_results(jol, ss), fetch_laps(jol, ss), fetch_pit_stops(jol, ss)
    lc = read_lap_conditions()
    nums = jolpica_driver_numbers(jol, ss)
    rec = reconcile_pit_stops(pits, lc, nums)
    d = prepare_laps(laps, res; pits = :recorded, pit_stops = rec.stops, circuits = circuit_table(), stoppage = false,
                     keep_laps = Diomedes.wet_green_laps(lc))
    ctx = LapContext(d, laps; traffic = true, fpace = true, short = true)
    cnd = LapConditions(d, lc, nums; wetdev = true, wetcar = true, wetmin = 1, wetglobal = true, wetrace = true,
                        racedeg = true, racedegc = true, tyreage = true, noage = rec.noage)
    lm = lap_effects(d; slopes = true, lead = true, tyre_mgmt = true, starts = true, age = false, circuits = true,
                     track = true, dinc = true, big = true, rwalk = true, sdrv = false, ctx, cond = cnd, rwrel = true)
    return d, lm
end

# chain k's draws of a fit saved in blocks
function block_chain_k(prefix, k)
    fs = sort(filter(f -> occursin(Regex("^\\Q$(prefix)_chain$(k)_block\\E\\d+\\.jls\$"), f), readdir("output")),
              by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
    isempty(fs) && error("no saved blocks for $(prefix)_chain$(k)")
    return reduce(vcat, [Diomedes.block_chain(joinpath("output", f)) for f in fs])
end

if action == "bench"
    g, rm, _ = race_model()
    d, lm = lap_model()
    jd = JointDrivers(g, d)
    jm = joint_effects(rm, lm, jd)
    LDF = Turing.DynamicPPL.LogDensityFunction
    LDP = Turing.DynamicPPL.LogDensityProblems
    f = LDF(jm, Turing.DynamicPPL.getlogjoint_internal, Turing.DynamicPPL.LinkAll(); adtype = Diomedes.gap_sampler().adtype)
    x = 0.1 .* randn(Xoshiro(1), LDP.dimension(f))
    for _ in 1:2; LDP.logdensity_and_gradient(f, x); end
    n = parse(Int, opt("n", "10"))
    t = @elapsed for _ in 1:n; LDP.logdensity_and_gradient(f, x); end
    @printf("joint model: %d parameters (%s); gradient %.0f ms (mean of %d)\n", LDP.dimension(f), string(jd), 1000t / n, n)
    @printf("hard stop for iteration 100: 100 × 1023 × %.3f s × 1.05 = %.1f h\n", t / n, 100 * 1023 * t / n * 1.05 / 3600)
elseif action == "chain"
    k = parse(Int, args[2])
    g, rm, s = race_model()
    d, lm = lap_model()
    jd = JointDrivers(g, d)
    jm = joint_effects(rm, lm, jd)
    r0 = Diomedes.paceloss_init_values(g, rm; rng = Xoshiro(k), loss_duration = s.loss_duration, era = s.era,
                                       pace_scale = s.pace_scale, age = s.age, big_loss = s.big_loss, slopes = s.slopes,
                                       dev = s.dev, race_mean = s.race_mean, circ = s.circ, track = s.track, walk = s.walk,
                                       walk1 = s.walk1, works = s.works, dinc = s.dinc)
    lc = block_chain_k(LAP_TAG, k)
    init = joint_init(r0, Diomedes.draw_params(lc), jd)
    println("joint chain $k: $(jd); race part from its starting values, lap part from lap chain $k's last draw"); flush(stdout)
    mkpath("output")
    t = @elapsed ch = Diomedes.run_nuts_checkpointed(jm, init, "output/$(tag)_chain$(k)";
                                                     n_samples = parse(Int, opt("draws", "1000")),
                                                     n_adapts = parse(Int, opt("warmup", "500")),
                                                     δ = parse(Float64, opt("delta", "0.8")), block = parse(Int, opt("block", "25")),
                                                     seed = k, progress_log = stdout)
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw $(round(mean(ch[:n_steps]); digits = 1))")
elseif action == "combine"
    n = parse(Int, get(args, 2, "4"))
    chs = [block_chain_k(tag, k) for k in 1:n]
    m = minimum(size(c[:σ_a], 1) for c in chs)
    chs = [c[iter = 1:m] for c in chs]
    arr(c, p) = reduce(hcat, [vec(Array(x[p])) for x in c])
    println("$(n) chains × $(m) draws")
    for p in (:σ_a, :τ_r, :τ_l)
        A = arr(chs, p)
        @printf("%-4s %.3f ± %.3f (90%% %.3f to %.3f; R-hat %.3f, ESS %.0f)\n", p, mean(A), std(A), quantile(vec(A), 0.05),
                quantile(vec(A), 0.95), MDT.rhat(reshape(A, size(A)..., 1))[1], MDT.ess(reshape(A, size(A)..., 1))[1])
    end
    g, _, _ = race_model()
    d, _ = lap_model()
    jd = JointDrivers(g, d)
    # shared ability: σ_a · z_a, centred over every driver (share of time)
    A = reduce(vcat, [permutedims(stack(vec(c[:z_a]))) .* vec(Array(c[:σ_a])) for c in chs])
    A .-= mean(A; dims = 2)
    out = DataFrame(driver = jd.ids, mean = vec(mean(A; dims = 1)), sd = vec(std(A; dims = 1)),
                    in_race = in.(eachindex(jd.ids), Ref(Set(jd.race_idx))), in_laps = in.(eachindex(jd.ids), Ref(Set(jd.lap_idx))))
    CSV.write("output/$(tag)_shared_drivers.csv", out)
    println("wrote output/$(tag)_shared_drivers.csv")
else
    error("unknown action $action (chain, combine, bench)")
end
