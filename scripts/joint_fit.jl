# The joint race + lap model (#128): one chain, or a combine of saved chains.
#
#   julia --project scripts/joint_fit.jl chain <k> [draws=1000] [warmup=500] [block=25] [delta=0.8] [probe]
#   julia --project scripts/joint_fit.jl combine [<n>=4] [probe]
#   julia --project scripts/joint_fit.jl bench [n=10]
#
# The race part is the published race model (`durc`) on 1950–2020, so no race is in both
# parts. The lap part is the lap model of #159 on 2021–2026 (recorded pit stops, reconciled;
# `cond wetallg wetrace racedegc tyreage pitfix keepslow keepwet rwrel`, no age curve), on every
# lap (`alllaps`): the laps the clean-lap model leaves out come in with their own shifts, and the
# drivers' and cars' effects are on every lap, so both parts measure a share of total time.
# Incidents are the race model's process in both parts: at most one per driver-race, its size in
# % of race time landing on one lap (`incident`, in place of the lap model's big losses and
# incident proneness), with the drivers' offsets shared and each part's own baseline and sizes.
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
alllaps = opt("alllaps", "true") == "true"     # alllaps=false: the lap part on clean laps only (diagnostics)
tag = "joint_" * RACE_MODEL * "_inc" * (alllaps ? "" : "_cleanlaps") * (probe ? "_probe" : "")

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
                     keep_laps = Diomedes.wet_green_laps(lc), alllaps)
    ctx = LapContext(d, laps; traffic = true, fpace = true, short = true)
    cnd = LapConditions(d, lc, nums; wetdev = true, wetcar = true, wetmin = 1, wetglobal = true, wetrace = true,
                        racedeg = true, racedegc = true, tyreage = true, noage = rec.noage)
    kw = (; slopes = true, lead = true, tyre_mgmt = true, starts = true, age = false, circuits = true, track = true,
          dinc = true, big = true, rwalk = true, sdrv = false, ctx, cond = cnd, rwrel = true)
    # its age curve, slopes and incident offsets come from the joint model's shared ones; the race model's
    # incident process (`incident`) replaces the lap model's big losses and incident proneness
    lm = lap_effects(d; kw..., age = true, alllaps, big = false, dinc = false, incident = true)
    # the lap model as the lap fit had it (clean laps only): its layout maps that fit's adapted metric
    lm_std = lap_effects(d; kw...)
    return d, lm, lm_std
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
    d, lm, lm_std = lap_model()
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
    d, lm, lm_std = lap_model()
    jd = JointDrivers(g, d)
    jm = joint_effects(rm, lm, jd)
    r0 = Diomedes.paceloss_init_values(g, rm; rng = Xoshiro(k), loss_duration = s.loss_duration, era = s.era,
                                       pace_scale = s.pace_scale, age = s.age, big_loss = s.big_loss, slopes = s.slopes,
                                       dev = s.dev, race_mean = s.race_mean, circ = s.circ, track = s.track, walk = s.walk,
                                       walk1 = s.walk1, works = s.works, dinc = s.dinc)
    lc = block_chain_k(LAP_TAG, k)
    # the lap fit's last draw, with the other laps' terms (not in that fit) started from the laps
    l0 = alllaps ? merge(Diomedes.draw_params(lc), Diomedes.other_laps_init(d, lm.args.xl)) : Diomedes.draw_params(lc)
    init = joint_init(r0, l0, jd; incident = true)
    # warm-up from a metric with the lap fit's adapted variances on the lap part (see joint_warm_metric)
    la = Diomedes.checkpoint_adapted("output/$(LAP_TAG)_chain$(k)")
    M0 = Diomedes.joint_warm_metric(jm, lm_std, la.Minv)
    δ = parse(Float64, opt("delta", "0.8"))
    external = lap_warm_sampler(M0, 0.1 * la.ϵ; δ)
    println("joint chain $k: $(jd); race part from its starting values, lap part from lap chain $k's last draw; ",
            "metric warm on the lap part"); flush(stdout)
    mkpath("output")
    t = @elapsed ch = Diomedes.run_nuts_checkpointed(jm, init, "output/$(tag)_chain$(k)";
                                                     n_samples = parse(Int, opt("draws", "1000")),
                                                     n_adapts = parse(Int, opt("warmup", "500")),
                                                     δ, block = parse(Int, opt("block", "25")), seed = k, progress_log = stdout,
                                                     external)
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw $(round(mean(ch[:n_steps]); digits = 1))")
elseif action == "check"
    # the starting point of chain k: log density by part, and the gradient's largest entries
    k = parse(Int, get(args, 2, "2"))
    g, rm, s = race_model()
    d, lm, lm_std = lap_model()
    jd = JointDrivers(g, d)
    jm = joint_effects(rm, lm, jd)
    r0 = Diomedes.paceloss_init_values(g, rm; rng = Xoshiro(k), loss_duration = s.loss_duration, era = s.era,
                                       pace_scale = s.pace_scale, age = s.age, big_loss = s.big_loss, slopes = s.slopes,
                                       dev = s.dev, race_mean = s.race_mean, circ = s.circ, track = s.track, walk = s.walk,
                                       walk1 = s.walk1, works = s.works, dinc = s.dinc)
    l0 = merge(Diomedes.draw_params(block_chain_k(LAP_TAG, k)), Diomedes.other_laps_init(d, lm.args.xl))
    v = Diomedes.joint_init_values(r0, l0, jd; incident = true)
    DP = Turing.DynamicPPL
    vi = DP.VarInfo(jm)
    vi = last(DP.init!!(jm, vi, Turing.InitFromParams(v), DP.UnlinkAll()))
    @printf("log prior %.1f, log likelihood %.1f\n", DP.getlogprior(vi), DP.getloglikelihood(vi))
    rv = first(DP.evaluate!!(jm, vi))                 # the model's return value at the starting point
    @printf("race part log lik %.1f; lap part log lik %.1f\n",
            DP.loglikelihood(Diomedes.race_given(jm.args.rm, rv.a_race, rv.fage, jd.Sr * rv.sl, rv.oir), v.race),
            DP.loglikelihood(Diomedes.lap_given(jm.args.lm, rv.a_lap, jd.Sl * rv.sl, rv.fage[jm.args.lapage], rv.oil), v.lap))
    xl = lm.args.xl
    println("other laps: ", length(d.x_y), " in ", length(d.group_type), " groups; types ",
            [Diomedes.LAP_TYPES[t] for t in xl.types], "; μ_x ", round.(v.lap.μ_x; digits = 1), "; σ_x ", round.(v.lap.σ_x; digits = 2))
    f = DP.LogDensityFunction(jm, DP.getlogjoint_internal, DP.LinkAll(); adtype = Diomedes.gap_sampler().adtype)
    vil = DP.link!!(deepcopy(vi), jm)
    x = vil[:]
    lp, gr = DP.LogDensityProblems.logdensity_and_gradient(f, x)
    println("linked log density $(round(lp; digits = 1)); gradient finite: $(all(isfinite, gr)); max |grad| $(round(maximum(abs, gr); digits = 1))")
    rows = let pos = 0, rows = Tuple{String,Float64,Int}[]
        for vn in keys(vil)
            n = length(DP.getindex_internal(vil, vn))
            push!(rows, (string(vn), maximum(abs, gr[(pos + 1):(pos + n)]), n)); pos += n
        end
        rows
    end
    sort!(rows; by = r -> -r[2])
    for r in rows[1:min(12, end)]; @printf("  %-22s max |grad| %10.1f (%d)\n", r...); end
elseif action == "combine"
    n = parse(Int, get(args, 2, "4"))
    chs = [block_chain_k(tag, k) for k in 1:n]
    m = minimum(size(c[:σ_a], 1) for c in chs)
    chs = [c[iter = 1:m] for c in chs]
    arr(c, p) = reduce(hcat, [vec(Array(x[p])) for x in c])
    println("$(n) chains × $(m) draws")
    for p in (:σ_a, :τ_r, :τ_l, :τ_inc)
        P = arr(chs, p)
        @printf("%-5s %.3f ± %.3f (90%% %.3f to %.3f; R-hat %.3f, ESS %.0f)\n", p, mean(P), std(P), quantile(vec(P), 0.05),
                quantile(vec(P), 0.95), MDT.rhat(reshape(P, size(P)..., 1))[1], MDT.ess(reshape(P, size(P)..., 1))[1])
    end
    g, _, _ = race_model()
    d, _, _ = lap_model()
    jd = JointDrivers(g, d)
    # shared ability of every driver (share of time), centred over them
    A = reduce(vcat, [permutedims(stack(vec(c[:a]))) for c in chs])
    A .-= mean(A; dims = 2)
    out = DataFrame(driver = jd.ids, mean = vec(mean(A; dims = 1)), sd = vec(std(A; dims = 1)),
                    in_race = in.(eachindex(jd.ids), Ref(Set(jd.race_idx))), in_laps = in.(eachindex(jd.ids), Ref(Set(jd.lap_idx))))
    CSV.write("output/$(tag)_shared_drivers.csv", out)
    println("wrote output/$(tag)_shared_drivers.csv")
else
    error("unknown action $action (chain, combine, bench)")
end
