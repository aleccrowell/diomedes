# Fit the lap-time model (#12) on F1 lap times (Jolpica, 1996 on).
#
#   julia --project scripts/lap_fit.jl <first>-<last> <recorded|inferred> chain <k> [warm=<first>-<last>]
#                                                       # one chain, checkpointed; optionally warm-started
#   julia --project scripts/lap_fit.jl <first>-<last> <recorded|inferred> combine [n=4 | k1,k2,...]
#
# `recorded` excludes the in/out laps of Ergast's recorded pit stops (2011 on);
# `inferred` infers stops from the lap times (`infer_pit_stops`). Chains are
# saved in blocks of 100 draws under output/laps_<first>-<last>_<pits>_chain<k>;
# rerun a killed chain to resume. `combine` prints convergence and writes
# output/laps_<first>-<last>_<pits>_{drivers,cars}.csv. With LAP_MAX_BLOCKS=k only the
# first k blocks of each chain are used. A `notyres` argument (anywhere) fits
# without the stint-age terms, `slopes` adds per-driver career slopes and `dev`
# in-season car development (tags laps_<first>-<last>_<pits>[_notyres][_slopes][_dev]);
# a warm start comes from an earlier fit over the warm seasons: one with the same terms,
# else the same without age/circuits, else also without starts, else also without
# lead/tyremgmt, else the plain fit (with tyres, without slopes or dev).
# `combine` also writes output/<tag>_slopes.csv and <tag>_dev.csv for those terms.
# `lead` adds the lead-lap effect and `tyremgmt` driver and car-season tyre-slope
# deviations (#66; tags …_lead, …_tyremgmt; combine writes <tag>_tyremgmt_{drivers,cars}.csv);
# `starts` adds lap 1 with driver and car start effects and a grid-slot curve (…_starts;
# combine writes <tag>_starts_{drivers,cars}.csv).
# `delta=<x>` (chain) sets NUTS's target acceptance rate (default 0.8; see fit_laps, #62).
# After warm-up the step size is raised to the largest that still meets that target
# (`tune_step_size`, #62; output/<tag>_chain<k>_steptune.txt); `notune` turns it off.
# `step=<x>` (chain) samples the blocks not yet saved at step size x (e.g. to resume a
# chain whose tuned step size left it at a deeper tree than the others; #66).
# `draws=<n>` (chain) samples n draws after warm-up (default 1000), e.g. for a short test fit.
# `age` adds the population age curve (#12; …_age; combine writes <tag>_age_curve.csv) and
# `circuits` the car-season slopes on circuit speed, street score and race temperature
# (#14; …_circ; covariates from reference/circuits via `circuit_table`; combine writes
# <tag>_circuits.csv).
# #108, terms learned from the race model: `track` walls, mixed and wet races on the
# incident probability and loss size (…_track); `dinc` per-driver incident proneness
# (…_dinc; combine writes <tag>_incidents.csv); `walk` chassis and engine walks over
# seasons, with `works` the works-team term (…_walk, …_works; cars.csv then holds each
# car-season's total). The 2026 races so far: seasons 1996-2026.

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics, Turing
using Diomedes: sum_to_zero, sum_to_zero_by, lap_mach_counts
import MCMCDiagnosticTools

tyres = "notyres" ∉ ARGS
slopes = "slopes" ∈ ARGS
dev = "dev" ∈ ARGS
lead = "lead" ∈ ARGS
tyre_mgmt = "tyremgmt" ∈ ARGS
starts = "starts" ∈ ARGS
age = "age" ∈ ARGS
circuits = "circuits" ∈ ARGS
track = "track" ∈ ARGS            # walls, mixed and wet races on the incident terms (#108)
dinc = "dinc" ∈ ARGS              # per-driver incident proneness (#108)
walk = "walk" ∈ ARGS              # chassis and engine walks (#108)
works = "works" ∈ ARGS            # works-team term (with walk, #108)
δarg = findfirst(a -> startswith(a, "delta="), ARGS)
δ = δarg === nothing ? 0.8 : parse(Float64, split(ARGS[δarg], "=")[2])
tune_step = "notune" ∉ ARGS
steparg = findfirst(a -> startswith(a, "step="), ARGS)
step_kw = steparg === nothing ? (;) : (; fixed_step = parse(Float64, split(ARGS[steparg], "=")[2]))
drawsarg = findfirst(a -> startswith(a, "draws="), ARGS)
n_draws = drawsarg === nothing ? 1000 : parse(Int, split(ARGS[drawsarg], "=")[2])
args = filter(a -> a ∉ ("notyres", "slopes", "dev", "lead", "tyremgmt", "starts", "age", "circuits", "track", "dinc",
                         "walk", "works", "notune") &&
                   !startswith(a, "delta=") && !startswith(a, "step=") && !startswith(a, "draws="), ARGS)
m = match(r"^(\d{4})-(\d{4})$", args[1])
seasons = parse(Int, m[1]):parse(Int, m[2])
pits = Symbol(args[2])
action = get(args, 3, "combine")
# tag suffix of a set of terms: (tyres, slopes, dev, lead, tyre_mgmt, starts, age, circuits, track, dinc, walk, works)
flag_sfx(f) = (f[1] ? "" : "_notyres") * (f[2] ? "_slopes" : "") * (f[3] ? "_dev" : "") * (f[4] ? "_lead" : "") *
              (f[5] ? "_tyremgmt" : "") * (f[6] ? "_starts" : "") * (f[7] ? "_age" : "") * (f[8] ? "_circ" : "") *
              (f[9] ? "_track" : "") * (f[10] ? "_dinc" : "") * (f[11] ? "_walk" : "") * (f[12] ? "_works" : "")
flags = (tyres, slopes, dev, lead, tyre_mgmt, starts, age, circuits, track, dinc, walk, works)
tag = "laps_$(args[1])_$(pits)" * flag_sfx(flags)
# All seasons from the Jolpica API (#55: one source; its lap times match the Ergast
# dump's on 1996–2019 but for 76 of 472,504 laps). Run scripts/fetch_jolpica.jl first
# (laps_from=1996) so the responses are cached; `recorded` pit stops are fetched on
# first use.
jolpica = JolpicaF1()
src_data(ss) = (res = fetch_results(jolpica, collect(ss)), laps = fetch_laps(jolpica, collect(ss)),
                pits = pits === :recorded ? fetch_pit_stops(jolpica, collect(ss)) : nothing)
sd = src_data(seasons)
res = sd.res
circ = circuits || track ? circuit_table() : nothing
d = prepare_laps(sd.laps, res; pits, pit_stops = sd.pits, circuits = circ)
println(d); flush(stdout)
mkpath("output")

if action == "chain"
    k = parse(Int, args[4])
    # warm=<first>-<last>: start from the adapted metric and step size of chain k of that
    # (finished, same-pits) fit, and adapt for 300 iterations instead of 500
    warm = match(r"^warm=(\d{4})-(\d{4})$", get(args, 5, ""))
    ws, n_adapts = nothing, 500
    if warm !== nothing
        old_seasons = parse(Int, warm[1]):parse(Int, warm[2])
        # the old fit: the first that exists of: the same terms; the same without the
        # age curve and circuit slopes; also without starts; also without the lead-lap and
        # tyre-management terms (#66); the plain fit (tyres, nothing else).
        # A fit with the same terms has adapted every block, so it is the best start.
        chains_of(t) = sort(unique(parse(Int, m[1]) for f in readdir("output")
                                   for m in (match(Regex("^$(t)_chain(\\d+)_(block|warmstate)"), f),) if m !== nothing))
        stem = "laps_$(warm[1])-$(warm[2])_$(pits)"
        F4 = (false, false, false, false)       # without the #108 terms
        cf = [flags, (flags[1:8]..., F4...), (flags[1:6]..., false, false, F4...), (flags[1:5]..., false, false, false, F4...),
              (flags[1:3]..., false, false, false, false, false, F4...),
              (true, false, false, false, false, false, false, false, F4...)]
        cands = [(f, flag_sfx(f)) for f in cf]
        old_flags, sfx = cands[something(findfirst(c -> !isempty(chains_of(stem * c[2])), cands), length(cands))]
        base = stem * sfx
        # chain k of the old fit, or (if it has fewer chains) one of them in turn
        have = chains_of(base)
        old_tag = "$(base)_chain$(k in have ? k : have[mod1(k, length(have))])"
        # the adapted metric and step size: from <old_tag>_warmstate.jls if it exists
        # (written after each chain; it survives changes to LapData, which the
        # checkpoint blocks embed and then can't be read back), else the last block
        wsfile = joinpath("output", old_tag * "_warmstate.jls")
        if isfile(wsfile)
            st = deserialize(wsfile)                        # (; Minv, ϵ)
            last_block = basename(wsfile)
        else
            blocks = filter(f -> startswith(f, old_tag * "_block"), readdir("output"))
            last_block = argmax(f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]), blocks)
            st = (b = Diomedes.read_block(joinpath("output", last_block)); (; Minv = b.Minv, ϵ = b.ϵ))
        end
        od = src_data(old_seasons)
        d_old = prepare_laps(od.laps, od.res; pits, pit_stops = od.pits, circuits = old_flags[8] || old_flags[9] ? circ : nothing)
        ws, n_adapts = lap_warm_start(st, d_old, d; tyres_old = old_flags[1], slopes_old = old_flags[2], dev_old = old_flags[3],
                                      lead_old = old_flags[4], tyre_mgmt_old = old_flags[5], starts_old = old_flags[6],
                                      age_old = old_flags[7], circuits_old = old_flags[8],
                                      starts_new = starts, age_new = age, circuits_new = circuits,
                                      tyres_new = tyres, slopes_new = slopes, dev_new = dev, lead_new = lead,
                                      tyre_mgmt_new = tyre_mgmt, track_old = old_flags[9], dinc_old = old_flags[10],
                                      walk_old = old_flags[11], works_old = old_flags[12], track_new = track,
                                      dinc_new = dinc, walk_new = walk, works_new = works), 300
        println("warm start from $last_block: step size $(round(ws.ϵ; sigdigits = 3))"); flush(stdout)
    end
    t = @elapsed chain = fit_laps(d; rng = Xoshiro(k), checkpoint = "output/$(tag)_chain$(k)", seed = 1000k, n_samples = n_draws,
                                  progress_log = stdout, warm_start = ws, n_adapts, tyres, slopes, dev, lead, tyre_mgmt, starts,
                                  age, circuits, track, dinc, walk, works, δ, tune_step, step_kw...)
    # adapted metric and step size, for warm starts of later fits (see above)
    try
        serialize("output/$(tag)_chain$(k)_warmstate.jls", Diomedes.checkpoint_adapted("output/$(tag)_chain$(k)"))
    catch err
        @warn "could not save the warm-start state" err
    end
    println("chain $k: $(round(t / 60; digits = 1)) min, target acceptance $δ, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", step size ", round(mean(chain[:step_size]); sigdigits = 3),
            ", divergences ", count(chain[:numerical_error] .> 0))
    exit()
end

function load_blocks(prefix)
    dir, base = dirname(prefix), basename(prefix) * "_block"
    files = sort(filter(f -> startswith(f, base), readdir(dir)), by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
    # LAP_MAX_BLOCKS=k combines only the first k blocks of each chain (chains still running
    # can be compared at equal length)
    haskey(ENV, "LAP_MAX_BLOCKS") && (files = files[1:min(end, parse(Int, ENV["LAP_MAX_BLOCKS"]))])
    return reduce(vcat, [Diomedes.block_chain(joinpath(dir, f)) for f in files])
end
arg = get(args, 4, "4")              # chains 1..n, or a comma-separated list (e.g. 2,3,4)
ks = occursin(",", arg) ? parse.(Int, split(arg, ",")) : 1:parse(Int, arg)
chain = reduce(hcat, [load_blocks("output/$(tag)_chain$(k)") for k in ks])
ni, nc = size(chain[:σ])
mc = lap_mach_counts(d)
function collect3(f)
    A = Array{Float64,3}(undef, ni, nc, length(f(1, 1)))
    for c in 1:nc, i in 1:ni; A[i, c, :] = f(i, c); end
    return A
end
drv = collect3((i, c) -> sum_to_zero(chain[:a_comp][i, c]))
cwl = walk ? CarWalks(d) : nothing
function car_total(i, c)            # residual + chassis + engine (+ works), centred within season (#108)
    b = sum_to_zero_by(chain[:b_mach][i, c], d.mach_season, mc)
    walk || return b
    θw = (; τ_ch = chain[:τ_ch][i, c], σ0_ch = chain[:σ0_ch][i, c], z_ch = chain[:z_ch][i, c], τ_en = chain[:τ_en][i, c],
          σ0_en = chain[:σ0_en][i, c], z_en = chain[:z_en][i, c], δ_own = chain[:δ_own][i, c], δ_reg = chain[:δ_reg][i, c])
    xc, xe = car_walk_states(θw, cwl)
    w = Diomedes.walk_offsets(xc, cwl.row_ch, xe, cwl.row_en)
    works && (w = w .+ chain[:β_works][i, c] .* cwl.row_works)
    return b .+ sum_to_zero_by(w, d.mach_season, mc)
end
car = collect3(car_total)
dincv = dinc ? collect3((i, c) -> career_slopes(chain[:τ_dinc][i, c], chain[:u_dinc][i, c])) : nothing   # τ·(u - mean u)
scal_names = (:σ_comp, :σ_mach, :σ, :μ_γ, :μ_δ, :τ_δ, :a_π, :a_λ)
scal = collect3((i, c) -> vcat([chain[s][i, c] for s in scal_names], tyres ? chain[:β][i, c] : Float64[],
                                slopes ? [chain[:τ_slope][i, c]] : Float64[], dev ? [chain[:τ_dev][i, c]] : Float64[],
                                lead ? chain[:η][i, c] : Float64[],
                                tyre_mgmt ? [chain[:τ_td][i, c], chain[:τ_tm][i, c]] : Float64[],
                                starts ? vcat([chain[:τ_sd][i, c], chain[:τ_sc][i, c]], chain[:g1][i, c],
                                              [chain[:σ1][i, c], chain[:a_π1][i, c], chain[:a_λ1][i, c]]) : Float64[],
                                age ? [chain[:τ_age][i, c]] : Float64[], circuits ? chain[:τ_cx][i, c] : Float64[],
                                track ? vcat(chain[:β_track_π][i, c], chain[:β_track_λ][i, c]) : Float64[],
                                dinc ? [chain[:τ_dinc][i, c]] : Float64[],
                                walk ? [chain[s][i, c] for s in (:τ_ch, :σ0_ch, :τ_en, :σ0_en, :δ_own, :δ_reg)] : Float64[],
                                works ? [chain[:β_works][i, c]] : Float64[]))
sdv = starts ? collect3((i, c) -> career_slopes(chain[:τ_sd][i, c], chain[:u_sd][i, c])) : nothing
scv = starts ? collect3((i, c) -> chain[:τ_sc][i, c] .* sum_to_zero_by(chain[:u_sc][i, c], d.mach_season, mc)) : nothing
tdv = tyre_mgmt ? collect3((i, c) -> career_slopes(chain[:τ_td][i, c], chain[:u_td][i, c])) : nothing
tmv = tyre_mgmt ? collect3((i, c) -> chain[:τ_tm][i, c] .* sum_to_zero_by(chain[:u_tm][i, c], d.mach_season, mc)) : nothing
slo = slopes ? collect3((i, c) -> career_slopes(chain[:τ_slope][i, c], chain[:u_slope][i, c])) : nothing
dvt = dev ? collect3((i, c) -> chain[:τ_dev][i, c] .* sum_to_zero_by(chain[:u_dev][i, c], d.mach_season, mc)) : nothing
# population age curve per bin, and the circuit slopes (machine-major per covariate, as in the model)
agebasis = age ? AgeCurveBasis(d.age_years, d.age_rows) : nothing
agev = age ? collect3((i, c) -> age_curve(chain[:τ_age][i, c], chain[:e_age][i, c], agebasis)) : nothing
nmach, nk = length(d.machines), size(d.race_x, 2)
cxv = circuits ? collect3((i, c) -> reduce(vcat, [chain[:τ_cx][i, c][k] .*
                                                  sum_to_zero_by(chain[:u_cx][i, c][((k - 1) * nmach + 1):(k * nmach)],
                                                                 d.mach_season, mc) for k in 1:nk])) : nothing
println("\nconvergence (max R-hat, min bulk ESS):")
for (name, A) in (("drivers", drv), ("cars", car), ("scalars", scal), (slopes ? (("slopes", slo),) : ())..., (dev ? (("dev", dvt),) : ())...,
                  (tyre_mgmt ? (("tyre (drv)", tdv), ("tyre (car)", tmv)) : ())...,
                  (starts ? (("start (drv)", sdv), ("start (car)", scv)) : ())...,
                  (age ? (("age curve", agev),) : ())..., (circuits ? (("circuits", cxv),) : ())...,
                  (dinc ? (("incidents", dincv),) : ())...)
    println(rpad(name, 10), round(maximum(MCMCDiagnosticTools.rhat(A)); digits = 3), "  ",
            round(Int, minimum(MCMCDiagnosticTools.ess(A))))
end
println("divergences: ", count(chain[:numerical_error] .> 0))
flat(A) = reshape(A, ni * nc, :)
println("\nscalars (mean ± sd):")
for (j, nm) in enumerate((string.(scal_names)..., (tyres ? Tuple(n for e in d.eras for n in ("β_stint[$e]", "β_stint²/10[$e]")) : ())..., (slopes ? ("τ_slope",) : ())...,
                          (dev ? ("τ_dev",) : ())..., (lead ? ("η_lead",) : ())..., (tyre_mgmt ? ("τ_td", "τ_tm") : ())...,
                          (starts ? ("τ_sd", "τ_sc", "g_grid", "g_grid²", "σ1", "a_π1", "a_λ1") : ())...,
                          (age ? ("τ_age",) : ())..., (circuits ? Tuple("τ_cx[$n]" for n in d.race_x_names) : ())...,
                          (track ? Tuple("β_track_$(p)[$n]" for p in ("π", "λ") for n in d.race_track_names) : ())...,
                          (dinc ? ("τ_dinc",) : ())..., (walk ? ("τ_ch", "σ0_ch", "τ_en", "σ0_en", "δ_own", "δ_reg") : ())...,
                          (works ? ("β_works",) : ())...))
    v = flat(scal)[:, j]
    println(rpad(nm, 12), round(mean(v); digits = 4), " ± ", round(std(v); digits = 4))
end
names_of = Dict(r.competitor_id => r.competitor_name for r in eachrow(res))
D, C = flat(drv), flat(car)
dt = DataFrame(driver = d.competitors, name = [names_of[c] for c in d.competitors], mean = vec(mean(D; dims = 1)),
               sd = vec(std(D; dims = 1)), laps = [count(==(k), d.comp) for k in eachindex(d.competitors)])
ct = DataFrame(machine = d.machines, mean = vec(mean(C; dims = 1)), sd = vec(std(C; dims = 1)),
               laps = [count(==(k), d.mach) for k in eachindex(d.machines)])
sort!(dt, :mean); sort!(ct, :mean)
CSV.write("output/$(tag)_drivers.csv", dt)
CSV.write("output/$(tag)_cars.csv", ct)
if dinc                              # incident proneness, logit offsets on π (#108)
    I = flat(dincv)
    it = DataFrame(driver = d.competitors, name = [names_of[c] for c in d.competitors],
                   laps = [count(==(k), d.comp) for k in eachindex(d.competitors)],
                   mean = vec(mean(I; dims = 1)), sd = vec(std(I; dims = 1)))
    sort!(it, :mean)
    CSV.write("output/$(tag)_incidents.csv", it)
    println("\nincident proneness (logit offset on π; ≥ 1000 laps):")
    bi = it[it.laps .>= 1000, :]
    show(stdout, MIME"text/plain"(), vcat(first(bi, 6), last(bi, 6))); println()
end
println("\nfastest drivers (≥ 1000 laps):")
show(stdout, MIME"text/plain"(), first(dt[dt.laps .>= 1000, :], 10)); println()
if slopes
    S = flat(slo)
    span(k) = (a = d.rows.season[d.comp .== k]; (minimum(a), maximum(a)))
    st = DataFrame(driver = d.competitors, name = [names_of[c] for c in d.competitors],
                   laps = [count(==(k), d.comp) for k in eachindex(d.competitors)],
                   first_season = [span(k)[1] for k in eachindex(d.competitors)],
                   last_season = [span(k)[2] for k in eachindex(d.competitors)],
                   mean = vec(mean(S; dims = 1)), sd = vec(std(S; dims = 1)))
    sort!(st, :mean)
    CSV.write("output/$(tag)_slopes.csv", st)
    println("\ncareer slopes (% per decade of age, relative to the average driver; ≥ 1000 laps):")
    big = st[st.laps .>= 1000, :]
    show(stdout, MIME"text/plain"(), vcat(first(big, 6), last(big, 6))); println()
end
if dev
    V = flat(dvt)
    dtab = DataFrame(machine = d.machines, laps = [count(==(k), d.mach) for k in eachindex(d.machines)],
                     mean = vec(mean(V; dims = 1)), sd = vec(std(V; dims = 1)))
    sort!(dtab, :mean)
    CSV.write("output/$(tag)_dev.csv", dtab)
    println("\nin-season development (% from first round to last, relative to the field; ≥ 500 laps):")
    bigd = dtab[dtab.laps .>= 500, :]
    show(stdout, MIME"text/plain"(), vcat(first(bigd, 6), last(bigd, 6))); println()
end
if tyre_mgmt
    # deviations from the shared tyre slope, % per 10 laps of stint age (negative =
    # degrades less than the average driver / car of its season)
    TD, TM = flat(tdv), flat(tmv)
    tdt = DataFrame(driver = d.competitors, name = [names_of[c] for c in d.competitors],
                    laps = [count(==(k), d.comp) for k in eachindex(d.competitors)],
                    mean = vec(mean(TD; dims = 1)), sd = vec(std(TD; dims = 1)))
    tmt = DataFrame(machine = d.machines, laps = [count(==(k), d.mach) for k in eachindex(d.machines)],
                    mean = vec(mean(TM; dims = 1)), sd = vec(std(TM; dims = 1)))
    sort!(tdt, :mean); sort!(tmt, :mean)
    CSV.write("output/$(tag)_tyremgmt_drivers.csv", tdt)
    CSV.write("output/$(tag)_tyremgmt_cars.csv", tmt)
    println("\ntyre management, drivers (% per 10 laps vs the shared slope; ≥ 1000 laps):")
    bt = tdt[tdt.laps .>= 1000, :]
    show(stdout, MIME"text/plain"(), vcat(first(bt, 6), last(bt, 6))); println()
    println("\ntyre management, car-seasons (≥ 500 laps):")
    bm = tmt[tmt.laps .>= 500, :]
    show(stdout, MIME"text/plain"(), vcat(first(bm, 6), last(bm, 6))); println()
end
if starts
    # start effects: lap 1 against the driver's / car's pace, % of the race's median
    # lap-1 time (negative = gains on lap 1)
    SD, SC = flat(sdv), flat(scv)
    sdt = DataFrame(driver = d.competitors, name = [names_of[c] for c in d.competitors],
                    starts = [count(==(k), d.s_comp) for k in eachindex(d.competitors)],
                    mean = vec(mean(SD; dims = 1)), sd = vec(std(SD; dims = 1)))
    sct = DataFrame(machine = d.machines, starts = [count(==(k), d.s_mach) for k in eachindex(d.machines)],
                    mean = vec(mean(SC; dims = 1)), sd = vec(std(SC; dims = 1)))
    sort!(sdt, :mean); sort!(sct, :mean)
    CSV.write("output/$(tag)_starts_drivers.csv", sdt)
    CSV.write("output/$(tag)_starts_cars.csv", sct)
    println("\nstarts, drivers (% of lap 1 against their pace; ≥ 30 starts):")
    bs = sdt[sdt.starts .>= 30, :]
    show(stdout, MIME"text/plain"(), vcat(first(bs, 6), last(bs, 6))); println()
end
if age
    # the population curve's shape (no constant or linear part, see AgeCurveBasis), % per age
    AG = flat(agev)
    at = DataFrame(age = d.age_years, laps = Int.(d.age_rows), mean = vec(mean(AG; dims = 1)), sd = vec(std(AG; dims = 1)),
                   q05 = [quantile(c, 0.05) for c in eachcol(AG)], q95 = [quantile(c, 0.95) for c in eachcol(AG)])
    CSV.write("output/$(tag)_age_curve.csv", at)
    println("\npopulation age curve (% per age, shape only):")
    show(stdout, MIME"text/plain"(), at); println()
end
if circuits
    # car-season slopes per standard deviation of each circuit covariate (negative =
    # relatively faster where the covariate is high: faster circuits, street circuits,
    # hotter races), with the covariates' scale
    CX = flat(cxv)
    sds = [std(d.race_x_raw[:, k]) for k in 1:nk]
    cxt = DataFrame(machine = d.machines, laps = [count(==(m), d.mach) for m in eachindex(d.machines)])
    for (k, n) in enumerate(d.race_x_names)
        cols = ((k - 1) * nmach + 1):(k * nmach)
        cxt[!, Symbol(n, "_mean")] = vec(mean(CX[:, cols]; dims = 1))
        cxt[!, Symbol(n, "_sd")] = vec(std(CX[:, cols]; dims = 1))
    end
    CSV.write("output/$(tag)_circuits.csv", cxt)
    println("\ncircuit covariates: ", join(["$(n) sd $(round(sds[k]; sigdigits = 3))" for (k, n) in enumerate(d.race_x_names)], ", "))
    for n in d.race_x_names
        println("\ncar-seasons by $(n) slope (% per sd; ≥ 500 laps):")
        bc = sort(cxt[cxt.laps .>= 500, [:machine, :laps, Symbol(n, "_mean"), Symbol(n, "_sd")]], Symbol(n, "_mean"))
        show(stdout, MIME"text/plain"(), vcat(first(bc, 5), last(bc, 5))); println()
    end
end
