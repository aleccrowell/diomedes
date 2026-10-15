# Fit the lap-time model (#12) on F1 lap times (Ergast to 2019, Jolpica from 2020).
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
# else the same without lead/tyremgmt, else the plain fit (with tyres, without slopes or dev).
# `combine` also writes output/<tag>_slopes.csv and <tag>_dev.csv for those terms.
# `lead` adds the lead-lap effect and `tyremgmt` driver and car-season tyre-slope
# deviations (#66; tags …_lead, …_tyremgmt; combine writes <tag>_tyremgmt_{drivers,cars}.csv).

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics, Turing
using Diomedes: sum_to_zero, sum_to_zero_by, lap_mach_counts
import MCMCDiagnosticTools

tyres = "notyres" ∉ ARGS
slopes = "slopes" ∈ ARGS
dev = "dev" ∈ ARGS
lead = "lead" ∈ ARGS
tyre_mgmt = "tyremgmt" ∈ ARGS
args = filter(∉(("notyres", "slopes", "dev", "lead", "tyremgmt")), ARGS)
m = match(r"^(\d{4})-(\d{4})$", args[1])
seasons = parse(Int, m[1]):parse(Int, m[2])
pits = Symbol(args[2])
action = get(args, 3, "combine")
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
tag = "laps_$(args[1])_$(pits)" * (tyres ? "" : "_notyres") * (slopes ? "_slopes" : "") * (dev ? "_dev" : "") *
      (lead ? "_lead" : "") * (tyre_mgmt ? "_tyremgmt" : "")
# Seasons to 2019 come from the local Ergast dump; later ones from the Jolpica API
# (#55; same ids, laps checked equal on 2019-01). Run scripts/fetch_jolpica.jl
# first so the Jolpica responses are cached.
const ERGAST_LAST = 2019
ergast, jolpica = ErgastCSV(data_dir), JolpicaF1()
function by_source(f, ss)
    parts = [f(ergast, filter(<=(ERGAST_LAST), collect(ss))), f(jolpica, filter(>(ERGAST_LAST), collect(ss)))]
    return reduce(vcat, filter(p -> nrow(p) > 0, parts); init = first(parts)[1:0, :])
end
src_data(ss) = (res = by_source(fetch_results, ss), laps = by_source(fetch_laps, ss),
                pits = pits === :recorded ? by_source(fetch_pit_stops, ss) : nothing)
sd = src_data(seasons)
res = sd.res
d = prepare_laps(sd.laps, res; pits, pit_stops = sd.pits)
println(d); flush(stdout)
mkpath("output")

if action == "chain"
    k = parse(Int, args[4])
    # warm=<first>-<last>: start from the adapted metric and step size of chain k of that
    # (finished, same-pits) fit, and adapt for 300 iterations instead of 500
    warm = something(match(r"^warm=(\d{4})-(\d{4})$", get(args, 5, "")), nothing)
    ws, n_adapts = nothing, 500
    if warm !== nothing
        old_seasons = parse(Int, warm[1]):parse(Int, warm[2])
        # the old fit: the first that exists of: the same terms; the same without the
        # lead-lap and tyre-management terms (#66); the plain fit (tyres, nothing else).
        # A fit with the same terms has adapted every block, so it is the best start.
        chains_of(t) = sort(unique(parse(Int, m[1]) for f in readdir("output")
                                   for m in (match(Regex("^$(t)_chain(\\d+)_(block|warmstate)"), f),) if m !== nothing))
        stem = "laps_$(warm[1])-$(warm[2])_$(pits)"
        cands = [((tyres, slopes, dev, lead, tyre_mgmt), (tyres ? "" : "_notyres") * (slopes ? "_slopes" : "") * (dev ? "_dev" : "") *
                                                         (lead ? "_lead" : "") * (tyre_mgmt ? "_tyremgmt" : "")),
                 ((tyres, slopes, dev, false, false), (tyres ? "" : "_notyres") * (slopes ? "_slopes" : "") * (dev ? "_dev" : "")),
                 ((true, false, false, false, false), "")]
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
            st = only(Turing.loadstate(rehash_chain!(deserialize(joinpath("output", last_block)))))
        end
        od = src_data(old_seasons)
        d_old = prepare_laps(od.laps, od.res; pits, pit_stops = od.pits)
        ws, n_adapts = lap_warm_start(st, d_old, d; tyres_old = old_flags[1], slopes_old = old_flags[2], dev_old = old_flags[3],
                                      lead_old = old_flags[4], tyre_mgmt_old = old_flags[5],
                                      tyres_new = tyres, slopes_new = slopes, dev_new = dev, lead_new = lead,
                                      tyre_mgmt_new = tyre_mgmt), 300
        println("warm start from $last_block: step size $(round(ws.ϵ; sigdigits = 3))"); flush(stdout)
    end
    t = @elapsed chain = fit_laps(d; rng = Xoshiro(k), checkpoint = "output/$(tag)_chain$(k)", seed = 1000k,
                                  progress_log = stdout, warm_start = ws, n_adapts, tyres, slopes, dev, lead, tyre_mgmt)
    # adapted metric and step size, for warm starts of later fits (see above)
    try
        serialize("output/$(tag)_chain$(k)_warmstate.jls", Diomedes.lap_adapted(only(Turing.loadstate(chain))))
    catch err
        @warn "could not save the warm-start state" err
    end
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", divergences ", count(chain[:numerical_error] .> 0))
    exit()
end

function load_blocks(prefix)
    dir, base = dirname(prefix), basename(prefix) * "_block"
    files = sort(filter(f -> startswith(f, base), readdir(dir)), by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
    # LAP_MAX_BLOCKS=k combines only the first k blocks of each chain (chains still running
    # can be compared at equal length)
    haskey(ENV, "LAP_MAX_BLOCKS") && (files = files[1:min(end, parse(Int, ENV["LAP_MAX_BLOCKS"]))])
    return reduce(vcat, [rehash_chain!(deserialize(joinpath(dir, f))) for f in files])
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
car = collect3((i, c) -> sum_to_zero_by(chain[:b_mach][i, c], d.mach_season, mc))
scal_names = (:σ_comp, :σ_mach, :σ, :μ_γ, :μ_δ, :τ_δ, :a_π, :a_λ)
scal = collect3((i, c) -> vcat([chain[s][i, c] for s in scal_names], tyres ? chain[:β][i, c] : Float64[],
                                slopes ? [chain[:τ_slope][i, c]] : Float64[], dev ? [chain[:τ_dev][i, c]] : Float64[],
                                lead ? chain[:η][i, c] : Float64[],
                                tyre_mgmt ? [chain[:τ_td][i, c], chain[:τ_tm][i, c]] : Float64[]))
tdv = tyre_mgmt ? collect3((i, c) -> career_slopes(chain[:τ_td][i, c], chain[:u_td][i, c])) : nothing
tmv = tyre_mgmt ? collect3((i, c) -> chain[:τ_tm][i, c] .* sum_to_zero_by(chain[:u_tm][i, c], d.mach_season, mc)) : nothing
slo = slopes ? collect3((i, c) -> career_slopes(chain[:τ_slope][i, c], chain[:u_slope][i, c])) : nothing
dvt = dev ? collect3((i, c) -> chain[:τ_dev][i, c] .* sum_to_zero_by(chain[:u_dev][i, c], d.mach_season, mc)) : nothing
println("\nconvergence (max R-hat, min bulk ESS):")
for (name, A) in (("drivers", drv), ("cars", car), ("scalars", scal), (slopes ? (("slopes", slo),) : ())..., (dev ? (("dev", dvt),) : ())...,
                  (tyre_mgmt ? (("tyre (drv)", tdv), ("tyre (car)", tmv)) : ())...)
    println(rpad(name, 10), round(maximum(MCMCDiagnosticTools.rhat(A)); digits = 3), "  ",
            round(Int, minimum(MCMCDiagnosticTools.ess(A))))
end
println("divergences: ", count(chain[:numerical_error] .> 0))
flat(A) = reshape(A, ni * nc, :)
println("\nscalars (mean ± sd):")
for (j, nm) in enumerate((string.(scal_names)..., (tyres ? ("β_stint", "β_stint²/10") : ())..., (slopes ? ("τ_slope",) : ())...,
                          (dev ? ("τ_dev",) : ())..., (lead ? ("η_lead",) : ())..., (tyre_mgmt ? ("τ_td", "τ_tm") : ())...))
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
