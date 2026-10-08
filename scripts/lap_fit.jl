# Fit the lap-time model (#12) on F1 lap times (Ergast, 1996–2019).
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
# first k blocks of each chain are used.

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics, Turing
using Diomedes: sum_to_zero, sum_to_zero_by, lap_mach_counts
import MCMCDiagnosticTools

m = match(r"^(\d{4})-(\d{4})$", ARGS[1])
seasons = parse(Int, m[1]):parse(Int, m[2])
pits = Symbol(ARGS[2])
action = get(ARGS, 3, "combine")
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
tag = "laps_$(ARGS[1])_$(pits)"
src = ErgastCSV(data_dir)
res = fetch_results(src, seasons)
d = prepare_laps(fetch_laps(src, seasons), res; pits,
                 pit_stops = pits === :recorded ? fetch_pit_stops(src, seasons) : nothing)
println(d); flush(stdout)
mkpath("output")

if action == "chain"
    k = parse(Int, ARGS[4])
    # warm=<first>-<last>: start from the adapted metric and step size of chain k of that
    # (finished, same-pits) fit, and adapt for 300 iterations instead of 500
    warm = something(match(r"^warm=(\d{4})-(\d{4})$", get(ARGS, 5, "")), nothing)
    ws, n_adapts = nothing, 500
    if warm !== nothing
        old_seasons = parse(Int, warm[1]):parse(Int, warm[2])
        # chain k of the old fit, or (if it has fewer chains) one of them in turn
        have = sort(unique(parse(Int, m[1]) for f in readdir("output")
                           for m in (match(Regex("^laps_$(warm[1])-$(warm[2])_$(pits)_chain(\\d+)_block"), f),)
                           if m !== nothing))
        old_tag = "laps_$(warm[1])-$(warm[2])_$(pits)_chain$(k in have ? k : have[mod1(k, length(have))])"
        blocks = filter(f -> startswith(f, old_tag * "_block"), readdir("output"))
        last_block = argmax(f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]), blocks)
        st = only(Turing.loadstate(rehash_chain!(deserialize(joinpath("output", last_block)))))
        d_old = prepare_laps(fetch_laps(src, old_seasons), fetch_results(src, old_seasons); pits,
                             pit_stops = pits === :recorded ? fetch_pit_stops(src, old_seasons) : nothing)
        ws, n_adapts = lap_warm_start(st, d_old, d), 300
        println("warm start from $last_block: step size $(round(ws.ϵ; sigdigits = 3))"); flush(stdout)
    end
    t = @elapsed chain = fit_laps(d; rng = Xoshiro(k), checkpoint = "output/$(tag)_chain$(k)", seed = 1000k,
                                  progress_log = stdout, warm_start = ws, n_adapts)
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
arg = get(ARGS, 4, "4")              # chains 1..n, or a comma-separated list (e.g. 2,3,4)
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
scal = collect3((i, c) -> vcat([chain[s][i, c] for s in scal_names], chain[:β][i, c]))
println("\nconvergence (max R-hat, min bulk ESS):")
for (name, A) in (("drivers", drv), ("cars", car), ("scalars", scal))
    println(rpad(name, 10), round(maximum(MCMCDiagnosticTools.rhat(A)); digits = 3), "  ",
            round(Int, minimum(MCMCDiagnosticTools.ess(A))))
end
println("divergences: ", count(chain[:numerical_error] .> 0))
flat(A) = reshape(A, ni * nc, :)
println("\nscalars (mean ± sd):")
for (j, nm) in enumerate((string.(scal_names)..., "β_stint", "β_stint²/10"))
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
