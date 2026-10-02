# Fit the retirement (DNF) model (#13) on F1 data and write rankings.
#
#   julia --project scripts/retire_fit.jl chain <k>          # one chain (seed k), saved to output/
#   julia --project scripts/retire_fit.jl combine [n=4]      # combine saved chains 1..n, write tables
#
# 1000 draws per chain after 500 warm-up iterations. Run the chains as separate
# processes (one per core), then `combine`. Writes, under output/:
#   retire_cars.csv     car-season mechanical reliability (log-hazard effect within season)
#   retire_drivers.csv  driver incident-proneness (log-hazard effect)
#   retire_seasons.csv  expected retirements per race distance by cause and season
# and prints convergence of the identified quantities and the scalar parameters.

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics
using Diomedes: sum_to_zero, sum_to_zero_by, rw_path, retire_counts
import MCMCDiagnosticTools

action = get(ARGS, 1, "combine")
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
chain_path(k) = "output/retire_chain$(k).jls"
d = prepare_retirements(fetch_results(ErgastCSV(data_dir), 1950:2100))
println(d); flush(stdout)
mkpath("output")

if action == "chain"
    k = parse(Int, ARGS[2])
    t = @elapsed chain = fit_retirement(d; rng = Xoshiro(k), progress = false, progress_log = stdout)
    serialize(chain_path(k), chain)
    println("chain $k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", step size ", round(mean(chain[:step_size]); digits = 4),
            ", divergences ", count(chain[:numerical_error] .> 0))
    exit()
end

n = parse(Int, get(ARGS, 2, "4"))
chain = reduce(hcat, [rehash_chain!(deserialize(chain_path(k))) for k in 1:n])
ni, nc = size(chain[:σ_car])
mc, rc = retire_counts(d)
S = length(d.seasons)

# identified quantities per draw: (draws, chains, k)
function collect3(f)
    v1 = f(1, 1)
    A = Array{Float64,3}(undef, ni, nc, length(v1))
    for c in 1:nc, i in 1:ni
        A[i, c, :] = f(i, c)
    end
    return A
end
cars = collect3((i, c) -> chain[:σ_car][i, c] .* sum_to_zero_by(chain[:z_car][i, c], d.mach_season, mc))
drvs = collect3((i, c) -> chain[:σ_drv][i, c] .* sum_to_zero(chain[:z_drv][i, c]))
level(k, e) = (i, c) -> chain[:a][i, c][k] .+ rw_path(chain[:τ_rw][i, c][k] .* chain[e][i, c])
lev = [collect3(level(1, :e_m)), collect3(level(2, :e_i)), collect3(level(3, :e_c))]
scal = collect3((i, c) -> vcat(chain[:a][i, c], chain[:τ_rw][i, c], chain[:b1][i, c], chain[:σ_car][i, c],
                               chain[:σ_drv][i, c], chain[:β_col][i, c], chain[:σ_race][i, c]))
scal_names = ["a_mech", "a_inc", "a_col", "τ_mech", "τ_inc", "τ_col", "b1_mech", "b1_inc", "b1_col",
              "σ_car", "σ_drv", "β_col", "σ_race_mech", "σ_race_inc", "σ_race_col"]

println("\nconvergence (max R-hat, min bulk ESS):")
for (name, A) in (("cars", cars), ("drivers", drvs), ("mech level", lev[1]), ("incident level", lev[2]),
                  ("collision level", lev[3]), ("scalars", scal))
    r = MCMCDiagnosticTools.rhat(A)
    e = MCMCDiagnosticTools.ess(A)
    println(rpad(name, 16), round(maximum(r); digits = 3), "  ", round(Int, minimum(e)))
end
println("divergences: ", count(chain[:numerical_error] .> 0))

flat(A) = reshape(A, ni * nc, :)
println("\nscalars (mean ± sd):")
for (j, nm) in enumerate(scal_names)
    v = flat(scal)[:, j]
    println(rpad(nm, 14), round(mean(v); digits = 3), " ± ", round(std(v); digits = 3))
end

r = d.rows
mech = d.cause .== 1
inc = d.cause .== 2
col = d.cause .== 3
expo = d.n ./ d.L

C = flat(cars)
car_tab = DataFrame(machine = d.machines, season = d.seasons[d.mach_season],
                    effect = vec(mean(C; dims = 1)), sd = vec(std(C; dims = 1)),
                    starts = [count(==(m), d.mach) for m in eachindex(d.machines)],
                    mech_dnf = [count(mech .& (d.mach .== m)) for m in eachindex(d.machines)])
sort!(car_tab, :effect)
CSV.write("output/retire_cars.csv", car_tab)

D = flat(drvs)
drv_tab = DataFrame(driver = d.competitors, name = [first(r.competitor_name[r.competitor_id .== c]) for c in d.competitors],
                    effect = vec(mean(D; dims = 1)), sd = vec(std(D; dims = 1)),
                    starts = [count(==(c), d.comp) for c in eachindex(d.competitors)],
                    incidents = [count(inc .& (d.comp .== c)) for c in eachindex(d.competitors)],
                    collisions = [count(col .& (d.comp .== c)) for c in eachindex(d.competitors)])
sort!(drv_tab, :effect)
CSV.write("output/retire_drivers.csv", drv_tab)

# expected retirements per race distance (hazard per race, ignoring the first-lap jump)
season_tab = DataFrame(season = d.seasons)
for (k, nm) in enumerate(("mech", "incident", "collision"))
    E = exp.(flat(lev[k]))
    season_tab[!, Symbol(nm)] = vec(mean(E; dims = 1))
    season_tab[!, Symbol(nm, "_obs")] = [count((d.cause .== k) .& (d.season .== s)) / sum(expo[d.season .== s])
                                         for s in 1:S]
end
CSV.write("output/retire_seasons.csv", season_tab)

println("\nmost reliable car-seasons (≥ 20 starts):")
show(stdout, MIME"text/plain"(), first(car_tab[car_tab.starts .>= 20, :], 10)); println()
println("least reliable car-seasons (≥ 20 starts):")
show(stdout, MIME"text/plain"(), last(car_tab[car_tab.starts .>= 20, :], 10)); println()
println("\nleast incident-prone drivers (≥ 50 starts):")
show(stdout, MIME"text/plain"(), first(drv_tab[drv_tab.starts .>= 50, :], 15)); println()
println("most incident-prone drivers (≥ 50 starts):")
show(stdout, MIME"text/plain"(), last(drv_tab[drv_tab.starts .>= 50, :], 15)); println()
println("\nseason hazards per race distance (every 5th season):")
show(stdout, MIME"text/plain"(), season_tab[1:5:end, :]); println()
