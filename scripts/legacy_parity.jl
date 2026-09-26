# Check the Julia pipeline against the legacy Python one.
#
#   julia --project -t 4 scripts/legacy_parity.jl [ergast_csv_dir]
#
# 1. Data parity: `prepare(ErgastCSV(dir))` must give exactly the rows and
#    z-scores in the legacy `processed.csv`.
# 2. Fits the faithful port (σ_y fixed at 1, as in the legacy model) on the
#    same data and writes effect tables to `output/`.

using Diomedes, CSV, DataFrames, Statistics

dir = get(ARGS, 1, joinpath(@__DIR__, "..", "data"))
legacy = CSV.read(joinpath(dir, "processed.csv"), DataFrame)
races = CSV.read(joinpath(dir, "races.csv"), DataFrame; missingstring = "\\N")
drivers = CSV.read(joinpath(dir, "drivers.csv"), DataFrame; missingstring = "\\N")
legacy = innerjoin(legacy, select(races, :raceId, :round), on = :raceId)
legacy = innerjoin(legacy, select(drivers, :driverId, :driverRef), on = :driverId)
legacy.event_id = string.(legacy.year, "-", lpad.(legacy.round, 2, '0'))

seasons = unique(races.year)
results = fetch_results(ErgastCSV(dir), seasons)
d = prepare(results)
println(d); flush(stdout)

@assert length(d) == nrow(legacy) "row count: julia $(length(d)) vs legacy $(nrow(legacy))"
j = innerjoin(d.rows, select(legacy, :event_id, :driverRef, :z_score);
              on = [:event_id, :competitor_id => :driverRef])
@assert nrow(j) == nrow(legacy) "only $(nrow(j)) of $(nrow(legacy)) legacy rows matched"
maxdiff = maximum(abs.(j.z .- j.z_score))
@assert maxdiff < 1e-9 "z-scores differ by up to $maxdiff"
@assert length(d.machines) == length(unique(legacy.cyd))
@assert length(d.competitors) == length(unique(legacy.dd))
println("data parity OK: $(nrow(j)) rows, max |Δz| = $maxdiff"); flush(stdout)

chain = fit_effects(d; σ_y = 1.0, n_samples = 1000, n_chains = max(1, Threads.nthreads()))
display(chain[[:α, :σ_comp, :σ_mach]])
eff = effects_table(chain, d)
mkpath("output")
CSV.write("output/legacy_competitor_effects.csv", eff.competitors)
CSV.write("output/legacy_machine_effects.csv", eff.machines)
println("\nFastest 10 drivers (standardised race time, negative = faster):")
show(first(eff.competitors, 10); allcols = true)
