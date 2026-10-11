# Export posterior summaries of a fitted race-time model for the documentation site (#45).
#
#   julia --project scripts/export_site.jl [model=pl_dur_rw_kappa_age_big_slope_dev_fgam]
#
# Reads the saved chains output/gap_all_<model>_chain<k>.jls (scripts/gap_fit.jl,
# F1, all rows) and writes compact tables under docs/src/assets/data/race/:
#   drivers.csv   driver effects (% of race time, average-season units; negative =
#                 faster), with career slopes if the model has them
#   cars.csv      car-season effects (% of race time within season), with in-season
#                 development trends if the model has them
#   seasons.csv   per season: incident probability π, mean loss λ, pace scale κ
#   age_curve.csv the population career curve, if the model has one
#   manifest.json what the tables came from: model, chains, draws, data, commit, date,
#                 convergence of the identified quantities (export_commit is the code that
#                 exported them; the fit itself is named by `fit`)
# The site is built from these files only; nothing is refitted in CI.

using Diomedes, CSV, DataFrames, Dates, JSON3, Serialization, Statistics
using Diomedes: draw, model_param_names, rw_path, season_counts
using LogExpFunctions: logistic

model = get(ARGS, 1, "pl_dur_rw_kappa_age_big_slope_dev_fgam")
spec = model_spec(model)
spec.family === :pl && spec.era === :rw && spec.pace_scale ||
    error("export_site expects a pace + loss model with random-walk eras and a pace scale (pl…_rw_kappa…)")
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
res = fetch_results(ErgastCSV(data_dir), 1950:2100)
g = prepare_gaps(res)
files = filter(isfile, ["output/gap_all_$(model)_chain$(k).jls" for k in 1:8])
isempty(files) && error("no saved chains for $model")
chain = reduce(hcat, rehash_chain!.(deserialize.(files)))
pnames = model_param_names(spec)
sc, cov = season_counts(g), LossCovariates(g)
ni, nc = size(chain[:σ])
ndraw = ni * nc

# one pass over the draws: everything derived from θ
seasons = minimum(g.race_season) .+ (0:(cov.n_seasons - 1))
lp, ll, lk = zeros(ndraw, cov.n_seasons), zeros(ndraw, cov.n_seasons), zeros(ndraw, cov.n_seasons)
SL = spec.slopes ? zeros(ndraw, length(g.competitors)) : nothing
DV = spec.dev ? zeros(ndraw, length(g.machines)) : nothing
B = spec.age ? AgeCurveBasis(g) : nothing
AC = spec.age ? zeros(ndraw, length(g.age_years)) : nothing
let k = 0
    for c in 1:nc, i in 1:ni
        θ = draw(chain, pnames, i, c)
        k += 1
        lp[k, :] = θ.a_π .+ rw_path(θ.τ_π_rw .* θ.e_π_rw)
        ll[k, :] = θ.a_λ .+ rw_path(θ.τ_λ_rw .* θ.e_λ_rw)
        lk[k, :] = rw_path(θ.τ_κ .* θ.e_κ)
        spec.slopes && (SL[k, :] = career_slopes(θ.τ_slope, θ.u_slope))
        spec.dev && (DV[k, :] = dev_trends(θ.τ_dev, θ.u_dev, g, sc))
        spec.age && (AC[k, :] = age_curve(θ.τ_age, θ.e_age, B))
    end
end
q(X, p) = [quantile(c, p) for c in eachcol(X)]
r4(x) = round.(x; digits = 4)
outdir = joinpath(@__DIR__, "..", "docs", "src", "assets", "data", "race")
mkpath(outdir)

# drivers
eff = gap_effects_table(chain, g)
comps = vcat(g.t_comp, g.c_comp)
racerow_season = vcat(g.race_season[g.t_race], g.race_season[g.c_race])
dname = Dict(r.competitor_id => r.competitor_name for r in eachrow(res))
dt = DataFrame(id = g.competitors, name = [dname[c] for c in g.competitors],
               rows = [count(==(d), comps) for d in eachindex(g.competitors)],
               first_season = [minimum(racerow_season[comps .== d]) for d in eachindex(g.competitors)],
               last_season = [maximum(racerow_season[comps .== d]) for d in eachindex(g.competitors)])
de = eff.competitors
@assert de.label isa AbstractVector
dt = leftjoin(dt, select(de, :label => :id, :mean, :sd, :q05, :q95); on = :id)
if spec.slopes
    dt.slope_mean, dt.slope_sd = vec(mean(SL; dims = 1)), vec(std(SL; dims = 1))
end
sort!(dt, :mean)
for c in (:mean, :sd, :q05, :q95, (spec.slopes ? (:slope_mean, :slope_sd) : ())...)
    dt[!, c] = r4(dt[!, c])
end
CSV.write(joinpath(outdir, "drivers.csv"), dt)

# car-seasons
machs = vcat(g.t_mach, g.c_mach)
ct = DataFrame(id = g.machines, season = [seasons[s] for s in g.mach_season],
               rows = [count(==(m), machs) for m in eachindex(g.machines)])
ct.constructor = [replace(m, Regex("_$(s)\$") => "") for (m, s) in zip(ct.id, ct.season)]
ct = leftjoin(ct, select(eff.machines, :label => :id, :mean, :sd, :q05, :q95); on = :id)
if spec.dev
    ct.dev_mean, ct.dev_sd = vec(mean(DV; dims = 1)), vec(std(DV; dims = 1))
end
sort!(ct, [:season, :mean])
for c in (:mean, :sd, :q05, :q95, (spec.dev ? (:dev_mean, :dev_sd) : ())...)
    ct[!, c] = r4(ct[!, c])
end
CSV.write(joinpath(outdir, "cars.csv"), select(ct, :id, :constructor, :season, Not([:id, :constructor, :season])))

# seasons
st = DataFrame(season = seasons, races = [count(==(s), g.race_season) for s in seasons],
               pi_mean = vec(mean(logistic.(lp); dims = 1)), pi_q05 = logistic.(q(lp, 0.05)), pi_q95 = logistic.(q(lp, 0.95)),
               lambda_mean = vec(mean(exp.(ll); dims = 1)), lambda_q05 = exp.(q(ll, 0.05)), lambda_q95 = exp.(q(ll, 0.95)),
               kappa_mean = vec(mean(exp.(lk); dims = 1)), kappa_q05 = exp.(q(lk, 0.05)), kappa_q95 = exp.(q(lk, 0.95)))
for c in names(st)[3:end]
    st[!, c] = r4(st[!, c])
end
CSV.write(joinpath(outdir, "seasons.csv"), st)

# age curve
if spec.age
    CSV.write(joinpath(outdir, "age_curve.csv"),
              DataFrame(age = g.age_years, rows = Int.(g.age_rows), mean = r4(vec(mean(AC; dims = 1))),
                        q05 = r4(q(AC, 0.05)), q95 = r4(q(AC, 0.95))))
end

# provenance
ic = identified_convergence(chain, g, model)
commit = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
manifest = (; model, fit = "gap_all_$(model)", chains = nc, draws_per_chain = ni,
            data = "F1 $(first(seasons))–$(last(seasons)), Ergast CSV, all classified finishers (lapped as intervals)",
            rows = (; timed = length(g.y), lapped = length(g.lo)),
            convergence = (; max_rhat = round(ic.overall.max_rhat; digits = 3), min_ess = round(ic.overall.min_ess; digits = 1),
                           divergences = count(identity, chain[:numerical_error])),
            exported_by = "scripts/export_site.jl", export_commit = commit, date = string(today()))
open(joinpath(outdir, "manifest.json"), "w") do io
    JSON3.pretty(io, manifest)
end
println("wrote $(outdir): $(nrow(dt)) drivers, $(nrow(ct)) car-seasons, $(nrow(st)) seasons; ",
        "$(nc) chains × $(ni) draws; R-hat $(manifest.convergence.max_rhat), min ESS $(manifest.convergence.min_ess)")
