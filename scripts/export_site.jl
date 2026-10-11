# Export posterior summaries of fitted models for the documentation site (#45).
#
#   julia --project scripts/export_site.jl race [model=pl_dur_rw_kappa_age_big_slope_dev_fgam]
#   julia --project scripts/export_site.jl retire
#   julia --project scripts/export_site.jl laps [tag=laps_2011-2019_recorded]
#   julia --project scripts/export_site.jl wrc [model=pl_dur_rw_kappa_cdrv_tdrv_fgam]
#
# Run from the repository root with the fits in output/. Each mode writes compact
# tables and a manifest.json (what they came from: model or fit tag, data, draws,
# convergence, export commit and date) under docs/src/assets/data/<mode>/. The site
# is built from these files only; nothing is refitted in CI.
#
# race: from the saved chains output/gap_all_<model>_chain<k>.jls (scripts/gap_fit.jl,
# F1, all rows):
#   drivers.csv   driver effects (% of race time, average-season units; negative =
#                 faster), with career slopes if the model has them
#   cars.csv      car-season effects (% of race time within season), with in-season
#                 development trends if the model has them
#   seasons.csv   per season: incident probability π, mean loss λ, pace scale κ
#   age_curve.csv the population career curve, if the model has one
# retire: the tables of scripts/retire_fit.jl combine (output/retire_*.csv), with
#   convergence from output/retire_combine.txt.
# laps: the tables of scripts/lap_fit.jl combine (output/<tag>_{drivers,cars}.csv),
#   plus the same drivers and car-seasons joined with the race model's effects (run
#   `race` first), with convergence from output/<tag>_combine.txt.
# wrc: the effect tables of scripts/gap_fit.jl wrc2018-2025 <model> combine. Its
#   convergence isn't recomputed: the WRC source refreshes its season list, so the
#   data could differ from the fit's; the manifest points to where it was reported.

using Diomedes, CSV, DataFrames, Dates, JSON3, Serialization, Statistics
using Diomedes: draw, model_param_names, rw_path, season_counts
using LogExpFunctions: logistic


const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "data")
r4(x) = round.(x; digits = 4)
round_floats!(df) = (for c in names(df); eltype(df[!, c]) <: Union{Missing,AbstractFloat} && (df[!, c] = r4(df[!, c])); end; df)
export_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
function write_manifest(dir, m)
    open(joinpath(dir, "manifest.json"), "w") do io
        JSON3.pretty(io, merge(m, (; export_commit = export_commit(), date = string(today()))))
    end
end

# "RetireData(22478 starts: …)" -> "22478 starts: …"
inner(s) = something(match(r"^\w+\((.*)\)$", s), (s,))[1]

# Convergence block of a combine log: lines "<group>  <max R-hat>  <min ESS>" after
# "convergence", and the "divergences: n" line.
function log_convergence(path)
    lines = readlines(path)
    i = findfirst(l -> startswith(l, "convergence"), lines)
    groups = Dict{String,Any}()
    for l in lines[(i + 1):end]
        m = match(r"^(\S.*?)\s+([\d.]+)\s+([\d.]+)\s*$", l)
        m === nothing && break
        groups[strip(m[1])] = (; max_rhat = parse(Float64, m[2]), min_ess = parse(Float64, m[3]))
    end
    d = something(match(r"divergences: (\d+)", join(lines, "\n")), (nothing, "missing"))
    return (; max_rhat = maximum(v.max_rhat for v in values(groups)), min_ess = minimum(v.min_ess for v in values(groups)),
            divergences = d[1] === nothing ? missing : parse(Int, d[1]), groups)
end

function export_race(model)
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
    outdir = joinpath(OUT, "race")
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
    # per-level columns go on while rows are still in g.competitors order (SL's
    # columns); the join then keeps the left order
    if spec.slopes
        dt.slope_mean, dt.slope_sd = vec(mean(SL; dims = 1)), vec(std(SL; dims = 1))
    end
    dt = leftjoin(dt, select(eff.competitors, :label => :id, :mean, :sd, :q05, :q95); on = :id, order = :left)
    sort!(dt, :mean)
    select!(dt, :id, :name, :rows, :first_season, :last_season, :mean, :sd, :q05, :q95, Not([:id, :name, :rows, :first_season, :last_season, :mean, :sd, :q05, :q95]))
    for c in (:mean, :sd, :q05, :q95, (spec.slopes ? (:slope_mean, :slope_sd) : ())...)
        dt[!, c] = r4(dt[!, c])
    end
    CSV.write(joinpath(outdir, "drivers.csv"), dt)

    # car-seasons
    machs = vcat(g.t_mach, g.c_mach)
    ct = DataFrame(id = g.machines, season = [seasons[s] for s in g.mach_season],
                   rows = [count(==(m), machs) for m in eachindex(g.machines)])
    ct.constructor = [replace(m, Regex("_$(s)\$") => "") for (m, s) in zip(ct.id, ct.season)]
    if spec.dev                                   # g.machines order, as DV's columns
        ct.dev_mean, ct.dev_sd = vec(mean(DV; dims = 1)), vec(std(DV; dims = 1))
    end
    ct = leftjoin(ct, select(eff.machines, :label => :id, :mean, :sd, :q05, :q95); on = :id, order = :left)
    sort!(ct, [:season, :mean])
    for c in (:mean, :sd, :q05, :q95, (spec.dev ? (:dev_mean, :dev_sd) : ())...)
        ct[!, c] = r4(ct[!, c])
    end
    CSV.write(joinpath(outdir, "cars.csv"), select(ct, :id, :constructor, :season, :rows, :mean, :sd, :q05, :q95,
                                                     Not([:id, :constructor, :season, :rows, :mean, :sd, :q05, :q95])))

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
    manifest = (; model, fit = "gap_all_$(model)", chains = nc, draws_per_chain = ni,
                data = "F1 $(first(seasons))–$(last(seasons)), Ergast CSV, all classified finishers (lapped as intervals)",
                rows = (; timed = length(g.y), lapped = length(g.lo)),
                convergence = (; max_rhat = round(ic.overall.max_rhat; digits = 3), min_ess = round(ic.overall.min_ess; digits = 1),
                               divergences = count(identity, chain[:numerical_error])),
                exported_by = "scripts/export_site.jl race")
    write_manifest(outdir, manifest)
    println("wrote $(outdir): $(nrow(dt)) drivers, $(nrow(ct)) car-seasons, $(nrow(st)) seasons; ",
            "$(nc) chains × $(ni) draws; R-hat $(manifest.convergence.max_rhat), min ESS $(manifest.convergence.min_ess)")
end

function export_retire()
    dir = joinpath(OUT, "retire")
    mkpath(dir)
    cars = round_floats!(CSV.read("output/retire_cars.csv", DataFrame))
    cars.constructor = [replace(m, Regex("_$(s)\$") => "") for (m, s) in zip(cars.machine, cars.season)]
    CSV.write(joinpath(dir, "cars.csv"), select(cars, :machine => :id, :constructor, Not([:machine, :constructor])))
    CSV.write(joinpath(dir, "drivers.csv"), select(round_floats!(CSV.read("output/retire_drivers.csv", DataFrame)),
                                                   :driver => :id, Not(:driver)))
    CSV.write(joinpath(dir, "seasons.csv"), round_floats!(CSV.read("output/retire_seasons.csv", DataFrame)))
    log = "output/retire_combine.txt"
    write_manifest(dir, (; model = "retirement_effects (competing risks: mechanical, incident, collision)",
                         fit = "retire", data = "F1 1950–2019 (Ergast CSV): " * inner(first(readlines(log))), convergence = log_convergence(log),
                         exported_by = "scripts/export_site.jl retire"))
    println("wrote $dir")
end

function export_laps(tag)
    dir = joinpath(OUT, "laps")
    mkpath(dir)
    d = round_floats!(CSV.read("output/$(tag)_drivers.csv", DataFrame))
    c = round_floats!(CSV.read("output/$(tag)_cars.csv", DataFrame))
    rename!(d, :driver => :id)
    rename!(c, :machine => :id)
    CSV.write(joinpath(dir, "drivers.csv"), d)
    CSV.write(joinpath(dir, "cars.csv"), c)
    # the same levels in the race model (export `race` first)
    race = joinpath(OUT, "race")
    rd = CSV.read(joinpath(race, "drivers.csv"), DataFrame)
    rc = CSV.read(joinpath(race, "cars.csv"), DataFrame)
    CSV.write(joinpath(dir, "vs_race_drivers.csv"),
              innerjoin(select(d, :id, :name, :laps, :mean => :lap_mean, :sd => :lap_sd),
                        select(rd, :id, :mean => :race_mean, :sd => :race_sd); on = :id))
    CSV.write(joinpath(dir, "vs_race_cars.csv"),
              innerjoin(select(c, :id, :laps, :mean => :lap_mean, :sd => :lap_sd),
                        select(rc, :id, :constructor, :season, :mean => :race_mean, :sd => :race_sd); on = :id))
    log = "output/$(tag)_combine.txt"
    yrs = match(r"(\d{4})-(\d{4})_(\w+)", tag)
    write_manifest(dir, (; model = "lap_effects", fit = tag,
                         data = "F1 $(yrs[1])–$(yrs[2]) clean laps (Ergast lap times, $(yrs[3]) pit stops): " * inner(first(readlines(log))),
                         race_model = JSON3.read(read(joinpath(race, "manifest.json"), String)).model,
                         convergence = log_convergence(log), exported_by = "scripts/export_site.jl laps"))
    println("wrote $dir")
end

function export_wrc(model)
    dir = joinpath(OUT, "wrc")
    mkpath(dir)
    tag = "gap_wrc2018-2025_$(model)"
    d = round_floats!(CSV.read("output/$(tag)_competitor_effects.csv", DataFrame))
    CSV.write(joinpath(dir, "drivers.csv"), select(d, :label => :id, :name, :mean, :sd, :q05, :q95, :n_obs => :stages))
    c = round_floats!(CSV.read("output/$(tag)_machine_effects.csv", DataFrame))
    m = [match(r"^(.*) (\S+)_(\d{4})$", l) for l in c.label]          # "Toyota Rally1_2024"
    c.manufacturer, c.tier, c.season = [x[1] for x in m], [x[2] for x in m], [parse(Int, x[3]) for x in m]
    CSV.write(joinpath(dir, "cars.csv"), select(c, :label => :id, :manufacturer, :tier, :season, :mean, :sd, :q05, :q95,
                                                 :n_obs => :stage_times))
    write_manifest(dir, (; model, fit = tag,
                         data = "WRC 2018–2025 stage times, top two tiers (Rally1 incl. WRC, Rally2 incl. R5), wrc.com timing API",
                         convergence = "reported in https://github.com/aleccrowell/diomedes/pull/33",
                         exported_by = "scripts/export_site.jl wrc"))
    println("wrote $dir")
end

mode = get(ARGS, 1, "race")
if mode == "race"
    export_race(get(ARGS, 2, "pl_dur_rw_kappa_age_big_slope_dev_fgam"))
elseif mode == "retire"
    export_retire()
elseif mode == "laps"
    export_laps(get(ARGS, 2, "laps_2011-2019_recorded"))
elseif mode == "wrc"
    export_wrc(get(ARGS, 2, "pl_dur_rw_kappa_cdrv_tdrv_fgam"))
else
    error("unknown mode $mode: race, retire, laps or wrc")
end
