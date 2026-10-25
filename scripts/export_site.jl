# Export posterior summaries of fitted models for the documentation site (#45).
#
#   julia --project scripts/export_site.jl race [model=pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc] [data=jolpica1950-2026+partial]
#   julia --project scripts/export_site.jl retire [prefix=retire]
#   julia --project scripts/export_site.jl laps [tag=laps_2011-2019_recorded]
#   julia --project scripts/export_site.jl wrc [model=pl_dur_rw_kappa_cdrv_tdrv_fgam]
#
# Run from the repository root with the fits in output/. Each mode writes compact
# tables and a manifest.json (what they came from: model or fit tag, data, draws,
# convergence, export commit and date) under docs/src/assets/data/<mode>/. The site
# is built from these files only; nothing is refitted in CI.
#
# race: from the saved chains output/gap_<data>_<model>_chain<k>.jls (scripts/gap_fit.jl
# <data> <model>, data as in scripts/gap_data.jl: `all` or `jolpica<a>-<b>[+partial]`).
# Chains are grouped into modes (`chain_modes`, on the loss scalars, #64). A mode
# whose mean log density is more than MODE_DROP_NATS below the best one's holds a
# negligible share of the posterior: the tables use the other modes' chains, and
# the dropped mode is reported in modes.csv, seasons_by_mode.csv and the manifest
# (with how far the effects differ between modes):
#   drivers.csv   driver effects (% of race time, average-season units; negative =
#                 faster), with career slopes and incident proneness (#101, logit
#                 offsets on the incident probability) if the model has them; with
#                 the latter also each driver's pace part (the effect), incident part
#                 (the time their proneness adds in a reference race of average
#                 incident rate and loss size) and overall, average-season units
#   cars.csv      car-season effects (% of race time within season), with in-season
#                 development trends and circuit slopes (speed, rough, temp; #81) if the
#                 model has them. With chassis and engine walks (#95) the effect is the
#                 total, and chassis_mean + engine_mean + residual_mean split its mean
#                 (each centred within season, so they add up); with the works-team term
#                 (#100) works_mean is a fourth part
#   walks.csv     chassis and engine walks (#95): each walk's state per season (% in
#                 average-season units, centred within season; negative = faster)
#   track.csv     track terms (#81): incident odds and loss-size multipliers for walls,
#                 mixed and wet races
#   seasons.csv   per season: incident probability π, mean loss λ (for a dry race at a
#                 circuit of average tightness when the model has track terms), pace scale κ
#   seasons_by_mode.csv  the same per mode, all chains (`mode`, `kept` columns)
#   modes.csv     per chain (all chains): its mode, whether it is used, and the loss scalars on readable scales
#                 (pace noise σ, routine-loss probability and mean, big-loss share and
#                 mean), with 90% intervals, one row per chain and quantity
#   age_curve.csv the population career curve, if the model has one
#   ranks.csv     per driver with ≥ 20 results: P(fastest), P(top 3), P(top 10) and the
#                 rank's median and 90% interval, ranking those drivers in every draw;
#                 with incident proneness on race pace (effect + incident part), and the
#                 `_pace` columns on the driver effect alone
#   driver_seasons.csv  per driver and car-season raced: the driver part
#                 (κ_season × (driver effect + career terms averaged over that season's
#                 results)), the car part (car-season effect) and their sum, in that
#                 season's % units, with 90% intervals
# retire: the tables of scripts/retire_fit.jl combine (output/<prefix>_*.csv, e.g.
#   prefix retire_jolpica1950-2025), with convergence from output/<prefix>_combine.txt
#   (the combine step's printed output).
# laps: the tables of scripts/lap_fit.jl combine (output/<tag>_{drivers,cars}.csv),
#   plus the same drivers and car-seasons joined with the race model's effects (run
#   `race` first), with convergence from output/<tag>_combine.txt.
# wrc: the effect tables of scripts/gap_fit.jl wrc2018-2025 <model> combine. Its
#   convergence isn't recomputed: the WRC source refreshes its season list, so the
#   data could differ from the fit's; the manifest points to where it was reported.

using Diomedes, CSV, DataFrames, Dates, JSON3, Serialization, Statistics, Turing
using Diomedes: draw, model_param_names, rw_path, season_counts, sum_to_zero, sum_to_zero_by, CareerRows, driver_offsets,
                 walk_offsets
using LogExpFunctions: logistic, logit


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
        # labels may run into the R-hat (e.g. "tyre (drv)1.031  188")
        m = match(r"^(.*?[^\s\d.])\s*([\d.]+)\s+([\d.]+)\s*$", l)
        m === nothing && break
        groups[strip(m[1])] = (; max_rhat = parse(Float64, m[2]), min_ess = parse(Float64, m[3]))
    end
    d = something(match(r"divergences: (\d+)", join(lines, "\n")), (nothing, "missing"))
    return (; max_rhat = maximum(v.max_rhat for v in values(groups)), min_ess = minimum(v.min_ess for v in values(groups)),
            divergences = d[1] === nothing ? missing : parse(Int, d[1]), groups)
end

include("gap_data.jl")

# the loss scalars that define a mode, on readable scales (#64)
function loss_quantities(chain, spec)
    q = [("sigma", "Pace noise σ (%)", vec(chain[:σ])),
         ("pi_base", "Routine-loss probability", logistic.(vec(chain[:a_π]))),
         ("lambda_base", "Routine-loss mean (%)", exp.(vec(chain[:a_λ])))]
    if spec.big_loss
        push!(q, ("rho", "Big-loss share of incidents", logistic.(vec(chain[:a_ρ]))),
              ("lambda_big", "Big-loss mean (%)", exp.(vec(chain[:a_λ]) .+ vec(chain[:δ_λ2]))))
    end
    return q
end

# A mode whose mean log density is more than this below the best mode's holds a
# negligible share of the posterior (#64): its draws are reported, not pooled
const MODE_DROP_NATS = 10

# Season paths of π and λ are drawn for a dry race at a circuit of average
# tightness (#81): with track terms, a_π and a_λ are a dry race at an open
# circuit, so the tight slope is added at the mean tight score over the races.
function track_shift(θ, cov)
    haskey(θ, :β_track_π) || return 0.0, 0.0
    t = mean(cov.circuits.track[:, 1])
    return θ.β_track_π[1] * t, θ.β_track_λ[1] * t
end

function export_modes(chs, g, spec, model, outdir)
    all = reduce(hcat, chs)
    ni = size(all[:σ], 1)
    nc = length(chs)
    modes = chain_modes(all, [:σ, :a_π, :a_λ, (spec.big_loss ? (:a_ρ, :δ_λ2) : ())...])
    labels = unique(modes)
    idx(m) = findall(==(m), modes)
    lj = Dict(m => mean(mean(vec(chs[c][:logjoint])) for c in idx(m)) for m in labels)
    best = maximum(values(lj))
    kept = [m for m in labels if lj[m] > best - MODE_DROP_NATS]

    # season paths per mode
    pnames, cov = model_param_names(spec), LossCovariates(g, spec)
    seasons = minimum(g.race_season) .+ (0:(cov.n_seasons - 1))
    q(X, p) = [quantile(c, p) for c in eachcol(X)]
    sm = DataFrame()
    for m in labels
        cs = idx(m)
        n = ni * length(cs)
        P, Λ, Kp = zeros(n, cov.n_seasons), zeros(n, cov.n_seasons), zeros(n, cov.n_seasons)
        k = 0
        for c in cs, i in 1:ni
            θ = draw(all, pnames, i, c)
            k += 1
            sπ, sλ = track_shift(θ, cov)
            P[k, :] = θ.a_π .+ sπ .+ rw_path(θ.τ_π_rw .* θ.e_π_rw)
            Λ[k, :] = θ.a_λ .+ sλ .+ rw_path(θ.τ_λ_rw .* θ.e_λ_rw)
            Kp[k, :] = rw_path(Diomedes.kappa_steps(θ))
        end
        append!(sm, DataFrame(mode = m, kept = m in kept, season = seasons,
                              pi_mean = vec(mean(logistic.(P); dims = 1)), pi_q05 = logistic.(q(P, 0.05)), pi_q95 = logistic.(q(P, 0.95)),
                              lambda_mean = vec(mean(exp.(Λ); dims = 1)), lambda_q05 = exp.(q(Λ, 0.05)), lambda_q95 = exp.(q(Λ, 0.95)),
                              kappa_mean = vec(mean(exp.(Kp); dims = 1)), kappa_q05 = exp.(q(Kp, 0.05)), kappa_q95 = exp.(q(Kp, 0.95))))
    end
    CSV.write(joinpath(outdir, "seasons_by_mode.csv"), round_floats!(sm))

    # the loss scalars per chain, on readable scales
    mt = DataFrame(chain = Int[], mode = String[], kept = Bool[], quantity = String[], label = String[],
                   mean = Float64[], q05 = Float64[], q95 = Float64[])
    for c in 1:nc, (key, label, x) in loss_quantities(chs[c], spec)
        push!(mt, (c, modes[c], modes[c] in kept, key, label, mean(x), quantile(x, 0.05), quantile(x, 0.95)))
    end
    CSV.write(joinpath(outdir, "modes.csv"), round_floats!(mt))

    # do the effects depend on the mode? (posterior means, against the best mode)
    eff = Dict(m => gap_effects_table(reduce(hcat, chs[idx(m)]), g) for m in labels)
    ref = labels[argmax([lj[m] for m in labels])]
    function agree(a, b)
        j = innerjoin(select(a, :label, :mean, :sd), select(b, :label, :mean => :mb); on = :label)
        d = abs.(j.mean .- j.mb)
        (; corr = round(cor(j.mean, j.mb); digits = 4), max_diff = round(maximum(d); digits = 3),
         max_diff_sd = round(maximum(d ./ j.sd); digits = 2))
    end
    effects_by_mode = [(; modes = "$(ref) vs $(m)", drivers = agree(eff[ref].competitors, eff[m].competitors),
                        cars = agree(eff[ref].machines, eff[m].machines)) for m in labels if m != ref]
    rconv(v) = (; max_rhat = round(v.max_rhat; digits = 3), min_ess = round(v.min_ess; digits = 1))
    info = [(; label = m, chains = idx(m), draws = ni * length(idx(m)), kept = m in kept,
             mean_logjoint = round(lj[m]; digits = 1),
             mean_loglikelihood = round(mean(mean(vec(chs[c][:loglikelihood])) for c in idx(m)); digits = 1),
             loss = NamedTuple(Symbol(k) => round(mean(x); digits = 4) for (k, _, x) in loss_quantities(reduce(hcat, chs[idx(m)]), spec)),
             # convergence within the mode (needs two chains)
             convergence = length(idx(m)) < 2 ? missing :
                           rconv(identified_convergence(reduce(hcat, chs[idx(m)]), g, model).overall)) for m in labels]
    return (; modes = info, kept, kept_chains = findall(in(kept), modes), effects_by_mode,
            drop_nats = MODE_DROP_NATS)
end

function export_race(model, source)
    spec = model_spec(model)
    spec.family === :pl && spec.era === :rw && spec.pace_scale ||
        error("export_site expects a pace + loss model with random-walk eras and a pace scale (pl…_rw_kappa…)")
    res, g = gap_data(source)
    fit = "gap_$(source)_$(model)"
    files = filter(isfile, ["output/$(fit)_chain$(k).jls" for k in 1:8])
    isempty(files) && error("no saved chains for $fit")
    chs = rehash_chain!.(deserialize.(files))
    outdir = joinpath(OUT, "race")
    mkpath(outdir)
    # chains grouped by where the loss mixture settled; the tables below use the
    # modes with the most posterior density, the rest are reported in modes.csv
    mi = export_modes(chs, g, spec, model, outdir)
    chain = reduce(hcat, chs[mi.kept_chains])
    pnames = model_param_names(spec)
    sc, cov = season_counts(g), LossCovariates(g, spec)
    ni, nc = size(chain[:σ])
    ndraw = ni * nc

    # one pass over the draws: everything derived from θ
    seasons = minimum(g.race_season) .+ (0:(cov.n_seasons - 1))
    lp, ll, lk = zeros(ndraw, cov.n_seasons), zeros(ndraw, cov.n_seasons), zeros(ndraw, cov.n_seasons)
    SL = spec.slopes ? zeros(ndraw, length(g.competitors)) : nothing
    DV = spec.dev ? zeros(ndraw, length(g.machines)) : nothing
    B = spec.age ? AgeCurveBasis(g) : nothing
    AC = spec.age ? zeros(ndraw, length(g.age_years)) : nothing
    CX = spec.circ ? zeros(ndraw, 3length(g.machines)) : nothing      # circuit slopes (#81), stacked by covariate
    TR = spec.track ? zeros(ndraw, 6) : nothing                         # track terms (#81): β_π, then β_λ
    # chassis and engine walks (#95): states, and each car-season's centred chassis and
    # engine parts (its rows' κ·state, averaged, then centred within season)
    cw = spec.walk ? cov.cars : nothing
    nm = length(g.machines)
    machs = vcat(g.t_mach, g.c_mach)
    nrows_m = [count(==(m), machs) for m in 1:nm]
    rowrace = vcat(g.t_race, g.c_race)
    WC = spec.walk ? zeros(ndraw, length(cw.chassis.prev)) : nothing
    WE = spec.walk ? zeros(ndraw, length(cw.engine.prev)) : nothing
    CH = spec.walk ? zeros(ndraw, nm) : nothing
    EN = spec.walk ? zeros(ndraw, nm) : nothing
    works = get(spec, :works, false)                 # works-team term (#100): its part per car-season
    WK = works ? zeros(ndraw, nm) : nothing
    dinc = get(spec, :dinc, false)                   # per-driver incident proneness (#101), logit offsets
    DI = dinc ? zeros(ndraw, length(g.competitors)) : nothing
    # speed against incident proneness (#101), both in average-season units so they compare
    # with the driver effects and ranks: the driver effect a, and the time the driver's
    # proneness adds in the same reference race, a race with the average log-odds of an
    # incident and average log loss size over all races: (π_d - π̄)·((1 - ρ)·λ̄ + ρ·λ2).
    # (Averaged over each driver's own races instead, both would mostly measure how
    # spread out and how incident-prone their era was: κ and λ differ several-fold.)
    nd = length(g.competitors)
    PP = dinc ? zeros(ndraw, nd) : nothing
    IP = dinc ? zeros(ndraw, nd) : nothing
    function carpart(x, rowstate, κr)
        v = zeros(nm)
        for (r, m) in enumerate(machs); v[m] += κr[rowrace[r]] * x[rowstate[r]]; end
        return sum_to_zero_by(v ./ nrows_m, g.mach_season, sc)
    end
    let k = 0
        for c in 1:nc, i in 1:ni
            θ = draw(chain, pnames, i, c)
            k += 1
            sπ, sλ = track_shift(θ, cov)
            lp[k, :] = θ.a_π .+ sπ .+ rw_path(θ.τ_π_rw .* θ.e_π_rw)
            ll[k, :] = θ.a_λ .+ sλ .+ rw_path(θ.τ_λ_rw .* θ.e_λ_rw)
            spec.circ && (CX[k, :] = circuit_slopes(θ.τ_cx, θ.u_cx, g, sc))
            spec.track && (TR[k, :] = vcat(θ.β_track_π, θ.β_track_λ))
            lk[k, :] = rw_path(Diomedes.kappa_steps(θ))
            spec.slopes && (SL[k, :] = career_slopes(θ.τ_slope, θ.u_slope))
            spec.dev && (DV[k, :] = dev_trends(θ.τ_dev, θ.u_dev, g, sc))
            spec.age && (AC[k, :] = age_curve(θ.τ_age, θ.e_age, B))
            if spec.walk
                xc, xe = car_walk_states(θ, cw)
                WC[k, :], WE[k, :] = xc, xe
                κr = race_pace_scale(θ, cov)
                CH[k, :], EN[k, :] = carpart(xc, cw.row_ch, κr), carpart(xe, cw.row_en, κr)
                works && (WK[k, :] = carpart([θ.β_works, 0.0], [w > 0 ? 1 : 2 for w in cw.row_works], κr))
            end
            if dinc
                o = Diomedes.driver_incident_offsets(θ)
                DI[k, :] = o
                πr, λr = Diomedes.race_loss(θ, cov, spec.loss_duration, spec.era)
                ρ, λ2 = logistic(θ.a_ρ), exp(θ.a_λ + θ.δ_λ2)
                Lbar, λbar = mean(logit.(πr)), exp(mean(log.(λr)))
                PP[k, :] = θ.σ_comp .* (θ.z_comp .- mean(θ.z_comp))
                IP[k, :] = (logistic.(Lbar .+ o) .- logistic(Lbar)) .* ((1 - ρ) * λbar + ρ * λ2)
            end
        end
    end
    q(X, p) = [quantile(c, p) for c in eachcol(X)]

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
    if dinc                                       # incident proneness: logit offset on π (#101)
        dt.incident_mean, dt.incident_q05, dt.incident_q95 = vec(mean(DI; dims = 1)), q(DI, 0.05), q(DI, 0.95)
        OV = PP .+ IP                             # overall: driver part + incident part, % of race time
        dt.pace_part_mean, dt.incident_part_mean = vec(mean(PP; dims = 1)), vec(mean(IP; dims = 1))
        dt.incident_part_q05, dt.incident_part_q95 = q(IP, 0.05), q(IP, 0.95)
        dt.overall_mean, dt.overall_q05, dt.overall_q95 = vec(mean(OV; dims = 1)), q(OV, 0.05), q(OV, 0.95)
    end
    dt = leftjoin(dt, select(eff.competitors, :label => :id, :mean, :sd, :q05, :q95); on = :id, order = :left)
    sort!(dt, :mean)
    select!(dt, :id, :name, :rows, :first_season, :last_season, :mean, :sd, :q05, :q95, Not([:id, :name, :rows, :first_season, :last_season, :mean, :sd, :q05, :q95]))
    for c in (:mean, :sd, :q05, :q95, (spec.slopes ? (:slope_mean, :slope_sd) : ())...,
              (dinc ? (:incident_mean, :incident_q05, :incident_q95, :pace_part_mean, :incident_part_mean, :incident_part_q05,
                       :incident_part_q95, :overall_mean, :overall_q05, :overall_q95) : ())...)
        dt[!, c] = r4(dt[!, c])
    end
    CSV.write(joinpath(outdir, "drivers.csv"), dt)

    # car-seasons (per-draw residuals; with walks the effect is residual + chassis + engine)
    Bm = reduce(vcat, [sum_to_zero_by(Diomedes.car_residuals(draw(chain, pnames, i, c), cov), g.mach_season, sc)'
                       for c in 1:nc for i in 1:ni])                    # loop order: i within c
    TOT = spec.walk ? Bm .+ CH .+ EN : Bm                                     # draws × cars
    works && (TOT = TOT .+ WK)
    ct = DataFrame(id = g.machines, season = [seasons[s] for s in g.mach_season],
                   rows = [count(==(m), machs) for m in eachindex(g.machines)])
    ct.constructor = [replace(m, Regex("_$(s)\$") => "") for (m, s) in zip(ct.id, ct.season)]
    if spec.dev                                   # g.machines order, as DV's columns
        ct.dev_mean, ct.dev_sd = vec(mean(DV; dims = 1)), vec(std(DV; dims = 1))
    end
    if spec.circ                                  # circuit slopes (#81), % per sd of each covariate
        nm = length(g.machines)
        for (k, n) in enumerate(cov.circuits.x_names)
            X = CX[:, ((k - 1) * nm + 1):(k * nm)]
            ct[!, Symbol(n, "_mean")], ct[!, Symbol(n, "_sd")] = vec(mean(X; dims = 1)), vec(std(X; dims = 1))
        end
    end
    if spec.walk                                  # totals from the draws, in g.machines order
        ct.mean, ct.sd, ct.q05, ct.q95 = vec(mean(TOT; dims = 1)), vec(std(TOT; dims = 1)), q(TOT, 0.05), q(TOT, 0.95)
        ct.chassis_mean, ct.engine_mean, ct.residual_mean = vec(mean(CH; dims = 1)), vec(mean(EN; dims = 1)), vec(mean(Bm; dims = 1))
        works && (ct.works_mean = vec(mean(WK; dims = 1)))
    else
        ct = leftjoin(ct, select(eff.machines, :label => :id, :mean, :sd, :q05, :q95); on = :id, order = :left)
    end
    sort!(ct, [:season, :mean])
    for c in (:mean, :sd, :q05, :q95, (spec.dev ? (:dev_mean, :dev_sd) : ())...,
              (spec.walk ? (:chassis_mean, :engine_mean, :residual_mean) : ())..., (works ? (:works_mean,) : ())...,
              (spec.circ ? Tuple(Symbol(n, s) for n in cov.circuits.x_names for s in ("_mean", "_sd")) : ())...)
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

    # track terms (#81): incident odds and loss-size multipliers, against a dry race at an open circuit
    if spec.track
        names_t = cov.circuits.track_names
        labels = Dict(:tight => "Walls (tight score 1 vs 0)", :mixed => "Mixed race (vs dry)", :wet => "Wet race (vs dry)")
        tt = DataFrame(term = string.(names_t), label = [labels[n] for n in names_t],
                       share = r4(vec(mean(cov.circuits.track; dims = 1))))
        for (j, (col, off)) in enumerate(((:pi_odds, 0), (:lambda_mult, 3)))
            E = exp.(TR[:, off .+ (1:3)])
            tt[!, Symbol(col, "_mean")] = r4(vec(mean(E; dims = 1)))
            tt[!, Symbol(col, "_q05")] = r4(q(E, 0.05))
            tt[!, Symbol(col, "_q95")] = r4(q(E, 0.95))
        end
        CSV.write(joinpath(outdir, "track.csv"), tt)
    end

    # age curve
    if spec.age
        CSV.write(joinpath(outdir, "age_curve.csv"),
                  DataFrame(age = g.age_years, rows = Int.(g.age_rows), mean = r4(vec(mean(AC; dims = 1))),
                            q05 = r4(q(AC, 0.05)), q95 = r4(q(AC, 0.95))))
    end

    # chassis and engine walks (#95)
    if spec.walk
        wt = DataFrame()
        for (family, ws, W) in (("chassis", cw.chassis, WC), ("engine", cw.engine, WE))
            parts = split.(ws.names)
            append!(wt, DataFrame(family = family, walk = first.(parts), season = parse.(Int, last.(parts)),
                                  mean = r4(vec(mean(W; dims = 1))), q05 = r4(q(W, 0.05)), q95 = r4(q(W, 0.95)),
                                  own_change = Bool.(ws.own), reg_change = Bool.(ws.reg)))
        end
        CSV.write(joinpath(outdir, "walks.csv"), wt)
    end

    # per-draw effects, in the draw order of the loop above (iterations within chains),
    # as gap_effects_table computes them (centred drivers and cars)
    stackd(sym) = stack(vec(chain[sym]); dims = 1)
    hasp(sym) = any(vn -> string(vn) == string(sym), Turing.FlexiChains.parameters(chain))
    A = hasp(:a_comp) ? mapslices(sum_to_zero, stackd(:a_comp); dims = 2) :         # draws × drivers
        mapslices(sum_to_zero, stackd(:z_comp); dims = 2) .* vec(chain[:σ_comp])
    K = exp.(lk)                                                              # draws × seasons
    # the stacked draws must line up with the loop's (i within c) order
    for (i, c) in ((1, 1), (ni, 1), (1, nc))
        k = (c - 1) * ni + i
        θk = draw(chain, pnames, i, c)
        @assert A[k, :] ≈ (hasp(:a_comp) ? sum_to_zero(θk.a_comp) : θk.σ_comp .* sum_to_zero(θk.z_comp))
        @assert lk[k, :] ≈ rw_path(Diomedes.kappa_steps(draw(chain, pnames, i, c)))
    end

    # rank probabilities among drivers with ≥ 20 results (ranked in every draw). With
    # incident proneness (#101) drivers are ranked on race pace: the driver effect plus
    # the time their proneness adds in the reference race (see above), since avoiding
    # time lost to incidents is part of being fast in a race; the `_pace` columns rank
    # the driver effect alone (open-track pace)
    nrows = [count(==(d), comps) for d in eachindex(g.competitors)]
    elig = findall(>=(20), nrows)
    function ranks(X)
        R = zeros(Int, ndraw, length(elig))
        for k in 1:ndraw
            R[k, sortperm(X[k, elig])] .= 1:length(elig)
        end
        return R
    end
    R = ranks(dinc ? A .+ IP : A)
    rt = DataFrame(id = g.competitors[elig], name = [dname[c] for c in g.competitors[elig]], rows = nrows[elig],
                   p_top1 = r4(vec(mean(R .== 1; dims = 1))), p_top3 = r4(vec(mean(R .<= 3; dims = 1))),
                   p_top10 = r4(vec(mean(R .<= 10; dims = 1))),
                   rank_median = [median(c) for c in eachcol(R)],
                   rank_q05 = [quantile(c, 0.05) for c in eachcol(R)], rank_q95 = [quantile(c, 0.95) for c in eachcol(R)])
    if dinc
        Rp = ranks(A)
        rt.p_top1_pace, rt.p_top3_pace = r4(vec(mean(Rp .== 1; dims = 1))), r4(vec(mean(Rp .<= 3; dims = 1)))
        rt.p_top10_pace, rt.rank_median_pace = r4(vec(mean(Rp .<= 10; dims = 1))), [median(c) for c in eachcol(Rp)]
    end
    sort!(rt, :rank_median)
    CSV.write(joinpath(outdir, "ranks.csv"), rt)

    # driver and car parts of each driver-season's pace (that season's % units)
    rowseason = vcat(g.race_season[g.t_race], g.race_season[g.c_race])
    dm = Dict{Tuple{Int,Int},Int}()
    for (d, m) in zip(comps, machs)
        dm[(d, m)] = get(dm, (d, m), 0) + 1
    end
    qs(x) = (r4(mean(x)), r4(quantile(x, 0.05)), r4(quantile(x, 0.95)))
    ds = DataFrame(driver = String[], name = String[], season = Int[], car = String[], constructor = String[], rows = Int[],
                   driver_mean = Float64[], driver_q05 = Float64[], driver_q95 = Float64[],
                   car_mean = Float64[], car_q05 = Float64[], car_q95 = Float64[],
                   total_mean = Float64[], total_q05 = Float64[], total_q95 = Float64[])
    # career terms (age curve, slopes) for each driver-season: their mean over the
    # pair's rows, per draw, so the driver part is that season's driver pace
    keys_dm = collect(keys(dm))
    pidx = Dict(p => i for (i, p) in enumerate(keys_dm))
    rowpair = [pidx[(d, m)] for (d, m) in zip(comps, machs)]
    H = zeros(ndraw, length(keys_dm))
    if spec.age || spec.slopes
        cr = CareerRows(g)
        for k in 1:ndraw
            hk = driver_offsets(cr, spec.age ? AC[k, :] : nothing, spec.slopes ? SL[k, :] : nothing)
            for (r, p) in enumerate(rowpair)
                H[k, p] += hk[r]
            end
        end
        H ./= [dm[p] for p in keys_dm]'
    end
    for (pj, (d, m)) in enumerate(keys_dm)
        n = dm[(d, m)]
        si = g.mach_season[m]
        dpart = K[:, si] .* (A[:, d] .+ H[:, pj])
        cpart = TOT[:, m]
        push!(ds, (g.competitors[d], dname[g.competitors[d]], seasons[si], g.machines[m], ct.constructor[findfirst(==(g.machines[m]), ct.id)], n,
                   qs(dpart)..., qs(cpart)..., qs(dpart .+ cpart)...))
    end
    sort!(ds, [:name, :season, :car])
    CSV.write(joinpath(outdir, "driver_seasons.csv"), ds)

    # provenance
    ic = identified_convergence(chain, g, model)
    rconv(v) = (; max_rhat = round(v.max_rhat; digits = 3), min_ess = round(v.min_ess; digits = 1))
    jol = match(r"^jolpica(\d{4})-(\d{4})(\+partial)?$", source)
    data = jol === nothing ?
        "F1 $(first(seasons))–$(last(seasons)), Ergast CSV, all classified finishers (lapped as intervals)" :
        "F1 $(first(seasons))–$(last(seasons)), Jolpica API, all classified finishers (lapped as intervals, or at exact " *
        "gaps from their own times from 2023)" * (jol[3] === nothing ? "" : ", plus retirements from 1996 as partial gaps")
    nlap_exact = hasproperty(g.rows, :lapped_exact) ? count(g.rows.lapped_exact) : 0
    npartial = hasproperty(g.rows, :partial) ? count(Bool.(g.rows.partial)) : 0
    manifest = (; model, fit, chains = nc, draws_per_chain = ni, chains_fitted = length(chs), chains_used = mi.kept_chains, data,
                rows = (; timed = length(g.y) - npartial - nlap_exact, lapped = length(g.lo), lapped_exact = nlap_exact,
                        partial = npartial),
                convergence = (; rconv(ic.overall)..., divergences = count(identity, chain[:numerical_error]),
                               groups = NamedTuple(k => (; rconv(v)..., v.worst) for (k, v) in pairs(ic) if k !== :overall)),
                modes = mi.modes, kept_modes = mi.kept, effects_by_mode = mi.effects_by_mode,
                circuits = spec.circ || spec.track ?
                    (; covariates = string.(cov.circuits.x_names),
                     covariate_sd = r4(vec(std(cov.circuits.x_raw; dims = 1))),
                     tau_cx = spec.circ ? [(; mean = r4(mean(x)), q05 = r4(quantile(x, 0.05)), q95 = r4(quantile(x, 0.95)))
                                           for x in eachrow(stack(vec(chain[:τ_cx])))] : missing,
                     seasons_note = "π and λ by season are for a dry race at a circuit of average tightness") : missing,
                walks = spec.walk ?
                    (; units = "% of race time in average-season units",
                     NamedTuple(Symbol(lbl) => (; mean = r4(mean(x)), q05 = r4(quantile(x, 0.05)), q95 = r4(quantile(x, 0.95)))
                                for (lbl, x) in (("chassis_step", vec(chain[:τ_ch])), ("chassis_first", vec(chain[:σ0_ch])),
                                                 ("engine_step", vec(chain[:τ_en])), ("engine_first", vec(chain[:σ0_en])),
                                                 ("residual_sd", vec(chain[:σ_mach])),
                                                 ("ownership_multiplier", exp.(vec(chain[:δ_own]))),
                                                 ("regulation_multiplier", exp.(vec(chain[:δ_reg])))))...,
                     states = (; chassis = length(cw.chassis.prev), engine = length(cw.engine.prev))) : missing,
                works = works ? (; units = "% of race time in average-season units (negative = faster)",
                                 beta = (; mean = r4(mean(vec(chain[:β_works]))), q05 = r4(quantile(vec(chain[:β_works]), 0.05)),
                                         q95 = r4(quantile(vec(chain[:β_works]), 0.95))),
                                 rows = Int(sum(cw.row_works))) : missing,
                driver_incidents = dinc ? (; units = "logit scale, offsets on the incident probability",
                                           tau = (; mean = r4(mean(vec(chain[:τ_dinc]))), q05 = r4(quantile(vec(chain[:τ_dinc]), 0.05)),
                                                  q95 = r4(quantile(vec(chain[:τ_dinc]), 0.95)))) : missing,
                exported_by = "scripts/export_site.jl race")
    write_manifest(outdir, manifest)
    println("wrote $(outdir): $(nrow(dt)) drivers, $(nrow(ct)) car-seasons, $(nrow(st)) seasons; ",
            "$(nc) chains × $(ni) draws; R-hat $(manifest.convergence.max_rhat), min ESS $(manifest.convergence.min_ess); ",
            "modes $(join(["$(m.label) (chains $(join(m.chains, ", ")), mean log density $(m.mean_logjoint))" for m in mi.modes], ", ")); ",
            "kept $(join(mi.kept, ", "))")
end

function export_retire(pre)
    dir = joinpath(OUT, "retire")
    mkpath(dir)
    cars = round_floats!(CSV.read("output/$(pre)_cars.csv", DataFrame))
    cars.constructor = [replace(m, Regex("_$(s)\$") => "") for (m, s) in zip(cars.machine, cars.season)]
    CSV.write(joinpath(dir, "cars.csv"), select(cars, :machine => :id, :constructor, Not([:machine, :constructor])))
    CSV.write(joinpath(dir, "drivers.csv"), select(round_floats!(CSV.read("output/$(pre)_drivers.csv", DataFrame)),
                                                   :driver => :id, Not(:driver)))
    CSV.write(joinpath(dir, "seasons.csv"), round_floats!(CSV.read("output/$(pre)_seasons.csv", DataFrame)))
    log = "output/$(pre)_combine.txt"
    jol = match(r"jolpica(\d{4})-(\d{4})", pre)
    src = jol === nothing ? "F1 1950–2019 (Ergast CSV)" :
          "F1 $(jol[1])–$(jol[2]) (Jolpica API; retirements with no recorded cause count under the total hazard)"
    write_manifest(dir, (; model = "retirement_effects (competing risks: mechanical, incident, collision)",
                         fit = pre, data = src * ": " * inner(first(readlines(log))), convergence = log_convergence(log),
                         exported_by = "scripts/export_site.jl retire"))
    println("wrote $dir")
end

# The lap fit's chains (checkpoint blocks, one file per 100 draws, #62/#73)
function lap_chains(tag)
    chain_files(k) = sort(filter(f -> occursin(Regex("^$(tag)_chain$(k)_block\\d+\\.jls\$"), f), readdir("output")),
                          by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
    ks = [k for k in 1:8 if !isempty(chain_files(k))]
    return [reduce(vcat, [Diomedes.block_chain(joinpath("output", f)) for f in chain_files(k)]) for k in ks]
end

# Chains grouped into modes on the scalars that split them (as for the race model, #64):
# the per-race trend spread and the lap-noise mixture. Per-mode effect tables come
# from `lap_fit.jl … combine <chains of the mode>`, saved as <tag>_mode<X>_<table>.csv.
function export_lap_modes(tag, dir)
    chs = lap_chains(tag)
    length(chs) < 2 && return nothing
    all = reduce(hcat, chs)
    modes = chain_modes(all, [:τ_δ, :σ, :a_π, :a_λ])
    labels = unique(modes)
    idx(m) = findall(==(m), modes)
    lj = Dict(m => mean(mean(vec(chs[c][:logjoint])) for c in idx(m)) for m in labels)
    best = maximum(values(lj))
    kept = [m for m in labels if lj[m] > best - MODE_DROP_NATS]
    quantities(c) = [("tau_delta", "Spread of per-race lap-time trends (τ_δ)", vec(c[:τ_δ])),
                     ("sigma", "Lap noise σ (%)", vec(c[:σ])),
                     ("pi_base", "Lost-time lap probability", logistic.(vec(c[:a_π]))),
                     ("lambda_base", "Lost-time lap mean (%)", exp.(vec(c[:a_λ])))]
    mt = DataFrame(chain = Int[], mode = String[], kept = Bool[], quantity = String[], label = String[],
                   mean = Float64[], q05 = Float64[], q95 = Float64[])
    for c in eachindex(chs), (key, label, x) in quantities(chs[c])
        push!(mt, (c, modes[c], modes[c] in kept, key, label, mean(x), quantile(x, 0.05), quantile(x, 0.95)))
    end
    CSV.write(joinpath(dir, "modes.csv"), round_floats!(mt))
    # do the effects depend on the mode?
    ref = labels[argmax([lj[m] for m in labels])]
    tables = (drivers = :driver, cars = :machine, tyremgmt_drivers = :driver, tyremgmt_cars = :machine,
              starts_drivers = :driver, starts_cars = :machine)
    function agree(t, key, a, b)
        fa, fb = "output/$(tag)_mode$(a)_$(t).csv", "output/$(tag)_mode$(b)_$(t).csv"
        (isfile(fa) && isfile(fb)) || return missing
        j = innerjoin(select(CSV.read(fa, DataFrame), key, :mean, :sd), select(CSV.read(fb, DataFrame), key, :mean => :mb); on = key)
        d = abs.(j.mean .- j.mb)
        (; n = nrow(j), corr = round(cor(j.mean, j.mb); digits = 4), max_diff = round(maximum(d); digits = 3),
         median_diff_sd = round(median(d ./ j.sd); digits = 2), max_diff_sd = round(maximum(d ./ j.sd); digits = 2))
    end
    effects_by_mode = [(; modes = "$(ref) vs $(m)", NamedTuple(t => agree(t, k, ref, m) for (t, k) in pairs(tables))...)
                       for m in labels if m != ref]
    mlog(m) = "output/$(tag)_mode$(m)_combine.txt"
    info = [(; label = m, chains = idx(m), draws = sum(length(vec(chs[c][:logjoint])) for c in idx(m)), kept = m in kept,
             mean_logjoint = round(lj[m]; digits = 1),
             scalars = NamedTuple(Symbol(k) => round(mean(x); digits = 4) for (k, _, x) in quantities(reduce(hcat, chs[idx(m)]))),
             convergence = isfile(mlog(m)) ? log_convergence(mlog(m)) : missing) for m in labels]
    return (; modes = info, kept, effects_by_mode, drop_nats = MODE_DROP_NATS)
end

# `source`: where the fit's lap times came from. Fits from #75 on read every season
# from Jolpica (`jolpica`); the default infers the earlier split from the seasons.
function export_laps(tag; source = nothing)
    dir = joinpath(OUT, "laps")
    mkpath(dir)
    d = round_floats!(CSV.read("output/$(tag)_drivers.csv", DataFrame))
    c = round_floats!(CSV.read("output/$(tag)_cars.csv", DataFrame))
    rename!(d, :driver => :id)
    rename!(c, :machine => :id)
    # career slopes and in-season development, if the fit has them (#12)
    sl, dv = "output/$(tag)_slopes.csv", "output/$(tag)_dev.csv"
    isfile(sl) && (d = leftjoin(d, select(round_floats!(CSV.read(sl, DataFrame)), :driver => :id, :first_season, :last_season,
                                          :mean => :slope_mean, :sd => :slope_sd); on = :id, order = :left))
    isfile(dv) && (c = leftjoin(c, select(round_floats!(CSV.read(dv, DataFrame)), :machine => :id, :mean => :dev_mean,
                                          :sd => :dev_sd); on = :id, order = :left))
    # tyre management and starts, if the fit has them (#66)
    for (t, cols) in (("tyremgmt", (:tyre_mean, :tyre_sd)), ("starts", (:start_mean, :start_sd)))
        fd, fc = "output/$(tag)_$(t)_drivers.csv", "output/$(tag)_$(t)_cars.csv"
        extra = t == "starts" ? [:starts] : Symbol[]
        isfile(fd) && (d = leftjoin(d, select(round_floats!(CSV.read(fd, DataFrame)), :driver => :id, extra...,
                                              :mean => cols[1], :sd => cols[2]); on = :id, order = :left))
        isfile(fc) && (c = leftjoin(c, select(round_floats!(CSV.read(fc, DataFrame)), :machine => :id,
                                              (t == "starts" ? [:starts] : Symbol[])...,
                                              :mean => cols[1], :sd => cols[2]); on = :id, order = :left))
    end
    # car-season circuit slopes (#14), if the fit has them: % per sd of each covariate
    cx = "output/$(tag)_circuits.csv"
    isfile(cx) && (c = leftjoin(c, select(round_floats!(CSV.read(cx, DataFrame)), :machine => :id,
                                          r"^(speed|street|temp)_(mean|sd)$"); on = :id, order = :left))
    CSV.write(joinpath(dir, "drivers.csv"), d)
    CSV.write(joinpath(dir, "cars.csv"), c)
    # population age curve (#12), if the fit has it
    ac = "output/$(tag)_age_curve.csv"
    isfile(ac) ? CSV.write(joinpath(dir, "age_curve.csv"), round_floats!(CSV.read(ac, DataFrame))) :
                 rm(joinpath(dir, "age_curve.csv"); force = true)
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
    m = match(r"(\d{4})-(\d{4})_(recorded|inferred)(_notyres)?(_slopes)?(_dev)?(_lead)?(_tyremgmt)?(_starts)?(_age)?(_circ)?", tag)
    first_y, last_y = parse(Int, m[1]), parse(Int, m[2])
    src = source == "jolpica" ? "Jolpica lap times" :
          last_y <= 2019 ? "Ergast lap times" : first_y > 2019 ? "Jolpica lap times" : "Ergast lap times to 2019, Jolpica from 2020"
    terms = join(filter(!isempty, [m[4] === nothing ? "stint age" : "", m[5] === nothing ? "" : "career slopes",
                                   m[6] === nothing ? "" : "in-season development", m[7] === nothing ? "" : "lead laps",
                                   m[8] === nothing ? "" : "tyre management", m[9] === nothing ? "" : "starts",
                                   m[10] === nothing ? "" : "age curve", m[11] === nothing ? "" : "circuit slopes"]), ", ")
    modes = export_lap_modes(tag, dir)
    write_manifest(dir, (; model = "lap_effects" * (isempty(terms) ? "" : " ($(terms))"), fit = tag,
                         data = "F1 $(first_y)–$(last_y) clean laps ($(src), $(m[3]) pit stops): " * inner(first(readlines(log))),
                         race_model = JSON3.read(read(joinpath(race, "manifest.json"), String)).model,
                         convergence = log_convergence(log),
                         (modes === nothing ? (;) : (; modes = modes.modes, kept_modes = modes.kept,
                                                     effects_by_mode = modes.effects_by_mode,
                                                     mode_drop_nats = modes.drop_nats))...,
                         exported_by = "scripts/export_site.jl laps"))
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
    export_race(get(ARGS, 2, "pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc"), get(ARGS, 3, "jolpica1950-2026+partial"))
elseif mode == "retire"
    export_retire(get(ARGS, 2, "retire_jolpica1950-2025"))
elseif mode == "laps"
    export_laps(get(ARGS, 2, "laps_1996-2025_inferred_slopes_dev_lead_tyremgmt_starts_age_circ"); source = get(ARGS, 3, nothing))
elseif mode == "wrc"
    export_wrc(get(ARGS, 2, "pl_dur_rw_kappa_cdrv_tdrv_fgam"))
else
    error("unknown mode $mode: race, retire, laps or wrc")
end
