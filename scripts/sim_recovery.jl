# Simulation recovery for the race model (#120): fit synthetic results with known driver,
# car-season and career-slope effects, and check they come back.
#
#   julia --project scripts/sim_recovery.jl chain <k> [model=pl_kappa_slope_dev] [seed=1] [draws=1000]
#   julia --project scripts/sim_recovery.jl combine [n=4] [model=…] [seed=1]
#
# The synthetic grid: 8 seasons, 10 teams of 2 drivers, 20 races each. Drivers have birth
# years; each season the 3 oldest retire, 3 rookies join, and 4 pairs swap teams. Planted:
#   - a dominant car: team T is 2.0 % faster than its baseline in seasons 4–8;
#   - a star driver (−1.0 %) who drives an average team in seasons 1–3 and joins T in season 4,
#     so from then on star and dominant car share every result (the #12 concern);
#   - a rookie who improves fast: career slope −2.5 % per decade of age (others ~ N(0, 0.3)).
# With a walk model (`model=…_walk…`, e.g. pl_kappa_slope_walk_walk1_works) the car is generated
# as the published race model sees it: each team its own chassis works walking from season to
# season (first state ~ N(0, 0.9), steps 0.4, ×e^0.5 across a 2014 rule change), three engine
# builders walking too (0.3, steps 0.2), each the works supplier of one team (works term −0.3)
# and customer supplier of others, plus a residual per car-season (0.3). Team 1's chassis takes
# a −2.0 step in 2013 (the dominant car) and walks on from there. Synthetic car tables replace
# the F1DB ones, and recovery is checked on each car-season's total.
# Pace + loss noise as the model assumes (σ 0.3 %, incidents with probability 0.2 and mean
# loss 3 %). `combine` reports, against the truth (centred as the model centres it): driver
# and car-season correlations and 90 % interval coverage, and the star, the dominant car and
# the improving driver one by one.
using Diomedes, DataFrames, Dates, Random, Serialization, Statistics, Printf
using Turing: NUTS
import Turing

opt(k, d) = (i = findfirst(a -> startswith(a, k * "="), ARGS); i === nothing ? d : String(split(ARGS[i], "="; limit = 2)[2]))
model = opt("model", "pl_kappa_slope_dev"); seed = parse(Int, opt("seed", "1"))
args = filter(a -> !occursin('=', a), ARGS)
tag = "sim_recovery_$(model)_s$(seed)"

function simulate(rng; walks = false)
    S, T, R = 8, 10, 20
    nd = 2T + 3(S - 1)                                   # 20 starting drivers, 3 rookies a season
    a = 0.5 .* randn(rng, nd)
    slope = 0.3 .* randn(rng, nd)                        # % per decade of age
    birth = [rand(rng, 1975:1990) for _ in 1:nd]
    star, rook = 1, 2T + 1                               # star starts in team 2; the first rookie improves
    a[star] = -1.0; slope[rook] = -2.5; birth[rook] = 1992 # a 19-year-old in season 2 (2011)
    if walks
        eng = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]            # engine builder of each team; works teams 1, 4, 7
        isworks = [t in (1, 4, 7) for t in 1:T]
        xch = zeros(T, S); xen = zeros(3, S)
        for t in 1:T, s in 1:S
            xch[t, s] = s == 1 ? 0.9 * randn(rng) : xch[t, s - 1] + 0.4 * (s == 5 ? exp(0.5) : 1.0) * randn(rng) - (t == 1 && s == 4 ? 2.0 : 0.0)
        end
        for e in 1:3, s in 1:S
            xen[e, s] = s == 1 ? 0.3 * randn(rng) : xen[e, s - 1] + 0.2 * (s == 5 ? exp(0.5) : 1.0) * randn(rng)
        end
        car = [xch[t, s] + xen[eng[t], s] - 0.3 * isworks[t] + 0.3 * randn(rng) for t in 1:T, s in 1:S]
    else
        eng, isworks = nothing, nothing
        base = 0.8 .* randn(rng, T)
        car = [base[t] + 0.4 * randn(rng) - (t == 1 && s >= 4 ? 2.0 : 0.0) for t in 1:T, s in 1:S]
    end
    seat = collect(1:2T)                                 # seat[j] = driver in seat j (team = cld(j, 2))
    seat[[1, 3]] = seat[[3, 1]]                          # star (driver 1) starts in team 2
    nextrook = 2T + 1
    rows = DataFrame()
    truth_seat = Dict{Tuple{Int,Int},Int}()
    for s in 1:S
        if s > 1
            # 3 oldest (excluding star and improving rookie) retire, rookies take their seats
            cand = sort([j for j in eachindex(seat) if seat[j] ∉ (star, rook)]; by = j -> birth[seat[j]])
            for j in cand[1:3]; seat[j] = nextrook; nextrook += 1; end
            # 4 swaps between random seats (never moving the star except into team 1 in season 4)
            for _ in 1:4
                i, j = rand(rng, 3:2T, 2)
                seat[i] ∉ (star,) && seat[j] ∉ (star,) && (seat[[i, j]] = seat[[j, i]])
            end
            if s == 4                                    # the star joins the dominant team
                js = findfirst(==(star), seat); seat[[1, js]] = seat[[js, 1]]
            end
        end
        for r in 1:R
            γ = randn(rng)
            for j in eachindex(seat)
                d, t = seat[j], cld(j, 2)
                date = Date(2009 + s, 3, 1) + Day(10r)             # the model needs dates for ages
                age = floor(Int, Dates.value(date - Date(birth[d], 6, 1)) / 365.2425); truth_seat[(s, d)] = t
                loss = rand(rng) < 0.2 ? -3.0 * log(rand(rng)) : 0.0
                gap = γ + a[d] + slope[d] * (age - 30) / 10 + car[t, s] + 0.3 * randn(rng) + loss
                push!(rows, (; series = "sim", season = 2009 + s, round = r, event_id = "$(2009 + s)-$(lpad(r, 2, '0'))",
                             event_name = "", event_date = date, age_true = age, stage_id = "race", competitor_id = "d$d", competitor_name = "d$d",
                             competitor_birth = Date(birth[d], 6, 1), codriver_id = missing, machine_id = "t$t",
                             class = missing, gap, position = missing, status = "Finished", classified = true))
            end
        end
    end
    L, lap = 60, 90_000.0
    rows.time_ms = Vector{Union{Missing,Float64}}(undef, nrow(rows)); rows.laps = fill(L, nrow(rows))
    for grp in groupby(rows, :event_id)
        T_ = L * lap .* exp.(grp.gap ./ 100); grp.time_ms .= T_
    end
    rows.suspended_ms .= 0.0; rows.own_time_ms .= missing
    return rows, (; a, slope, car, star, rook, birth, eng, isworks)
end

# synthetic car tables in car_table()'s layout: team t is chassis works "w<t>" with engine "e<k>"
function sim_cars(rows, truth)
    entries = DataFrame(season = rows.season, round = rows.round, driver_id = rows.competitor_id,
                        machine_id = rows.machine_id, laps = rows.laps, f1db_driver = rows.competitor_id,
                        f1db_constructor = rows.machine_id,
                        f1db_engine = ["e$(truth.eng[parse(Int, m[2:end])])" for m in rows.machine_id])
    T = length(truth.eng)
    works = DataFrame(f1db_constructor = ["t$t" for t in 1:T], first_year = 2000, last_year = 2030,
                      works_id = ["w$t" for t in 1:T], uncertain = false, notes = "", sources = "")
    engines = DataFrame(f1db_engine = ["e$k" for k in 1:3], first_year = 2000, last_year = 2030,
                        builder_id = ["e$k" for k in 1:3], uncertain = false, notes = "", sources = "")
    ownership = DataFrame(works_id = String[], season = Int[], change = String[], uncertain = Bool[], notes = String[], sources = String[])
    regulations = DataFrame(season = [2014], chassis = [true], engine = [true], change = ["synthetic"], sources = [""])
    works_engines = DataFrame(builder_id = ["e1", "e2", "e3"], works_id = ["w1", "w4", "w7"], first_year = 2000,
                              last_year = 2030, uncertain = false, notes = "", sources = "")
    return (; entries, works, engines, ownership, regulations, works_engines)
end
spec = model_spec(model)
rows, truth = simulate(Xoshiro(seed); walks = spec.walk)
g = prepare_gaps(rows)
cars = spec.walk ? sim_cars(rows, truth) : nothing
fit(; kw...) = fit_paceloss(g; pace_scale = spec.pace_scale, slopes = spec.slopes, dev = spec.dev, age = spec.age,
                            big_loss = spec.big_loss, loss_duration = spec.loss_duration, walk = spec.walk, walk1 = spec.walk1,
                            works = spec.works, crel = get(spec, :crel, false), cars, kw...)

if get(args, 1, "") == "chain"
    k = parse(Int, args[2])
    println(g); flush(stdout)
    chain = fit(; rng = Xoshiro(100 + k), n_samples = parse(Int, opt("draws", "1000")), progress = false,
                progress_log = stdout, sampler = NUTS(0.8; adtype = Diomedes.gap_sampler().adtype))
    serialize("output/$(tag)_chain$(k).jls", chain)
    println("chain $k done: divergences ", count(identity, chain[:numerical_error]))
    exit()
end

n = parse(Int, opt("n", "4"))
chain = reduce(hcat, [rehash_chain!(deserialize("output/$(tag)_chain$(k).jls")) for k in 1:n])
println(g); println("divergences: ", count(identity, chain[:numerical_error]))
lcov = LossCovariates(g; cars)
ic = identified_convergence(chain, g, model; cov = lcov)
@printf("convergence: max R-hat %.3f, min ESS %.0f\n", ic.overall.max_rhat, ic.overall.min_ess)
cols, label = identified_draws(chain, g, model; cov = lcov)
if spec.walk        # recovery on each car-season's total: residual + κ·(its chassis and engine states + works)
    cw = lcov.cars; mach = vcat(g.t_mach, g.c_mach); nm = length(g.machines)
    sh(idx, n) = (W = zeros(nm, n); for (i, m) in enumerate(mach); W[m, idx[i]] += 1; end; W ./ sum(W; dims = 2))
    Wc, We = sh(cw.row_ch, length(cw.chassis.prev)), sh(cw.row_en, length(cw.engine.prev))
    wk = [mean(cw.row_works[mach .== m]) for m in 1:nm]
    pn = Tuple(Symbol.(string.(Turing.FlexiChains.parameters(chain))))
    sc = Diomedes.season_counts(g)
    tot = zeros(size(cols[:cars])...)
    for c in axes(tot, 2), i in axes(tot, 1)
        θ = Diomedes.draw(chain, pn, i, c)
        xc, xe = car_walk_states(θ, cw)
        κm = Diomedes.season_pace_scale(θ)[lcov.mach_season]
        t = Diomedes.car_residuals(θ, lcov) .+ κm .* (Wc * xc .+ We * xe .+ (haskey(θ, :β_works) ? θ.β_works : 0.0) .* wk)
        tot[i, c, :] = Diomedes.sum_to_zero_by(t, g.mach_season, sc)
    end
    cols[:cars] = tot
end
draws(gr) = reshape(cols[gr], :, size(cols[gr], 3))                     # (draws·chains) × levels
q(x, p) = quantile(x, p)
function report(name, est, tru, labs)
    m = vec(mean(est; dims = 1)); lo = [q(c, 0.05) for c in eachcol(est)]; hi = [q(c, 0.95) for c in eachcol(est)]
    b = cov(m, tru) / var(tru)          # below 1: the estimates are pulled towards zero (shrinkage)
    @printf("%-12s n %3d: corr %.3f, 90 %% coverage %.0f %%, mean |error| %.2f, slope of estimate on truth %.2f\n",
            name, length(tru), cor(m, tru), 100mean(lo .<= tru .<= hi), mean(abs.(m .- tru)), b)
    return m, lo, hi
end
# drivers: model centres across drivers, in average-season units
di = [parse(Int, c[2:end]) for c in g.competitors]
# the model's driver effect is the level at the driver's own mean age over their results
mage = [mean(r.age_true for r in eachrow(rows) if r.competitor_id == "d$d") for d in di]
ta = truth.a[di] .+ truth.slope[di] .* (mage .- 30) ./ 10
ta .-= mean(ta)
md, lod, hid = report("drivers", draws(:drivers), ta, g.competitors)
# car-seasons: centred within season
mk = [match(r"^t(\d+)_(\d{4})$", m) for m in g.machines]
tc = [truth.car[parse(Int, x[1]), parse(Int, x[2]) - 2009] for x in mk]
ss = [parse(Int, x[2]) for x in mk]
for s in unique(ss); i = ss .== s; tc[i] .-= mean(tc[i]); end
mc, loc, hic = report("car-seasons", draws(:cars), tc, g.machines)
if haskey(cols, :slopes)
    tsl = truth.slope[di] .- mean(truth.slope[di])
    ms, los, his = report("slopes", draws(:slopes), tsl, g.competitors)
    j = findfirst(==("d$(truth.rook)"), g.competitors)
    @printf("improving driver d%d: slope truth %.2f, estimate %.2f [%.2f, %.2f] %% per decade\n", truth.rook, tsl[j], ms[j], los[j], his[j])
end
j = findfirst(==("d$(truth.star)"), g.competitors)
@printf("star d%d: truth %.2f, estimate %.2f [%.2f, %.2f]\n", truth.star, ta[j], md[j], lod[j], hid[j])
for s in 2010:2017
    i = findfirst(==("t1_$s"), g.machines)
    @printf("team t1 %d: truth %.2f, estimate %.2f [%.2f, %.2f]%s\n", s, tc[i], mc[i], loc[i], hic[i], s >= 2013 ? "  (dominant)" : "")
end
