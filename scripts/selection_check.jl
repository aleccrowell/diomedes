# Does dropping retirements bias the pace effects? (#13, stage 2)
#
#   julia --project scripts/selection_check.jl calib                  # check the partial-gap calibration
#   julia --project scripts/selection_check.jl chain <a|b> <k>        # one chain (checkpointed, resumable)
#   julia --project scripts/selection_check.jl compare [n=4] [n_b]    # residuals and effect shifts
#                                         (chains 1..n, or lists per fit, e.g. `compare 1,2,3,4 1,2,4`)
#
# F1 1996–2019 (lap times exist), model MODEL. Retirements that completed at
# least a quarter of the distance enter as calibrated partial gaps (see
# src/partial.jl). Both fits use one GapData (finishers + partial rows), so their
# parameters line up:
#   a: finishers only (the partial rows held out, as in K-fold CV),
#   b: finishers and partial rows.
# `compare` reports
#   1. the residuals of the partial rows under fit a: if retirees ran slower (or
#      faster) than their driver and car effects predict, these are shifted
#      relative to finishers' residuals, after removing the calibration's own
#      bias (from `calib`): direct evidence of selection on race-day pace;
#   2. how much driver and car effects move from a to b, in posterior sds, and
#      whether the moves line up with incident-proneness and unreliability
#      (output/retire_drivers.csv, retire_cars.csv from scripts/retire_fit.jl).

using Diomedes, CSV, DataFrames, Random, Serialization, Statistics
using Diomedes: _row_means, season_counts, LossCovariates, draw, model_param_names, sum_to_zero,
                sum_to_zero_by, AgeCurveBasis

const MODEL = "pl_dur_rw_kappa_age_big_slope_dev"
const SEASONS = 1996:2019
action = get(ARGS, 1, "compare")
data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
mkpath("output")

res = fetch_results(ErgastCSV(data_dir), SEASONS)
cum = ergast_cumulative_laps(data_dir)
pg = partial_gaps(res, cum)
g = prepare_gaps(with_partial_rows(res, pg))
partial = partial_mask(g)
train = .!partial
println(g, "; ", count(partial), " partial rows"); flush(stdout)

if action == "calib"
    # Impute each timed finisher's final gap from a random lap n in [L/4, L), with the
    # shift taken over the other finishers, as partial_gaps does for retirees.
    rng = Xoshiro(1)
    err = Float64[]
    for ev in groupby(res, :event_id)
        timed = findall(!ismissing, ev.time_ms)
        isempty(timed) && continue
        w = timed[argmin(ev.time_ms[timed])]
        L = ev.laps[w]
        Cw = get(cum, (ev.event_id[w], ev.competitor_id[w]), nothing)
        (Cw === nothing || length(Cw) < L) && continue
        C = Dict(j => cum[(ev.event_id[j], ev.competitor_id[j])] for j in timed
                 if j != w && length(get(cum, (ev.event_id[j], ev.competitor_id[j]), Float64[])) >= L)
        length(C) < 3 && continue
        for (j, Cj) in C
            n = rand(rng, max(2, ceil(Int, L / 4)):(L - 1))
            shift = median(100 * (log(Ck[L] / Cw[L]) - log(Ck[n] / Cw[n])) for (k, Ck) in C if k != j)
            push!(err, 100 * log(Cj[n] / Cw[n]) + shift - 100 * log(Cj[L] / Cw[L]))
        end
    end
    println("calibrated imputation error (imputed - actual final gap), $(length(err)) finishers:")
    println("  mean ", round(mean(err); digits = 3), " ± ", round(std(err) / sqrt(length(err)); digits = 3),
            ", median ", round(median(err); digits = 3), ", sd ", round(std(err); digits = 3))
    exit()
end

spec = model_spec(MODEL)
fit_kw = (; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale, age = spec.age,
          big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev, race_mean = spec.race_mean)
chain_prefix(v, k) = "output/selection_$(v)_chain$(k)"

if action == "chain"
    v, k = ARGS[2], parse(Int, ARGS[3])
    v in ("a", "b") || error("variant must be a or b")
    data = v == "a" ? subset_gaps(g, train) : g
    t = @elapsed chain = fit_paceloss(data; fit_kw..., rng = Xoshiro(k), checkpoint = chain_prefix(v, k),
                                      seed = 1000k + (v == "b"), progress_log = stdout)
    println("chain $v$k: $(round(t / 60; digits = 1)) min, mean leapfrog steps per draw ",
            round(mean(chain[:n_steps]); digits = 1), ", divergences ", count(chain[:numerical_error] .> 0))
    exit()
end

# chains to use per fit: 1..n, or comma-separated lists for a and b (e.g. `compare 1,2,3,4 1,2,4`)
chains_arg(x) = occursin(",", x) ? parse.(Int, split(x, ",")) : 1:parse(Int, x)
ks_a = chains_arg(get(ARGS, 2, "4"))
ks_b = chains_arg(get(ARGS, 3, get(ARGS, 2, "4")))
loadchain(v) = reduce(hcat, [run_nuts_loaded(chain_prefix(v, k)) for k in (v == "a" ? ks_a : ks_b)])
function run_nuts_loaded(prefix)
    blocks = sort(filter(f -> startswith(f, basename(prefix) * "_block"), readdir(dirname(prefix))),
                  by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
    return reduce(vcat, [rehash_chain!(deserialize(joinpath(dirname(prefix), f))) for f in blocks])
end

ca, cb = loadchain("a"), loadchain("b")
println("\nconvergence of identified quantities (max R-hat, min bulk ESS, worst group):")
for (v, ch, data) in (("a", ca, subset_gaps(g, train)), ("b", cb, g))
    ic = identified_convergence(ch, data, MODEL)
    worst = first(sort([k for k in keys(ic) if k !== :overall], by = k -> ic[k].min_ess))
    println("  fit $v: ", round(ic.overall.max_rhat; digits = 3), ", ", round(Int, ic.overall.min_ess), " (", worst,
            "); divergences ", count(ch[:numerical_error] .> 0))
end
names = model_param_names(spec)
sc, cov = season_counts(g), LossCovariates(g)
basis_a = AgeCurveBasis(subset_gaps(g, train))
nt = length(g.y)
yall = vcat(g.y, fill(NaN, length(g.lo)))
pidx = findall(partial)
fidx = findall(i -> i <= nt && train[i], eachindex(partial))     # timed finishers

# 1. residuals under fit a (row mean from the finishers-only fit)
ni, nc = size(ca[:σ])
rp, rf = zeros(ni * nc, length(pidx)), zeros(ni * nc, length(fidx))
let s = 0
    for c in 1:nc, i in 1:ni
        s += 1
        μt, _ = _row_means(draw(ca, names, i, c), g, sc, cov; train, age_basis = basis_a)
        rp[s, :] = yall[pidx] .- μt[pidx]
        rf[s, :] = yall[fidx] .- μt[fidx]
    end
end
# median residuals (the loss component makes the noise right-skewed), per draw
dmed = [median(rp[s, :]) - median(rf[s, :]) for s in axes(rp, 1)]
println("\n1. residuals under fit a (finishers only), in % points:")
println("  median residual: retirees' partial gaps ", round(median(vec(mean(rp; dims = 1))); digits = 3),
        ", finishers ", round(median(vec(mean(rf; dims = 1))); digits = 3))
println("  difference of medians (retirees - finishers): ", round(mean(dmed); digits = 3), " ± ",
        round(std(dmed); digits = 3), " (parameter uncertainty only)")
rpm = vec(mean(rp; dims = 1))
# row-sampling uncertainty of the retirees' median residual, by bootstrap over rows
rng = Xoshiro(2)
boot = [median(rpm[rand(rng, eachindex(rpm), length(rpm))]) for _ in 1:2000]
println("  retirees' median residual, bootstrap sd over rows: ", round(std(boot); digits = 3))
cause_of = Dict((r.event_id, r.competitor_id) => r.cause for r in eachrow(pg))
prow = g.rows[g.rows.kind .== :timed, :][pidx, :]
causes = [cause_of[(r.event_id, r.competitor_id)] for r in eachrow(prow)]
fmed = median(vec(mean(rf; dims = 1)))
for k in (:mech, :incident, :collision, :other)
    sel = causes .== k
    println("  ", rpad(k, 10), count(sel), " rows: median residual - finishers' ",
            round(median(rpm[sel]) - fmed; digits = 3))
end
frac_of = Dict((r.event_id, r.competitor_id) => r.n / r.L for r in eachrow(pg))
prow.n_frac = [frac_of[(r.event_id, r.competitor_id)] for r in eachrow(prow)]
for (lo, hi) in ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
    sel = lo .<= prow.n_frac .< hi
    println("  laps done $(lo)-$(hi) of distance: ", count(sel), " rows, median residual - finishers' ",
            round(median(rpm[sel]) - fmed; digits = 3))
end

# robustness: the same rows with the gap taken 3 laps before the retirement lap
# (a failing or damaged car can limp through its last laps; one slow lap moves a
# cumulative gap by up to ~1 point). Row means are unchanged, only y differs.
pg3 = partial_gaps(res, cum; trim = 3)
y3 = Dict((r.event_id, r.competitor_id) => r.y_hat for r in eachrow(pg3))
has3 = [haskey(y3, (r.event_id, r.competitor_id)) for r in eachrow(prow)]
μp = vec(mean(yall[pidx]' .- rp; dims = 1))          # posterior mean row means of the partial rows
r3 = [y3[(r.event_id, r.competitor_id)] for r in eachrow(prow)[has3]] .- μp[has3]
println("  gap 3 laps before the last lap ($(count(has3)) rows): median residual - finishers' ",
        round(median(r3) - fmed; digits = 3), " (same rows at the last lap: ",
        round(median(rpm[has3]) - fmed; digits = 3), ")")
for k in (:mech, :incident, :collision)
    sel = causes[has3] .== k
    println("    ", rpad(k, 10), round(median(r3[sel]) - fmed; digits = 3))
end

# 2. effect shifts from a to b
function effects(ch)
    ni, nc = size(ch[:σ])
    A = zeros(ni * nc, length(g.competitors)); B = zeros(ni * nc, length(g.machines))
    s = 0
    for c in 1:nc, i in 1:ni
        s += 1
        A[s, :] = ch[:σ_comp][i, c] .* sum_to_zero(ch[:z_comp][i, c])
        B[s, :] = sum_to_zero_by(ch[:b_mach][i, c], g.mach_season, sc)
    end
    return A, B
end
Aa, Ba = effects(ca); Ab, Bb = effects(cb)
nobs_c = [count(==(k), g.t_comp[train[1:nt]]) + count(==(k), g.c_comp) for k in eachindex(g.competitors)]
nobs_m = [count(==(k), g.t_mach[train[1:nt]]) + count(==(k), g.c_mach) for k in eachindex(g.machines)]
function shift_report(name, Xa, Xb, labels, nobs, minobs)
    keep = nobs .>= minobs
    ma, mb, sa = vec(mean(Xa; dims = 1))[keep], vec(mean(Xb; dims = 1))[keep], vec(std(Xa; dims = 1))[keep]
    z = (mb .- ma) ./ sa
    println("\n2. $name effects (≥ $minobs finishing rows, n = $(count(keep))): shift b - a")
    println("  correlation a vs b ", round(cor(ma, mb); digits = 4), "; |shift|/sd: mean ",
            round(mean(abs.(z)); digits = 2), ", max ", round(maximum(abs.(z)); digits = 2),
            "; mean shift ", round(mean(mb .- ma); digits = 3), "% (effects are centred)")
    o = sortperm(abs.(z); rev = true)[1:min(8, length(z))]
    show(stdout, MIME"text/plain"(), DataFrame(level = labels[keep][o], a = round.(ma[o]; digits = 2),
                                                 b = round.(mb[o]; digits = 2), shift_sd = round.(z[o]; digits = 2)))
    println()
    return DataFrame(level = labels[keep], a = ma, b = mb, sd_a = sa, shift_sd = z)
end
dshift = shift_report("driver", Aa, Ab, g.competitors, nobs_c, 30)
mshift = shift_report("car-season", Ba, Bb, g.machines, nobs_m, 20)
CSV.write("output/selection_driver_shifts.csv", dshift)
CSV.write("output/selection_car_shifts.csv", mshift)

# do the shifts line up with retirement-proneness?
if isfile("output/retire_drivers.csv")
    rdv = Dict(r.driver => r.effect for r in eachrow(CSV.read("output/retire_drivers.csv", DataFrame)))
    rcr = Dict(r.machine => r.effect for r in eachrow(CSV.read("output/retire_cars.csv", DataFrame)))
    kd = [haskey(rdv, l) for l in dshift.level]
    km = [haskey(rcr, l) for l in mshift.level]
    println("\ncorrelation of shift (b - a) with incident-proneness (drivers): ",
            round(cor(dshift.b[kd] .- dshift.a[kd], [rdv[l] for l in dshift.level[kd]]); digits = 3))
    println("correlation of shift (b - a) with unreliability (car-seasons): ",
            round(cor(mshift.b[km] .- mshift.a[km], [rcr[l] for l in mshift.level[km]]); digits = 3))
end
