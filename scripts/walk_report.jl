# Report of a race fit with chassis and engine walks (#95).
#
#   julia --project scripts/walk_report.jl [model] [baseline] [data=<mode>]
#
# Defaults: model `pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk` against the
# baseline `pl_dur_rw_kappa_age_big_slope_dev_circ_track`, both on the published
# fit's data. Prints the walk scales and step multipliers (and their difference:
# ownership against regulation changes), the
# residual spread against the baseline's car spread, total car-season effects
# against the baseline's, and example walks.

using Diomedes, DataFrames, Statistics, Printf

include("gap_data.jl")
mode, args = data_arg(ARGS)
model = get(args, 1, "pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk")
base = get(args, 2, "pl_dur_rw_kappa_age_big_slope_dev_circ_track")
_, g = gap_data(mode; io = devnull)
spec = model_spec(model)
cov, sc = LossCovariates(g, spec), Diomedes.season_counts(g)
cw = cov.cars
names = Diomedes.model_param_names(spec)
chain = gap_chains(mode, model)
ni, nc = size(chain[:σ_comp])
draws = [(i, c) for c in 1:nc for i in 1:5:ni]

q(v) = (mean(v), quantile(v, 0.05), quantile(v, 0.95))
show_q(lbl, v) = @printf("  %-22s %7.3f [%7.3f, %7.3f]\n", lbl, q(v)...)
println("$model on $mode: $(nc) chains × $(ni) draws\n")
println("Walk scales (% of race time, average-season units):")
for k in (:τ_ch, :σ0_ch, :τ_en, :σ0_en, :σ_mach); show_q(string(k), vec(chain[k])); end
println("\nStep multipliers exp(δ) at flagged steps:")
for k in (:δ_own, :δ_reg); show_q("exp($k)", exp.(vec(chain[k]))); end
println("\nDifference of the δs (one multiplier if it sits near 0):")
for (a, b) in ((:δ_own, :δ_reg),)
    d = vec(chain[a]) .- vec(chain[b])
    show_q("$a − $b", d); @printf("  %-22s P(> 0) = %.2f\n", "", mean(d .> 0))
end
bchain = gap_chains(mode, base)
println("\nResidual car spread σ_mach against the baseline's car spread:")
show_q("baseline σ_mach", vec(bchain[:σ_mach])); show_q("walk-model σ_mach", vec(chain[:σ_mach]))

# total car-season effect: residual + mean over the car's rows of κ·(chassis + engine),
# centred within season like the baseline's car effects
nt = length(g.t_comp)
mach = vcat(g.t_mach, g.c_mach)
race = vcat(g.t_race, g.c_race)
nm = length(g.machines)
nrows = [count(==(m), mach) for m in 1:nm]
tot = zeros(length(draws), nm)
for (d, (i, c)) in enumerate(draws)
    θ = Diomedes.draw(chain, names, i, c)
    κ = race_pace_scale(θ, cov)
    xc, xe = car_walk_states(θ, cw)
    hw = Diomedes.walk_offsets(xc, cw.row_ch, xe, cw.row_en) .* κ[race]
    w = zeros(nm); for (r, m) in enumerate(mach); w[m] += hw[r]; end
    tot[d, :] = Diomedes.sum_to_zero_by(θ.b_mach .+ w ./ nrows, g.mach_season, sc)
end
bn = Diomedes.model_param_names(model_spec(base))
bd = [(i, c) for c in 1:size(bchain[:σ_comp], 2) for i in 1:5:size(bchain[:σ_comp], 1)]
btot = reduce(vcat, [Diomedes.sum_to_zero_by(Diomedes.draw(bchain, bn, i, c).b_mach, g.mach_season, sc)'
                     for (i, c) in bd])
mn, bm = vec(mean(tot; dims = 1)), vec(mean(btot; dims = 1))
sn, bs = vec(std(tot; dims = 1)), vec(std(btot; dims = 1))
println("\nTotal car-season effects against the baseline's ($(nm) car-seasons):")
@printf("  Pearson %.3f, sd of shifts %.3f%%\n", cor(mn, bm), std(mn .- bm))
for (lo, hi) in ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10_000))
    s = (nrows .>= lo) .& (nrows .<= hi)
    any(s) && @printf("  car-seasons with %d-%d rows (%3d): posterior sd %.3f against %.3f (ratio %.2f); mean |shift| %.3f\n",
                      lo, min(hi, maximum(nrows)), count(s), mean(sn[s]), mean(bs[s]), mean(sn[s] ./ bs[s]), mean(abs.(mn[s] .- bm[s])))
end
println("  largest shifts:")
for m in first(sortperm(abs.(mn .- bm); rev = true), 12)
    @printf("    %-26s %6.2f → %6.2f  (sd %.2f → %.2f, %d rows)\n", g.machines[m], bm[m], mn[m], bs[m], sn[m], nrows[m])
end

# example walks: posterior mean state by season
function walk_means(ws, which)
    S = zeros(length(draws), length(ws.prev))
    for (d, (i, c)) in enumerate(draws)
        S[d, :] = car_walk_states(Diomedes.draw(chain, names, i, c), cw)[which]
    end
    return vec(mean(S; dims = 1))
end
mc, me = walk_means(cw.chassis, 1), walk_means(cw.engine, 2)
for (lbl, ws, mv, keys) in (("Chassis", cw.chassis, mc, ("enstone", "brackley", "milton-keynes", "ferrari", "lotus", "mclaren")),
                            ("Engine", cw.engine, me, ("renault", "honda", "cosworth", "ferrari", "mercedes", "climax")))
    println("\n$lbl walks (posterior mean, % in average-season units, centred within season):")
    for k in keys
        idx = findall(n -> startswith(n, k * " "), ws.names)
        println("  $k: ", join(["$(split(ws.names[j])[2]) $(round(mv[j]; digits = 2))" for j in idx], ", "))
    end
end
