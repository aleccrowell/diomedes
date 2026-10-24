# Report of a race fit with chassis and engine walks (#95, #99–#102).
#
#   julia --project scripts/walk_report.jl [model] [baseline] [data=<mode>] [base_data=<mode>]
#
# Defaults: model `pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk` against the
# baseline `pl_dur_rw_kappa_age_big_slope_dev_circ_track`, both on the published
# fit's data; `base_data` (default: `data`) when the baseline was fitted to other
# data. Prints the walk scales and step multipliers (and their differences), the
# works-team coefficient and the spread of per-driver incident proneness when the
# model has them, the residual spread against the baseline's, total car-season
# effects against the baseline's (joined by car-season), and example walks.

using Diomedes, DataFrames, Statistics, Printf

include("gap_data.jl")
mode, args = data_arg(ARGS)
bi = findfirst(a -> startswith(a, "base_data="), args)
base_mode = bi === nothing ? mode : String(split(args[bi], "="; limit = 2)[2])
pos = filter(a -> !startswith(a, "base_data="), args)
model = get(pos, 1, "pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk")
base = get(pos, 2, "pl_dur_rw_kappa_age_big_slope_dev_circ_track")

q(v) = (mean(v), quantile(v, 0.05), quantile(v, 0.95))
show_q(lbl, v) = @printf("  %-26s %7.3f [%7.3f, %7.3f]\n", lbl, q(v)...)
has(chain, k) = any(vn -> string(vn) == string(k), Diomedes.Turing.FlexiChains.parameters(chain))

# total car-season effects per draw (draws × car-seasons): residual + mean over the
# car's rows of κ·(chassis + engine [+ works term]), centred within season
function car_totals(chain, mdl, g; thin = 5)
    spec = model_spec(mdl)
    cov, sc = LossCovariates(g, spec), Diomedes.season_counts(g)
    names = Diomedes.model_param_names(spec)
    ni, nc = size(chain[:σ_comp])
    mach, race = vcat(g.t_mach, g.c_mach), vcat(g.t_race, g.c_race)
    nm = length(g.machines)
    nrows = [count(==(m), mach) for m in 1:nm]
    draws = [(i, c) for c in 1:nc for i in 1:thin:ni]
    T = zeros(length(draws), nm)
    for (d, (i, c)) in enumerate(draws)
        θ = Diomedes.draw(chain, names, i, c)
        b = Diomedes.car_residuals(θ, cov)
        if spec.walk
            κ = race_pace_scale(θ, cov)
            xc, xe = car_walk_states(θ, cov.cars)
            hw = Diomedes.walk_offsets(xc, cov.cars.row_ch, xe, cov.cars.row_en)
            haskey(θ, :β_works) && (hw = hw .+ θ.β_works .* cov.cars.row_works)
            w = zeros(nm); for (r, m) in enumerate(mach); w[m] += κ[race[r]] * hw[r]; end
            b = b .+ w ./ nrows
        end
        T[d, :] = Diomedes.sum_to_zero_by(b, g.mach_season, sc)
    end
    return T, nrows
end

_, g = gap_data(mode; io = devnull)
spec = model_spec(model)
cov = LossCovariates(g, spec)
cw = cov.cars
names = Diomedes.model_param_names(spec)
chain = gap_chains(mode, model)
ni, nc = size(chain[:σ_comp])
println("$model on $mode: $(nc) chains × $(ni) draws\n")
println("Walk scales (% of race time, average-season units):")
for k in (:τ_ch, :σ0_ch, :τ_en, :σ0_en, :σ_mach); has(chain, k) && show_q(string(k), vec(chain[k])); end
δs = [k for k in (:δ_own, :δ_reg, :δ_reg_ch, :δ_reg_en) if has(chain, k)]
println("\nStep multipliers exp(δ) at flagged steps:")
for k in δs; show_q("exp($k)", exp.(vec(chain[k]))); end
println("\nDifferences of the δs:")
for i in eachindex(δs), j in (i + 1):length(δs)
    d = vec(chain[δs[i]]) .- vec(chain[δs[j]])
    show_q("$(δs[i]) − $(δs[j])", d); @printf("  %-26s P(> 0) = %.2f\n", "", mean(d .> 0))
end
if has(chain, :β_works)
    println("\nWorks team (the engine builder owns the chassis works; % of race time, average-season units, negative = faster):")
    show_q("β_works", vec(chain[:β_works]))
    @printf("  works rows: %d of %d\n", Int(sum(cw.row_works)), length(cw.row_works))
end
if has(chain, :τ_dinc)
    println("\nPer-driver incident proneness (logit scale):")
    show_q("τ_dinc", vec(chain[:τ_dinc]))
    O = reduce(vcat, [Diomedes.driver_incident_offsets(Diomedes.draw(chain, names, i, c))' for c in 1:nc for i in 1:5:ni])
    om = vec(mean(O; dims = 1))
    comps = vcat(g.t_comp, g.c_comp); nr = [count(==(d), comps) for d in eachindex(g.competitors)]
    big = findall(>=(40), nr)
    println("  most incident-prone (≥ 40 results): ", join(["$(g.competitors[d]) $(round(om[d]; digits = 2))" for d in big[sortperm(om[big]; rev = true)][1:8]], ", "))
    println("  least incident-prone: ", join(["$(g.competitors[d]) $(round(om[d]; digits = 2))" for d in big[sortperm(om[big])][1:8]], ", "))
end

_, gb = base_mode == mode ? (nothing, g) : gap_data(base_mode; io = devnull)
bchain = gap_chains(base_mode, base)
println("\nResidual car spread σ_mach against the baseline's ($base on $base_mode):")
show_q("baseline σ_mach", vec(bchain[:σ_mach])); show_q("model σ_mach", vec(chain[:σ_mach]))

T, nrows = car_totals(chain, model, g)
Tb, _ = car_totals(bchain, base, gb)
mt = DataFrame(car = g.machines, mean = vec(mean(T; dims = 1)), sd = vec(std(T; dims = 1)), rows = nrows)
bt = DataFrame(car = gb.machines, bmean = vec(mean(Tb; dims = 1)), bsd = vec(std(Tb; dims = 1)))
j = innerjoin(mt, bt; on = :car)
println("\nTotal car-season effects against the baseline's ($(nrow(j)) shared car-seasons):")
@printf("  Pearson %.3f, sd of shifts %.3f%%\n", cor(j.mean, j.bmean), std(j.mean .- j.bmean))
for (lo, hi) in ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10_000))
    s = (j.rows .>= lo) .& (j.rows .<= hi)
    any(s) && @printf("  car-seasons with %d-%d rows (%3d): posterior sd %.3f against %.3f (ratio %.2f); mean |shift| %.3f\n",
                      lo, min(hi, maximum(j.rows)), count(s), mean(j.sd[s]), mean(j.bsd[s]), mean(j.sd[s] ./ j.bsd[s]),
                      mean(abs.(j.mean[s] .- j.bmean[s])))
end
println("  largest shifts:")
for r in eachrow(first(sort(transform(j, [:mean, :bmean] => ByRow((x, y) -> abs(x - y)) => :shift), :shift; rev = true), 12))
    @printf("    %-26s %6.2f → %6.2f  (sd %.2f → %.2f, %d rows)\n", r.car, r.bmean, r.mean, r.bsd, r.sd, r.rows)
end

# example walks: posterior mean state by season
function walk_means(which)
    S = reduce(vcat, [car_walk_states(Diomedes.draw(chain, names, i, c), cw)[which]' for c in 1:nc for i in 1:5:ni])
    return vec(mean(S; dims = 1))
end
mc, me = walk_means(1), walk_means(2)
for (lbl, ws, mv, keys) in (("Chassis", cw.chassis, mc, ("enstone", "brackley", "milton-keynes", "ferrari", "mclaren", "hinwil", "cadillac")),
                            ("Engine", cw.engine, me, ("renault", "honda", "ferrari", "mercedes", "red-bull-powertrains", "audi")))
    println("\n$lbl walks (posterior mean, % in average-season units, centred within season):")
    for k in keys
        idx = findall(n -> startswith(n, k * " "), ws.names)
        isempty(idx) && continue
        println("  $k: ", join(["$(split(ws.names[i])[2]) $(round(mv[i]; digits = 2))" for i in idx[max(1, end - 14):end]], ", "))
    end
end
