# Compare the gap model fitted to lead-lap finishers only vs with lapped cars (#9).
#
#   julia --project scripts/gap_fit.jl jolpica1950-2025+timed t4
#   julia --project scripts/gap_fit.jl jolpica1950-2025 t4
#   julia --project scripts/compare_gap_fits.jl [jolpica1950-2025]

using CSV, DataFrames, Statistics

data = get(ARGS, 1, "jolpica1950-2025")
rd(mode, kind) = CSV.read("output/gap_$(mode)_t4_$(kind)_effects.csv", DataFrame)
spearman(x, y) = cor(invperm(sortperm(x)), invperm(sortperm(y)))

for kind in ("competitor", "machine")
    t, a = rd(data * "+timed", kind), rd(data, kind)
    j = innerjoin(t, a; on = :label, makeunique = true)
    println("== $(kind) effects: $(nrow(j)) in both fits, $(nrow(a) - nrow(j)) only with lapped cars")
    println("   Pearson $(round(cor(j.mean, j.mean_1); digits = 3)), Spearman $(round(spearman(j.mean, j.mean_1); digits = 3))")
    # n_obs in the fit with lapped cars counts timed + lapped rows; the difference is the lapped share
    j.lapped_share = 1 .- j.n_obs ./ j.n_obs_1
    j.shift = j.mean_1 .- j.mean
    cols = kind == "competitor" ? [:name, :mean, :mean_1, :shift, :n_obs, :n_obs_1, :lapped_share] :
                                  [:label, :mean, :mean_1, :shift, :n_obs, :n_obs_1, :lapped_share]
    println("   biggest shifts (mean = finishers-only, mean_1 = with lapped cars; % of race time):")
    show(first(sort(j, :shift; by = abs, rev = true)[:, cols], 8); allcols = true, eltypes = false)
    println("\n   correlation of |shift| with lapped share: ",
            round(cor(abs.(j.shift), j.lapped_share); digits = 3), "\n")
end
