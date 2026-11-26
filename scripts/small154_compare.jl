# #154 small tests: baseline, durc and dense on jolpica1950-1990, read-only on the probe blocks.
#
#   julia --project scripts/small154_compare.jl
#
# Per design: sampler cost (leapfrog steps per draw), per-group max R-hat / min ESS
# (identified_convergence), the scalars' slowest held-out direction (slow_directions),
# the σ_mach/τ_ch split, and the coupling of β_dur with the era walks. Against the
# baseline: correlation of driver and car-season effects, and every shared scalar's
# difference in combined Monte Carlo SEs.
using Diomedes, Statistics, Printf, Serialization
import MCMCDiagnosticTools as MDT
include("gap_data.jl")
const MODE = "jolpica1950-1990"
const DESIGNS = [("base", "pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc", ""),
                 ("durc", "pl_durc_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc", ""),
                 ("dense", "pl_dur_rw_kappa_age_big_slope_dev_circ_track_walk1_works_dinc", "_dense")]
res, g = gap_data(MODE; io = devnull)

function load(model, sfx)
    tag = "gap_$(MODE)_$(model)$(sfx)_probe"
    chs = []
    for k in 1:4
        fs = sort(filter(f -> occursin(Regex("^\\Q$(tag)_chain$(k)_block\\E\\d+\\.jls\$"), f), readdir("output")),
                  by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
        isempty(fs) && continue
        push!(chs, reduce(vcat, [Diomedes.block_chain(joinpath("output", f)) for f in fs]))
    end
    n = minimum(size(c[:σ], 1) for c in chs)
    return reduce(hcat, [c[iter = 1:n] for c in chs])
end
arr(M) = reshape(M, size(M, 1), size(M, 2), 1)
st(M) = (MDT.rhat(arr(M))[1], MDT.ess(arr(M))[1])

# β_dur against the walk's slope on season-mean duration (as on #154, step 1)
function dur_coupling(chain, cov, part)
    s, ld = cov.season, cov.log_duration
    raw = LossCovariates(g).log_duration       # the uncentred covariate, whose season means the walk can carry
    ns = cov.n_seasons
    nr = [count(==(k), s) for k in 1:ns]; ok = nr .> 0; w = nr ./ sum(nr)
    dbar = [ok[k] ? mean(raw[s .== k]) : 0.0 for k in 1:ns]
    dc = dbar .- sum(w[ok] .* dbar[ok])
    β = Array(chain[Symbol("β_dur_$part")]); τ = Array(chain[Symbol("τ_$(part)_rw")]); E = chain[Symbol("e_$(part)_rw")]
    wd = similar(β)
    for i in eachindex(β)
        p = vcat(0.0, cumsum(τ[i] .* E[i])); p .-= sum(w[ok] .* p[ok])
        wd[i] = sum(w[ok] .* p[ok] .* dc[ok]) / sum(w[ok] .* dc[ok] .^ 2)
    end
    return β, wd
end

out = Dict{String,Any}()
for (name, model, sfx) in DESIGNS
    chain = try load(model, sfx) catch e; println("$name: not available ($e)"); continue end
    cov = LossCovariates(g, model_spec(model))
    println("\n================ $name ($(model)$(sfx)): $(size(chain[:σ], 1)) draws × $(size(chain[:σ], 2)) chains")
    steps = Array(chain[:n_steps]); div = count(>(0), Array(chain[:numerical_error]))
    @printf("leapfrog steps per draw %.0f (per chain %s); divergences %d\n", mean(steps),
            join([@sprintf("%.0f", mean(steps[:, c])) for c in axes(steps, 2)], " "), div)
    conv = identified_convergence(chain, g, model; cov)
    for (k, v) in sort(collect(pairs(conv)); by = first)
        @printf("  %-10s R-hat %.3f  ESS %.0f  %s\n", k, v.max_rhat, v.min_ess, get(v, :worst, ""))
    end
    cols, label = identified_draws(chain, g, model; cov)
    sd = slow_directions(cols[:scalars], [label(:scalars, i) for i in axes(cols[:scalars], 3)]; k = 2, top = 6)
    for (j, d) in enumerate(sd)
        @printf("scalars slow direction %d: %.0f %% of spread, R-hat %.3f, ESS %.0f, ESS per 1k steps/chain %.2f\n   %s\n",
                j, 100d.share, d.rhat, d.ess, 1000d.ess / sum(steps) * size(steps, 2),
                join([@sprintf("%s %+.2f", l, w) for (l, w) in d.loadings], ", "))
    end
    split = log.(Array(chain[:σ_mach])) .- log.(Array(chain[:τ_ch]))
    @printf("split log σ_mach − log τ_ch: R-hat %.3f, ESS %.0f\n", st(split)...)
    for part in ("π", "λ")
        β, wd = dur_coupling(chain, cov, part)
        @printf("β_dur_%s %.3f ± %.3f (R-hat %.3f ESS %.0f); corr with walk's duration slope %.2f\n",
                part, mean(β), std(β), st(β)..., cor(vec(β), vec(wd)))
    end
    out[name] = (; chain, cov, model, cols, label)
end

# against the baseline
if haskey(out, "base")
    b = out["base"]
    for name in ("durc", "dense")
        haskey(out, name) || continue
        o = out[name]
        println("\n== $name against base")
        for gr in (:drivers, :cars)
            haskey(o.cols, gr) && haskey(b.cols, gr) || continue
            mo = vec(mean(o.cols[gr]; dims = (1, 2))); mb = vec(mean(b.cols[gr]; dims = (1, 2)))
            @printf("  %s: correlation %.4f, largest difference %.3f\n", gr, cor(mo, mb), maximum(abs.(mo .- mb)))
        end
        A, B = o.cols[:scalars], b.cols[:scalars]
        la = [o.label(:scalars, i) for i in axes(A, 3)]
        z = [abs(mean(A[:, :, i]) - mean(B[:, :, i])) / sqrt(std(A[:, :, i])^2 / MDT.ess(A[:, :, i:i])[1] +
                                                         std(B[:, :, i])^2 / MDT.ess(B[:, :, i:i])[1]) for i in axes(A, 3)]
        ord = sortperm(z; rev = true)[1:5]
        println("  scalars, largest |difference| / combined MCSE: ",
                join([@sprintf("%s %.1f", la[i], z[i]) for i in ord], ", "))
    end
end
