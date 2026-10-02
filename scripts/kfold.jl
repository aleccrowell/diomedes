# Grouped K-fold cross-validation of the gap models (#18).
#
#   julia --project scripts/kfold.jl fold <model> <K> <k> [n_samples=1000]
#       fit <model> without fold k (one chain) and save the held-out rows'
#       log-likelihood draws to output/kfold_<model>_K<K>_fold<k>.jls
#   julia --project scripts/kfold.jl compare <K> <model>...
#       elpd from all K folds of each model, and paired differences
#
# Folds are the same for every model (fixed seed per K) and stratified within
# races (see `kfold_folds`). Folds are independent processes, so run several
# in parallel (one per core).

using Diomedes, Random, Serialization, Statistics

data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data"))
g = prepare_gaps(fetch_results(ErgastCSV(data_dir), 1950:2100))
path(model, K, k) = "output/kfold_$(model)_K$(K)_fold$(k).jls"

if ARGS[1] == "fold"
    model, K, k = ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4])
    n_samples = parse(Int, get(ARGS, 5, "1000"))
    folds = kfold_folds(g, K; rng = Xoshiro(2026 + K))
    train = folds .!= k
    gt = subset_gaps(g, train)
    spec = model_spec(model)
    println("$model, fold $k of $K: training on $(count(train)) rows, holding out $(count(!, train))")
    t = @elapsed chain = spec.family === :t4 ?
        fit_gaps(gt; n_samples, rng = Xoshiro(k), progress = false, progress_log = stdout, log_every = 100) :
        fit_paceloss(gt; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                     age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, n_samples,
                     rng = Xoshiro(k), progress = false, progress_log = stdout, log_every = 100)
    ll = heldout_loglik(chain, g, train, model)
    mkpath("output")
    serialize(path(model, K, k), (; test = findall(!, train), ll))
    println("fold $k: $(round(t / 60; digits = 1)) min; held-out elpd $(round(sum(elpd_rows(ll)); digits = 1))")
    println(convergence_summary(chain))
elseif ARGS[1] == "compare"
    K = parse(Int, ARGS[2])
    n = length(g.y) + length(g.lo)
    elpd = Dict{String,Vector{Float64}}()
    for model in ARGS[3:end]
        e = fill(NaN, n)
        for k in 1:K
            isfile(path(model, K, k)) || (println("$model: fold $k missing"); continue)
            f = deserialize(path(model, K, k))
            e[f.test] = elpd_rows(f.ll)
        end
        done = .!isnan.(e)
        println("$model: elpd_kfold = $(round(sum(e[done]); digits = 1)) ± ",
                "$(round(sqrt(count(done) * var(e[done])); digits = 1)) over $(count(done)) of $n rows")
        elpd[model] = e
    end
    models = ARGS[3:end]
    best = models[argmax([sum(filter(!isnan, elpd[m])) for m in models])]
    for m in models
        m == best && continue
        d = elpd[best] .- elpd[m]
        ok = .!isnan.(d)
        println("$best − $m: Δelpd = $(round(sum(d[ok]); digits = 1)) ± $(round(sqrt(count(ok) * var(d[ok])); digits = 1))",
                count(ok) < n ? " (over $(count(ok)) rows with both models)" : "")
    end
else
    error("usage: kfold.jl fold <model> <K> <k> [n_samples] | compare <K> <model>...")
end
