# Per-row log-likelihoods for model comparison with PSIS-LOO (#9).
#
# The gap models add their likelihood in one `@addlogprob!` call, so Turing
# cannot produce pointwise log-likelihoods itself. These functions recompute
# them per row (timed rows first, then lapped rows, as in GapData) from the
# posterior draws. They reuse the same per-row functions as the likelihoods,
# and the tests check that the rows sum to the fused likelihood.

# Centred effects and per-row means for one parameter draw (with the pace scale
# κ if θ has one, see `race_pace_scale`).
#
# For held-out rows (K-fold CV, #18), `train` marks the rows (timed, then lapped)
# the model was fitted to: race centring and each driver's mean age use only
# those rows, and the age curve's projection comes from `age_basis` (built from
# the training data), so a held-out row's mean is what the fitted model predicts.
function _row_means(θ, g::GapData, sc, cov = LossCovariates(g); train = nothing, age_basis = nothing)
    a = haskey(θ, :a_comp) ? sum_to_zero(θ.a_comp) : θ.σ_comp .* sum_to_zero(θ.z_comp)   # centred drivers (#28)
    centred = haskey(θ, :b_mach) || haskey(θ, :z_res)   # pace-scale model (#15): cars in % units, κ on drivers
    b = centred ? sum_to_zero_by(car_residuals(θ, cov), g.mach_season, sc) : θ.σ_mach .* sum_to_zero_by(θ.z_mach, g.mach_season, sc)
    κ = haskey(θ, :τ_κ) ? race_pace_scale(θ, cov) : ones(length(θ.γ))
    kb = centred ? ones(length(θ.γ)) : κ
    at, ac = a[g.t_comp], a[g.c_comp]
    if haskey(θ, :τ_age) || haskey(θ, :τ_slope)     # career terms (#15 stage 2)
        fage = haskey(θ, :τ_age) ? age_curve(θ.τ_age, θ.e_age, something(age_basis, AgeCurveBasis(g))) : nothing
        slope = haskey(θ, :τ_slope) ? career_slopes(θ.τ_slope, θ.u_slope) : nothing
        h = driver_offsets(CareerRows(g; train), fage, slope)
        nt = length(g.t_comp)
        at = at .+ h[1:nt]
        ac = ac .+ h[(nt + 1):end]
    end
    ct = κ[g.t_race] .* at .+ kb[g.t_race] .* b[g.t_mach]
    cc = κ[g.c_race] .* ac .+ kb[g.c_race] .* b[g.c_mach]
    if haskey(θ, :τ_dev)                  # in-season car development (#14), not scaled by κ
        dr = DevRows(g; train)
        hcar = dev_trends(θ.τ_dev, θ.u_dev, g, sc)[dr.mach] .* dr.posc
        nt = length(g.t_comp)
        ct = ct .+ hcar[1:nt]
        cc = cc .+ hcar[(nt + 1):end]
    end
    if haskey(θ, :τ_cx)                   # car-season circuit slopes (#81), not scaled by κ
        hcx = circuit_offsets(circuit_slopes(θ.τ_cx, θ.u_cx, g, sc), CircuitRows(g, cov.circuits; train))
        nt = length(g.t_comp)
        ct = ct .+ hcx[1:nt]
        cc = cc .+ hcx[(nt + 1):end]
    end
    if haskey(θ, :τ_ch)                   # chassis and engine walks (#95), scaled by κ like the drivers
        xc, xe = car_walk_states(θ, cov.cars)
        hw = walk_offsets(xc, cov.cars.row_ch, xe, cov.cars.row_en)
        haskey(θ, :β_works) && (hw = hw .+ θ.β_works .* cov.cars.row_works)     # works term (#100)
        nt = length(g.t_comp)
        ct = ct .+ κ[g.t_race] .* hw[1:nt]
        cc = cc .+ κ[g.c_race] .* hw[(nt + 1):end]
    end
    csum, n = zeros(length(θ.γ)), zeros(Int, length(θ.γ))
    nt = length(ct)
    use(i) = train === nothing || train[i]
    for (i, (c, r)) in enumerate(zip(ct, g.t_race)); use(i) && (csum[r] += c; n[r] += 1); end
    for (i, (c, r)) in enumerate(zip(cc, g.c_race)); use(nt + i) && (csum[r] += c; n[r] += 1); end
    cbar = csum ./ max.(n, 1)
    return θ.γ[g.t_race] .+ ct .- cbar[g.t_race], θ.γ[g.c_race] .+ cc .- cbar[g.c_race]
end

"Per-row log-likelihoods of `gap_effects` for parameter values `θ` (a NamedTuple)."
function gap_rows(θ, g::GapData; noise::Noise = StudentTNoise(4), sc = season_counts(g), train = nothing)
    μt, μc = _row_means(θ, g, sc; train)
    σ = θ.σ_y
    timed = [logpdf_std(noise, (g.y[i] - μt[i]) / σ) - log(σ) for i in eachindex(μt)]
    lapped = [logdiffcdf_std(noise, (g.lo[i] - μc[i]) / σ, (g.hi[i] - μc[i]) / σ) for i in eachindex(μc)]
    return vcat(timed, lapped)
end

"Per-driver logit offsets on the incident probability (#101), centred across drivers, or nothing."
driver_incident_offsets(θ) = haskey(θ, :τ_dinc) ? θ.τ_dinc .* (θ.u_dinc .- mean(θ.u_dinc)) : nothing

"Per-row log-likelihoods of `paceloss_effects` for parameter values `θ` (a NamedTuple)."
function paceloss_rows(θ, g::GapData; loss_duration::Bool = false, era::Symbol = :none,
                       sc = season_counts(g), cov = LossCovariates(g),   # pace scale if θ has τ_κ
                       train = nothing, age_basis = nothing)             # held-out rows: see _row_means
    μt, μc = _row_means(θ, g, sc, cov; train, age_basis)
    πr, λr = race_loss(θ, cov, loss_duration, era)
    if haskey(θ, :a_ρ)                    # two-component incident loss (#16)
        ρ, λ2 = logistic(θ.a_ρ), exp(θ.a_λ + θ.δ_λ2)
        πd = driver_incident_offsets(θ)    # per-driver incident proneness (#101), or nothing
        qt(i) = row_π(πr, g.t_race[i], πd, g.t_comp[i])
        qc(i) = row_π(πr, g.c_race[i], πd, g.c_comp[i])
        timed = [pacebig_logpdf(g.y[i] - μt[i], θ.σ, qt(i), λr[g.t_race[i]], ρ, λ2) for i in eachindex(μt)]
        lapped = [pacebig_loginterval(g.lo[i] - μc[i], g.hi[i] - μc[i], θ.σ, qc(i), λr[g.c_race[i]], ρ, λ2)
                  for i in eachindex(μc)]
        return vcat(timed, lapped)
    end
    timed = [paceloss_logpdf(g.y[i] - μt[i], θ.σ, πr[g.t_race[i]], λr[g.t_race[i]]) for i in eachindex(μt)]
    lapped = [paceloss_loginterval(g.lo[i] - μc[i], g.hi[i] - μc[i], θ.σ, πr[g.c_race[i]], λr[g.c_race[i]])
              for i in eachindex(μc)]
    return vcat(timed, lapped)
end

"""
    rehash_chain!(chain) -> chain

Rebuild the key index of a chain read back with `deserialize`. A FlexiChain
stores its parameters in a hash table keyed by VarNames, and the hashes saved
with it can differ from those of the reading process (e.g. after packages are
recompiled). Lookups then fail (`KeyError`), and `hcat` of such chains silently
yields `missing` for every key it cannot match. Call this on each chain before
using it: `chain = reduce(hcat, rehash_chain!.(deserialize.(paths)))`.
"""
function rehash_chain!(chain)
    d = getfield(chain, :_data)
    parentmodule(typeof(d)).rehash!(d)
    return chain
end

# The parameter values of draw (i, c) of a chain, as a NamedTuple.
draw(chain, names, i, c) = NamedTuple{names}(Tuple(chain[n][i, c] for n in names))

"""
    model_spec(name) -> (; family, loss_duration, era)

Parse a model name used by the scripts: `t4` (Student-t(4) `gap_effects`) or
`pl[_dur][_<era>][_kappa][_age][_big][_slope]` (`paceloss_effects`), e.g. `pl`,
`pl_dur`, `pl_regime`, `pl_dur_rw_kappa_age`, with `<era>` one of `decade`,
`regime`, `rw`, `_kappa` adding the pace scale, `_age` the career curve and
`_slope` per-driver career slopes (#15), `_big` the big-loss component (#16),
`_dev` in-season development (#14), `_circ` car-season circuit slopes and
`_track` tight/mixed/wet on incidents (#81), `_cdrv`/`_tdrv` centred/Student-t drivers
(#28), and `_hgam` hierarchical race intercepts or `_fgam` the fixed race-intercept
prior (#32). By default the common race level is learned (`race_mean`, #36);
`_mgam` names that default explicitly and is accepted for older names.
"""
function model_spec(name::AbstractString)
    name == "t4" && return (; family = :t4, loss_duration = false, era = :none, pace_scale = false, age = false,
                            big_loss = false, slopes = false, dev = false, centred_drivers = false,
                            driver_ν = nothing, race_hier = false, race_mean = false, circ = false, track = false, walk = false,
                            walk1 = false, works = false, dinc = false, ncres = false, wlev = false,
                            kc = false, kst = false, wst = false, wstc = false, wstl = false, wlevn = false, durc = false)
    parts = split(name, "_")
    first(parts) == "pl" || throw(ArgumentError("unknown model $name"))
    durc = "durc" in parts          # duration centred within season (#154); implies dur
    dur, kappa, age, big, slopes, dev = "dur" in parts || durc, "kappa" in parts, "age" in parts, "big" in parts,
                                        "slope" in parts, "dev" in parts
    circ, track = "circ" in parts, "track" in parts      # circuit terms (#81)
    walk1 = "walk1" in parts        # walks with one regulation multiplier (#97)
    walk = "walk" in parts || walk1 # chassis and engine walks (#95)
    works, dinc = "works" in parts, "dinc" in parts     # works term (#100), driver incidents (#101)
    ncres = "ncres" in parts        # non-centred car effects (#102)
    wlev = "wlev" in parts          # walk levels as coordinates (#102)
    wlevn = "wlevn" in parts        # walk levels, the engine's non-centred (#102)
    kc = "kc" in parts              # centred pace-scale steps (#102)
    kst = "kst" in parts            # pace-scale log path sampled directly (#102)
    wst = "wst" in parts            # walk states sampled directly (#102)
    wstc = "wstc" in parts          # chassis states sampled directly, engine steps (#102)
    wstl = "wstl" in parts          # chassis states sampled directly, engine levels (#102)
    cdrv = "cdrv" in parts
    tdrv = "tdrv" in parts          # Student-t(3) driver prior (#28)
    hgam = "hgam" in parts          # hierarchical race intercepts (#32)
    fgam = "fgam" in parts          # fixed race-intercept prior; the learned common level is the default (#36)
    count(in(parts), ("hgam", "mgam", "fgam")) <= 1 || throw(ArgumentError("_hgam, _mgam and _fgam are alternatives: $name"))
    eras = [Symbol(p) for p in parts[2:end]
            if p ∉ ("dur", "durc", "kappa", "age", "big", "slope", "dev", "circ", "track", "walk", "walk1", "works", "dinc", "ncres", "wlev", "wlevn", "kc", "kst", "wst", "wstc", "wstl", "cdrv", "tdrv", "hgam", "mgam", "fgam")]
    length(eras) <= 1 && all(in(ERA_TERMS), eras) || throw(ArgumentError("unknown model $name"))
    return (; family = :pl, loss_duration = dur, era = isempty(eras) ? :none : only(eras),
            pace_scale = kappa, age, big_loss = big, slopes, dev, centred_drivers = cdrv,
            driver_ν = tdrv ? 3.0 : nothing, race_hier = hgam, race_mean = !hgam && !fgam, circ, track, walk, walk1, works, dinc, ncres, wlev, kc, kst, wst, wstc, wstl, wlevn, durc)
end

"""
    pointwise_loglik(chain, g::GapData, model) -> Array{Float64,3}

Log-likelihood of every row for every draw, shaped (draws, chains, rows) as
PosteriorStats' `loo` expects. `model` is a model name (see `model_spec`).
"""
function pointwise_loglik(chain, g::GapData, model::AbstractString)
    spec = model_spec(model)
    sc, cov = season_counts(g), LossCovariates(g, spec)
    names = model_param_names(spec)
    rows(θ) = spec.family === :t4 ? gap_rows(θ, g; sc) :
        paceloss_rows(θ, g; loss_duration = spec.loss_duration, era = spec.era, sc, cov)
    ni, nc = size(chain[:σ_comp])
    out = Array{Float64,3}(undef, ni, nc, length(g.y) + length(g.lo))
    for c in 1:nc, i in 1:ni
        out[i, c, :] = rows(draw(chain, names, i, c))
    end
    return out
end

"Names of the parameters of a model variant (see `model_spec`) needed to evaluate it."
function model_param_names(spec)
    spec.family === :t4 && return (:σ_comp, :σ_mach, :σ_y, :z_comp, :z_mach, :γ)
    cars = spec.pace_scale ? () : (:z_mach,)          # pace-scale models use centred b_mach
    comp = spec.centred_drivers ? :a_comp : :z_comp        # centred drivers (#28)
    races = spec.race_hier ? (:μ_γ, :τ_γ, :γ) : spec.race_mean ? (:μ_γ, :γ) : (:γ,)   # race intercepts (#32)
    return (:σ_comp, :σ_mach, :σ, comp, cars..., races..., loss_param_names(spec.loss_duration, spec.era, spec.track)...,
            pace_param_names(spec.pace_scale, get(spec, :ncres, false), get(spec, :kc, false), get(spec, :kst, false))..., age_param_names(spec.age)..., big_param_names(spec.big_loss)...,
            slope_param_names(spec.slopes)..., dev_param_names(spec.dev)..., circ_param_names(spec.circ)...,
            walk_param_names(spec.walk, get(spec, :walk1, false), get(spec, :wlev, false), get(spec, :wst, false), get(spec, :wstc, false),
                             get(spec, :wstl, false), get(spec, :wlevn, false))..., works_param_names(get(spec, :works, false))...,
            dinc_param_names(get(spec, :dinc, false))...)
end

"""
    identified_draws(chain, g::GapData, model; cov) -> (cols, label)

The draws behind [`identified_convergence`](@ref): `cols[group]` is a draws ×
chains × levels array of that group's identified quantities, and `label(group, k)`
names level `k`. Used by [`slow_directions`](@ref) too. `cov` defaults to the model's
covariates from the reference tables; pass it for other car tables (e.g. simulated ones).
For walk models the `cars` group is each car-season's residual, not its total.
"""
function identified_draws(chain, g::GapData, model::AbstractString; cov = LossCovariates(g, model_spec(model)))
    spec = model_spec(model)
    sc = season_counts(g)
    names = model_param_names(spec)
    basis = spec.age ? AgeCurveBasis(g) : nothing
    ni, nc = size(chain[:σ_comp])
    scalar_names = [n for n in names if chain[n][1, 1] isa Real]
    # short vectors read like scalars (#81): circuit-slope scales and track coefficients
    small_names = [n for n in names if n in (:τ_cx, :β_track_π, :β_track_λ)]
    small_labels = [string(n, "[", k, "]") for n in small_names for k in eachindex(chain[n][1, 1])]
    cols = Dict{Symbol,Array{Float64,3}}()
    function put!(group, i, c, v)
        A = get!(() -> Array{Float64,3}(undef, ni, nc, length(v)), cols, group)
        A[i, c, :] = v
    end
    for c in 1:nc, i in 1:ni
        θ = draw(chain, names, i, c)
        put!(:drivers, i, c, haskey(θ, :a_comp) ? sum_to_zero(θ.a_comp) : θ.σ_comp .* sum_to_zero(θ.z_comp))
        put!(:cars, i, c, haskey(θ, :b_mach) || haskey(θ, :z_res) ? sum_to_zero_by(car_residuals(θ, cov), g.mach_season, sc) :
                          θ.σ_mach .* sum_to_zero_by(θ.z_mach, g.mach_season, sc))
        haskey(θ, :τ_slope) && put!(:slopes, i, c, career_slopes(θ.τ_slope, θ.u_slope))
        haskey(θ, :τ_dev) && put!(:dev, i, c, dev_trends(θ.τ_dev, θ.u_dev, g, sc))
        haskey(θ, :τ_cx) && put!(:circ, i, c, circuit_slopes(θ.τ_cx, θ.u_cx, g, sc))
        if haskey(θ, :τ_ch)          # chassis and engine walk states, centred within season (#95)
            xc, xe = car_walk_states(θ, cov.cars)
            put!(:chassis, i, c, xc); put!(:engine, i, c, xe)
        end
        haskey(θ, :τ_dinc) && put!(:dinc, i, c, driver_incident_offsets(θ))
        basis === nothing || put!(:age, i, c, age_curve(θ.τ_age, θ.e_age, basis))
        if spec.family === :pl
            πr, λr = race_loss(θ, cov, spec.loss_duration, spec.era)
            put!(:races, i, c, vcat(πr, λr, haskey(θ, :τ_κ) ? race_pace_scale(θ, cov) : Float64[]))
        end
        put!(:scalars, i, c, Float64[(θ[n] for n in scalar_names)...; (θ[n] for n in small_names)...])
    end
    label(group, k) = group === :drivers ? g.competitors[k] : group in (:cars, :dev) ? g.machines[k] :
        group === :slopes ? g.competitors[k] : group === :age ? "age $(g.age_years[k])" :
        group === :scalars ? (k <= length(scalar_names) ? string(scalar_names[k]) : small_labels[k - length(scalar_names)]) :
        group === :circ ? (nm = length(g.machines); "$(g.machines[mod1(k, nm)]) x$(cld(k, nm))") :
        group === :chassis ? cov.cars.chassis.names[k] : group === :engine ? cov.cars.engine.names[k] :
        group === :dinc ? g.competitors[k] :
        (nr = length(g.races); k <= nr ? "π $(g.races[k])" : k <= 2nr ? "λ $(g.races[k - nr])" : "κ $(g.races[k - 2nr])")
    return cols, label
end

"""
    identified_convergence(chain, g::GapData, model) -> NamedTuple

R-hat and bulk ESS of the quantities the likelihood identifies, grouped:
`drivers` (centred driver effects σ_comp·(z - z̄)), `cars` (centred within
season), `slopes` (centred career slopes), `dev` (in-season development trends, #14),
`age` (the career curve), `races`
(per-race incident probability, mean loss and pace scale), and `scalars` (every
scalar parameter: scales, loss intercepts and slopes, step scales). Each group
is `(; max_rhat, min_ess, worst)`, with `worst` naming the level with the
lowest ESS; `overall` is the worst over groups.

Unlike `convergence_summary`, which reports the raw sampled coordinates, this
leaves out directions only the prior sees (the mean of z, of each season's car
effects, of the slopes) and the standardised coordinates behind scaled effects
(z = a/σ_comp inherits σ_comp's mixing), so it measures convergence of what the
results use (#25).
"""
function identified_convergence(chain, g::GapData, model::AbstractString; cov = LossCovariates(g, model_spec(model)))
    cols, label = identified_draws(chain, g, model; cov)
    groups = Dict{Symbol,Any}()
    for (group, A) in cols
        e, r = MCMCDiagnosticTools.ess(A), MCMCDiagnosticTools.rhat(A)
        k = argmin(e)
        groups[group] = (; max_rhat = maximum(r), min_ess = minimum(e), worst = label(group, k))
    end
    overall = (; max_rhat = maximum(v.max_rhat for v in values(groups)),
               min_ess = minimum(v.min_ess for v in values(groups)))
    return (; overall, NamedTuple(groups)...)
end

"""
    slow_directions(A, labels; k = 3, top = 8, holdout = true) -> Vector{NamedTuple}

The combinations of quantities on which the chains disagree most (#132). `A` is a
draws × chains × quantities array (as for MCMCDiagnosticTools) and `labels` names the
quantities. Each quantity is standardised by its pooled posterior sd; the chains'
means then form a chains × quantities matrix, and the leading right singular vectors
of its centred rows are the slow directions.

Returns up to `k` directions (at most chains − 1), each `(; share, rhat, ess,
loadings)`: its share of the between-chain spread, the R-hat and bulk ESS of the draws
projected onto it, and the `top` largest loadings as `label => weight`.

With many quantities and few chains, some direction always shows disagreement by
chance, and measuring it on the draws it was chosen from overstates its R-hat. With
`holdout` (the default) the directions are found on the first half of the draws and
their R-hat and ESS measured on the second half, so a direction that is only noise
comes out near R-hat 1. A direction that keeps a high held-out R-hat and whose
loadings group, say, a team's drivers with its car is a real ridge, which a
reparametrisation can remove (`rwrel` on the lap model, #116).
"""
function slow_directions(A::AbstractArray{<:Real,3}, labels::AbstractVector; k::Integer = 3, top::Integer = 8,
                         holdout::Bool = true)
    ni, nc, p = size(A)
    length(labels) == p || throw(ArgumentError("$(length(labels)) labels for $p quantities"))
    sdv = vec(std(reshape(A, ni * nc, p); dims = 1))
    keep = findall(>(0), sdv)
    Z = A[:, :, keep] ./ reshape(sdv[keep], 1, 1, :)
    fit_i, test_i = holdout ? (1:(ni ÷ 2), (ni ÷ 2 + 1):ni) : (1:ni, 1:ni)
    M = dropdims(mean(Z[fit_i, :, :]; dims = 1); dims = 1)        # chains × quantities
    F = svd(M .- mean(M; dims = 1))
    tot = sum(abs2, F.S)
    out = NamedTuple[]
    for j in 1:min(k, nc - 1, length(F.S))
        v = F.V[:, j]
        pr = reshape(reduce(+, (Z[test_i, :, q] .* v[q] for q in eachindex(v))), length(test_i), nc, 1)
        ord = sortperm(abs.(v); rev = true)[1:min(top, length(v))]
        push!(out, (; share = F.S[j]^2 / tot, rhat = MCMCDiagnosticTools.rhat(pr)[1], ess = MCMCDiagnosticTools.ess(pr)[1],
                    loadings = [labels[keep[q]] => v[q] for q in ord]))
    end
    return out
end

"""
    slow_trends(A, labels; segments = 10, nperm = 100, alpha = 0.05, lam = 0.5, kmax = 10,
                rng = Xoshiro(1)) -> Vector{NamedTuple}

Coherent slow movements across many quantities, with a permutation test for each
(#132), after the batch-effect step of LIMBR (github.com/aleccrowell/LIMBR, itself
after Leek and Storey's surrogate variable analysis). `A` is draws × chains ×
quantities, as for [`slow_directions`](@ref).

1. Each chain is cut into `segments` blocks. The matrix `R` has a row per quantity
   and a column per chain-segment: the segment's mean, minus the quantity's
   posterior mean, over its posterior sd.
2. The SVD of `R` (columns centred) gives *trends* over the chain-segments, each
   with its share of the variance.
3. Permuting each row of `R` independently keeps every quantity's own spread but
   breaks any movement shared between quantities. Trend k is significant when the
   permuted matrices' k-th share beats the observed one in under `alpha` of `nperm`
   permutations; trends are taken in order until one isn't (at most `kmax` are reported).
4. Each quantity is regressed on each significant trend. Its members are the
   quantities with Storey q-value below `alpha`, π₀ (the share unrelated to the
   trend, capped at 1) estimated from the p-values above `lam`. LIMBR instead keeps
   the (1 − π₀)·n smallest p-values and stops when π₀ ≥ 1, which on 300 quantities
   with 20 members dropped a clear trend (π₀ = 1.02 by chance). An SVD of the
   members' rows refines the trend.

5. Whether the trend *mixes slowly*. Posterior correlation alone makes coherent
   trends (a team's walk states move together however well the chains mix), so step
   3 finds structure, not slowness. Permuting the order of the chain-segments, the
   same way for every quantity, keeps that correlation but breaks any order. Against
   it, `between` (the trend's share of variance between chains: the chains sit in
   different places, a ridge) and `lag1` (its autocorrelation from one segment to
   the next within a chain: it drifts) get p-values `p_between` and `p_drift`.

Returns per significant trend `(; p, share, between, p_between, lag1, p_drift, members,
trend)`: the structure p-value, the variance share, the two mixing statistics and their
p-values, the members as `label => correlation` from strongest, and the refined trend
over the columns (chain 1's segments first). A trend with neither mixing p-value small
is posterior correlation that the chains explore well.

On the lap fits of #116 it did *not* flag the known driver/car ridge (best p_between
0.14 before `rwrel`), which `slow_directions` found (held-out R-hat 1.033, ESS 70;
after `rwrel` 1.006, 264). The trends with most variance are posterior correlation,
not the directions the chains disagree on, and 10 segments per chain give the mixing
test little power. Use `slow_directions` to find ridges; this shows which groups of
quantities move together and whether any of them drift.

A single slowly mixing quantity is not a trend: permuting keeps most of its
variance, and a trend with fewer than two members is skipped (its R-hat shows it).
Segments should be long against the autocorrelation time (e.g. 100 draws), or
smoothness alone, shared by independent quantities, can look like a trend.
"""
# share of a trend's variance between chains, and its lag-1 autocorrelation over each chain's segments
function trend_mixing(t::AbstractVector, chain_of::AbstractVector, segments::Integer)
    m = mean(t); d = t .- m; tot = sum(abs2, d)
    nc = maximum(chain_of)
    cm = [mean(t[chain_of .== c]) for c in 1:nc]
    between = sum(abs2, cm[chain_of] .- m) / tot
    lag = sum(d[i] * d[i + 1] for i in 1:(length(t) - 1) if chain_of[i] == chain_of[i + 1]) / tot
    return between, lag
end

function slow_trends(A::AbstractArray{<:Real,3}, labels::AbstractVector; segments::Integer = 10, nperm::Integer = 100,
                     alpha::Real = 0.05, lam::Real = 0.5, kmax::Integer = 10, rng::AbstractRNG = Xoshiro(1))
    ni, nc, p = size(A)
    length(labels) == p || throw(ArgumentError("$(length(labels)) labels for $p quantities"))
    L = ni ÷ segments
    L >= 1 || throw(ArgumentError("$ni draws can't be cut into $segments segments"))
    sdv = vec(std(reshape(A, ni * nc, p); dims = 1))
    keep = findall(>(0), sdv)
    R = Matrix{Float64}(undef, length(keep), nc * segments)
    for (r, q) in enumerate(keep), c in 1:nc, sgm in 1:segments
        R[r, (c - 1) * segments + sgm] = mean(@view A[((sgm - 1) * L + 1):(sgm * L), c, q])
    end
    R .= (R .- mean(R; dims = 2)) ./ sdv[keep]
    shares(X) = (sv = svdvals(X .- mean(X; dims = 1)); sv .^ 2 ./ sum(abs2, sv))
    tk = shares(R)
    beat = zeros(length(tk))
    for _ in 1:nperm
        Rs = similar(R)
        for r in axes(R, 1)
            Rs[r, :] = R[r, randperm(rng, size(R, 2))]
        end
        ts = shares(Rs)
        n = min(length(ts), length(tk))
        beat[1:n] .+= ts[1:n] .> tk[1:n]
    end
    pk = beat ./ nperm
    nsig = something(findfirst(>=(alpha), pk), length(pk) + 1) - 1
    F = svd(R .- mean(R; dims = 1))
    chain_of = repeat(1:nc; inner = segments)
    out = NamedTuple[]
    for j in 1:min(nsig, kmax)
        v = F.V[:, j]
        ν = length(v) - 2
        rs = [cor(view(R, r, :), v) for r in axes(R, 1)]
        ps = [2exp(tdistlogccdf(ν, abs(r) * sqrt(ν / max(1 - r^2, eps())))) for r in rs]
        π0 = min(1.0, count(>(lam), ps) / (length(ps) * (1 - lam)))
        o = sortperm(ps); n = length(ps)
        q = similar(ps); m = 1.0
        for i in n:-1:1                       # Storey q-values, monotone from the largest p
            m = min(m, π0 * n * ps[o[i]] / i); q[o[i]] = m
        end
        mem = findall(<(alpha), q)
        length(mem) < 2 && continue           # one quantity alone: its own R-hat shows it
        trend = v
        if length(mem) > 1
            Fs = svd(R[mem, :] .- mean(R[mem, :]; dims = 1))
            cs = [abs(cor(Fs.V[:, i], v)) for i in axes(Fs.V, 2)]
            trend = Fs.V[:, argmax(cs)]
            trend = sign(cor(trend, v)) * trend
        end
        between, lag1 = trend_mixing(trend, chain_of, segments)
        # mixing test: permuting the segments' order, jointly for every quantity, keeps the
        # posterior correlation that made the trend but breaks any chain or time structure
        bp = lp = 0
        for _ in 1:nperm
            perm = trend[randperm(rng, length(trend))]
            b, l = trend_mixing(perm, chain_of, segments)
            bp += b >= between; lp += l >= lag1
        end
        ord = mem[sortperm(abs.(rs[mem]); rev = true)]
        push!(out, (; p = pk[j], share = tk[j], between, p_between = bp / nperm, lag1, p_drift = lp / nperm,
                    members = [labels[keep[r]] => rs[r] for r in ord], trend))
    end
    return out
end

"""
    chain_modes(chain, names; threshold = 1.1) -> Vector{String}

Group the chains of `chain` into modes: two chains are in the same mode when the
R-hat of each scalar parameter in `names`, computed from those two chains alone,
is below `threshold`, and modes are the connected groups of such pairs. Returns
one label per chain ("A", "B", …, in order of each mode's first chain).

Chains that agree on every parameter get one label, so a converged fit has a
single mode. Used to report fits whose chains settle in different solutions
(#55, #64) by mode instead of pooled.
"""
function chain_modes(chain, names; threshold = 1.1)
    ni, nc = size(chain[first(names)])
    X = Array{Float64,3}(undef, ni, nc, length(names))
    for (p, n) in enumerate(names)
        X[:, :, p] = chain[n]
    end
    parent = collect(1:nc)
    root(i) = parent[i] == i ? i : (parent[i] = root(parent[i]))
    for i in 1:nc, j in (i + 1):nc
        maximum(MCMCDiagnosticTools.rhat(X[:, [i, j], :])) < threshold && (parent[root(j)] = root(i))
    end
    roots = root.(1:nc)
    order = unique(roots)
    return [string(Char('A' + findfirst(==(r), order) - 1)) for r in roots]
end
