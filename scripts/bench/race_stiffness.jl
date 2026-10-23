# Which posterior direction limits NUTS's step size (#102)?
#
#   julia --project scripts/bench/race_stiffness.jl <model> [data=<mode>] [draws=200] [iters=60] [as=wlev|kc|kst|wlev+kst…]
#
# From a model's saved chains (output/gap_<data>_<model>_chain<k>.jls): at the
# posterior mean, in the sampler's unconstrained coordinates scaled by each
# coordinate's posterior sd (what a diagonal mass matrix adapts to), power
# iteration on the negative Hessian of the log density (Hessian-vector products
# from central differences of the gradient) gives its largest eigenvalue λ. The
# leapfrog step size can't exceed about 2/√λ, so a stiff direction forces small
# steps and long trees. Prints λ, 2/√λ and the parameters with the largest
# weights in that direction. `as=wlev`, `as=kc`, `as=wlev+kc`: the same draws mapped
# to the level parametrisation of the walks (`walk_to_levels`) and/or centred
# pace-scale steps (s_κ = τ_κ·e_κ), evaluated in the matching model: the same
# posterior in other coordinates, so no new fit is needed.

using Diomedes, Random, Serialization, Statistics, Printf, LinearAlgebra, Turing
include(joinpath(@__DIR__, "..", "gap_data.jl"))
const DPPL = Turing.DynamicPPL
const LDP = DPPL.LogDensityProblems

mode, args = data_arg(ARGS)
opt(k, d) = (i = findfirst(a -> startswith(a, "$k="), args); i === nothing ? d : parse(Int, split(args[i], "=")[2]))
model_name = first(filter(a -> !occursin("=", a), args))
asi = findfirst(a -> startswith(a, "as="), args)
as = asi === nothing ? String[] : split(split(args[asi], "=")[2], "+")
aswlev, askc, askst, aswst = "wlev" in as, "kc" in as, "kst" in as, "wst" in as
ndraws, iters = opt("draws", 200), opt("iters", 60)
spec = model_spec(model_name)
_, g = gap_data(mode; io = devnull)
chain = gap_chains(mode, model_name)
names = Diomedes.model_param_names(spec)
spec = merge(spec, (; wlev = spec.wlev || aswlev, kc = spec.kc || askc, kst = spec.kst || askst, wst = spec.wst || aswst))
m = paceloss_effects(g; loss_duration = spec.loss_duration, era = spec.era, pace_scale = spec.pace_scale,
                     age = spec.age, big_loss = spec.big_loss, slopes = spec.slopes, dev = spec.dev,
                     centred_drivers = spec.centred_drivers, driver_ν = spec.driver_ν, race_hier = spec.race_hier,
                     race_mean = spec.race_mean, circ = spec.circ, track = spec.track, walk = spec.walk,
                     walk1 = spec.walk1, works = spec.works, dinc = spec.dinc, ncres = spec.ncres, wlev = spec.wlev, kc = spec.kc, kst = spec.kst, wst = spec.wst)

# unconstrained coordinates of a draw: the model's linked VarInfo
ni, nc = size(chain[:σ_comp])
cwm = aswlev || aswst ? LossCovariates(g, spec).cars : nothing
function towst(θ)
    xc, xe = Diomedes.car_walk_raw(θ, cwm)
    return NamedTuple(k === :z_ch ? (:xs_ch => xc) : k === :z_en ? (:xs_en => xe) : (k => v) for (k, v) in pairs(θ))
end
tokc(θ) = NamedTuple(k === :e_κ ? (:s_κ => θ.τ_κ .* v) : (k => v) for (k, v) in pairs(θ))
# the log path (mean zero) plus a mean level drawn from its prior N(0, 0.05), the mean's posterior
tokst(θ) = NamedTuple(k === :e_κ ? (:l_κ => Diomedes.rw_path(θ.τ_κ .* v) .+ 0.05 * randn(rng)) : (k => v) for (k, v) in pairs(θ))
pt(θ) = (θ = aswst ? towst(θ) : aswlev ? Diomedes.walk_to_levels(θ, cwm) : θ; askst ? tokst(θ) : askc ? tokc(θ) : θ)   # the draw in the model's coordinates
unc(θ) = (vi = DPPL.VarInfo(m, DPPL.InitFromParams(pt(θ)), DPPL.LinkAll()); collect(Float64, vi[:]))
rng = Xoshiro(1)
picks = [(rand(rng, 1:ni), rand(rng, 1:nc)) for _ in 1:ndraws]
X = reduce(hcat, [unc(Diomedes.draw(chain, names, i, c)) for (i, c) in picks])      # coords × draws
xbar, s = vec(mean(X; dims = 2)), vec(std(X; dims = 2))
labels = String[]
for (n, v) in pairs(pt(Diomedes.draw(chain, names, 1, 1)))
    v isa AbstractVector ? append!(labels, ["$n[$j]" for j in eachindex(v)]) : push!(labels, string(n))
end
length(labels) == length(xbar) || error("label count $(length(labels)) ≠ coordinates $(length(xbar))")

vi0 = DPPL.VarInfo(m, DPPL.InitFromParams(pt(Diomedes.draw(chain, names, 1, 1))), DPPL.LinkAll())
f = DPPL.LogDensityFunction(m, DPPL.getlogjoint_internal, vi0; adtype = Diomedes.gap_sampler().adtype)
grad(x) = LDP.logdensity_and_gradient(f, x)[2]
h = 1e-3
Hv(v) = -s .* (grad(xbar .+ h .* s .* v) .- grad(xbar .- h .* s .* v)) ./ (2h)       # whitened −Hessian × v
v = normalize(randn(rng, length(xbar)))
λ = 0.0
for k in 1:iters
    w = Hv(v)
    global λ = dot(v, w)
    global v = normalize(w)
    k % 10 == 0 && (@printf("  iter %d: λ = %.1f\n", k, λ); flush(stdout))
end
@printf("%s%s: %d coordinates; largest whitened curvature λ = %.1f, so step size ≤ ~%.4f\n",
        model_name, isempty(as) ? "" : " (as $(join(as, "+")))", length(xbar), λ, 2 / sqrt(λ))
println("largest weights in that direction:")
for j in first(sortperm(abs.(v); rev = true), 15)
    @printf("  %-22s %+.3f   (posterior sd %.3g)\n", labels[j], v[j], s[j])
end
