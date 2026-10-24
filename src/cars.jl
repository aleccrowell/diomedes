# Chassis and engine walks for the race model's car effects (#95), from the
# hand-checked tables in reference/cars/ (see its README).

const CAR_DIR = joinpath(dirname(@__DIR__), "reference", "cars")

"""
    car_table(dir = CAR_DIR) -> NamedTuple

The car reference tables (#95): `entries` (each Jolpica race entry matched to
F1DB's chassis constructor and engine maker), `works` (the works that built each
constructor's cars), `engines` (the builder behind each engine name), `ownership`
(changes of a works' owner, by the first season raced under the new owner),
`regulations` (major pace-relevant regulation changes, tagged chassis and/or
engine) and `works_engines` (seasons in which an engine builder owned the
chassis works it supplied: its works team, #100). Pass it to `CarWalks` or the race model (`paceloss_effects(…; walk = true)`).
"""
function car_table(dir::AbstractString = CAR_DIR)
    rd(f) = CSV.read(joinpath(dir, f), DataFrame; stringtype = String)
    return (; entries = rd("entries.csv"), works = rd("works.csv"), engines = rd("engine_builders.csv"),
            ownership = rd("ownership_changes.csv"), regulations = rd("pace_regulations.csv"),
            works_engines = rd("works_engines.csv"))
end

"""
    WalkStates

The states of one family of random walks over seasons (chassis works or engine
builders), one state per (walk, season) with a race in the data, ordered by walk
then season:

- `prev`: the walk's previous state (0 for its first season);
- `first`: the state is a walk's first season;
- `rootgap`: √(years since the previous state), so a walk that skipped seasons
  takes a proportionally wider step;
- `own`, `reg`: the step crosses a change of ownership (chassis only) or a
  regulation change tagged for this family;
- `season`, `counts`: each state's season index (as `LossCovariates.season`) and
  the number of states per season, for centring within season;
- `names`: "walk season" labels.
"""
struct WalkStates
    prev::Vector{Int}
    first::Vector{Bool}
    rootgap::Vector{Float64}
    own::Vector{Float64}
    reg::Vector{Float64}
    season::Vector{Int}
    counts::Vector{Int}
    names::Vector{String}
end

"""
    CarWalks(g::GapData, cars = car_table())

Per-row chassis and engine walk states for the race model (#95), rows in GapData
order (timed, then lapped): `row_ch` and `row_en` index `chassis` and `engine`
(see `WalkStates`). `row_works` is 1 for a row whose engine builder owned its
chassis works that season (the builder's works team, #100), else 0. Each row's chassis is its F1DB constructor's works and its
engine is its engine maker's builder, for that season. A row without an F1DB
match takes its car-season's most common chassis and engine.
"""
struct CarWalks
    chassis::WalkStates
    engine::WalkStates
    row_ch::Vector{Int}
    row_en::Vector{Int}
    row_works::Vector{Float64}
end

function _lookup_range(tab, key::Symbol, val, y, out::Symbol)
    i = findall(r -> r[key] == val && r.first_year <= y <= r.last_year, eachrow(tab))
    length(i) == 1 || throw(ArgumentError("$(length(i)) rows for $val $y in the car tables ($key)"))
    return tab[only(i), out]
end

function _walk_states(keys::Vector{Tuple{String,Int}}, s0::Int, n_seasons::Int, own::Set{Tuple{String,Int}},
                      reg_years::Vector{Int})
    states = sort(unique(keys))
    prev, first, rootgap, ow, rg = Int[], Bool[], Float64[], Float64[], Float64[]
    for (k, (w, y)) in enumerate(states)
        p = k > 1 && states[k - 1][1] == w ? k - 1 : 0
        y0 = p == 0 ? y : states[p][2]
        push!(prev, p); push!(first, p == 0); push!(rootgap, sqrt(max(y - y0, 1)))
        push!(ow, p > 0 && (w, y) in own ? 1.0 : 0.0)
        push!(rg, p > 0 && any(r -> y0 < r <= y, reg_years) ? 1.0 : 0.0)
    end
    season = [y - s0 + 1 for (_, y) in states]
    counts = [count(==(s), season) for s in 1:n_seasons]
    return WalkStates(prev, first, rootgap, ow, rg, season, counts, ["$w $y" for (w, y) in states]),
           Dict(s => k for (k, s) in enumerate(states))
end

function CarWalks(g::GapData, cars = car_table())
    t, c = g.rows[g.rows.kind .== :timed, :], g.rows[g.rows.kind .== :lapped, :]
    rows = vcat(t, c; cols = :union)
    # each row's F1DB constructor and engine maker; unmatched rows: their car-season's most common
    ent = Dict{Tuple{Int,Int,String,String},Tuple{String,String}}()
    bycar = Dict{Tuple{String,Int},Dict{Tuple{String,String},Int}}()
    for e in eachrow(cars.entries)
        ce = (e.f1db_constructor, e.f1db_engine)
        get!(ent, (e.season, e.round, e.driver_id, e.machine_id), ce)
        d = get!(bycar, (e.machine_id, e.season), Dict{Tuple{String,String},Int}())
        d[ce] = get(d, ce, 0) + 1
    end
    ce = map(eachrow(rows)) do r
        get(ent, (r.season, r.round, r.competitor_id, r.machine_id)) do
            d = get(bycar, (r.machine_id, r.season), nothing)
            d === nothing && throw(ArgumentError("no F1DB entry for $(r.machine_id) $(r.season)"))
            first(argmax(last, collect(d)))
        end
    end
    wk = [_lookup_range(cars.works, :f1db_constructor, ch, r.season, :works_id) for ((ch, _), r) in zip(ce, eachrow(rows))]
    bd = [_lookup_range(cars.engines, :f1db_engine, en, r.season, :builder_id) for ((_, en), r) in zip(ce, eachrow(rows))]
    s0 = minimum(g.race_season)
    ns = maximum(g.race_season) - s0 + 1
    own = Set((r.works_id, r.season) for r in eachrow(cars.ownership))
    regs = cars.regulations
    ch, ich = _walk_states(collect(zip(wk, rows.season)), s0, ns, own, regs.season[regs.chassis .== 1])
    en, ien = _walk_states(collect(zip(bd, rows.season)), s0, ns, Set{Tuple{String,Int}}(), regs.season[regs.engine .== 1])
    we = hasproperty(cars, :works_engines) ? cars.works_engines : DataFrame(builder_id = String[], works_id = String[],
                                                                            first_year = Int[], last_year = Int[])
    isworks(b, w, y) = any(r -> r.builder_id == b && r.works_id == w && r.first_year <= y <= r.last_year, eachrow(we))
    return CarWalks(ch, en, [ich[(w, y)] for (w, y) in zip(wk, rows.season)],
                    [ien[(b, y)] for (b, y) in zip(bd, rows.season)],
                    [Float64(isworks(b, w, y)) for (b, w, y) in zip(bd, wk, rows.season)])
end

"""
    walk_steps(τ, σ0, δ_own, δ_reg, z, ws::WalkStates) -> Vector

Each state's step from its walk's previous state: a first season is σ0·z; a later
one is τ·√gap·exp(δ_own·own + δ_reg·reg)·z (see `WalkStates`).
"""
function walk_steps(τ, σ0, δ_own, δ_reg, z, ws::WalkStates)
    f = Float64.(ws.first)          # arithmetic, not ifelse: ReverseDiff's broadcast makes the mask a Dual
    return z .* (f .* σ0 .+ (1 .- f) .* τ .* ws.rootgap .* exp.(δ_own .* ws.own .+ δ_reg .* ws.reg))
end

"""
    walk_path(steps, prev) -> Vector

Walk states from their steps: state k = steps[k] + state[prev[k]] (prev[k] < k, 0
for a walk's first state). One pass with a hand-written rule.
"""
function walk_path(steps, prev::Vector{Int})
    out = similar(steps, promote_type(eltype(steps), Float64))
    for k in eachindex(prev)
        out[k] = prev[k] == 0 ? steps[k] : steps[k] + out[prev[k]]
    end
    return out
end
function ChainRulesCore.rrule(::typeof(walk_path), steps, prev::Vector{Int})
    function pullback(ȳ)
        s̄ = collect(Float64, ChainRulesCore.unthunk(ȳ))
        for k in reverse(eachindex(prev))
            prev[k] == 0 || (s̄[prev[k]] += s̄[k])
        end
        return NoTangent(), s̄, NoTangent()
    end
    return walk_path(steps, prev), pullback
end
ReverseDiff.@grad_from_chainrules walk_path(steps::ReverseDiff.TrackedArray, prev::Vector{Int})

"""
    walk_level_path(v, prev) -> Vector

Walk states in the level parametrisation (`_wlev`, #102): for a walk's first
state v holds the walk's level L (its mean state over its seasons); for the
others v holds the step. States are L + c_k − mean(c) with c the cumulative
steps (c = 0 at the first state). A long, well-observed walk's level is then
one coordinate: in the step form it is spread over the first state and every
early step, a direction that limited the sampler's step size (Ferrari, 76
seasons). Linear in v, with a hand-written rule.
"""
function walk_level_path(v, prev::Vector{Int})
    T = promote_type(eltype(v), Float64)
    n = length(prev)
    wid, c = zeros(Int, n), zeros(T, n)
    nw = 0
    for k in 1:n
        if prev[k] == 0
            nw += 1; wid[k] = nw
        else
            wid[k] = wid[prev[k]]; c[k] = c[prev[k]] + v[k]
        end
    end
    csum, cnt, lev = zeros(T, nw), zeros(Int, nw), zeros(T, nw)
    for k in 1:n
        csum[wid[k]] += c[k]; cnt[wid[k]] += 1
        prev[k] == 0 && (lev[wid[k]] = v[k])
    end
    return [lev[wid[k]] + c[k] - csum[wid[k]] / cnt[wid[k]] for k in 1:n]
end
function ChainRulesCore.rrule(::typeof(walk_level_path), v, prev::Vector{Int})
    function pullback(ȳ)
        ȳ = collect(Float64, ChainRulesCore.unthunk(ȳ))
        n = length(prev)
        wid = zeros(Int, n); nw = 0
        for k in 1:n; wid[k] = prev[k] == 0 ? (nw += 1) : wid[prev[k]]; end
        tot, cnt = zeros(nw), zeros(Int, nw)
        for k in 1:n; tot[wid[k]] += ȳ[k]; cnt[wid[k]] += 1; end
        # ∂/∂c_k = ȳ_k − mean of ȳ over the walk; then accumulate down each walk (c_k feeds all later c)
        dc = [ȳ[k] - tot[wid[k]] / cnt[wid[k]] for k in 1:n]
        acc = copy(dc)
        for k in reverse(1:n)
            prev[k] == 0 || (acc[prev[k]] += acc[k])
        end
        dv = [prev[k] == 0 ? tot[wid[k]] : acc[k] for k in 1:n]
        return NoTangent(), dv, NoTangent()
    end
    return walk_level_path(v, prev), pullback
end
ReverseDiff.@grad_from_chainrules walk_level_path(v::ReverseDiff.TrackedArray, prev::Vector{Int})

"""
    walk_offsets(xc, row_ch, xe, row_en) -> Vector

Per-row car part from the walks, xc[row_ch[i]] + xe[row_en[i]], with a
hand-written rule (indexing tracked vectors row by row allocated heavily, #92).
"""
function walk_offsets(xc, row_ch::Vector{Int}, xe, row_en::Vector{Int})
    out = zeros(promote_type(eltype(xc), eltype(xe)), length(row_ch))
    for i in eachindex(out); out[i] = xc[row_ch[i]] + xe[row_en[i]]; end
    return out
end
function ChainRulesCore.rrule(::typeof(walk_offsets), xc, row_ch::Vector{Int}, xe, row_en::Vector{Int})
    function pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        ∂c, ∂e = zeros(length(xc)), zeros(length(xe))
        for i in eachindex(row_ch); ∂c[row_ch[i]] += ȳ[i]; ∂e[row_en[i]] += ȳ[i]; end
        return NoTangent(), ∂c, NoTangent(), ∂e, NoTangent()
    end
    return walk_offsets(xc, row_ch, xe, row_en), pullback
end
ReverseDiff.@grad_from_chainrules walk_offsets(xc::ReverseDiff.TrackedArray, row_ch::Vector{Int},
                                               xe::ReverseDiff.TrackedArray, row_en::Vector{Int})

"""
    car_walk_states(θ, cw::CarWalks) -> (chassis, engine)

The chassis and engine walk states for parameters θ, each centred within season
(a shift shared by a season's walks is absorbed by the race intercepts), in
average-season % units (the likelihood scales them by κ, like the driver effects).
A step across a tagged regulation change widens by exp(δ_reg), one multiplier for
both families (`_walk1`, #97: with one each, they were indistinguishable), or by
exp(δ_reg_ch) and exp(δ_reg_en) (`_walk`, the published #95 fit).
"""
function car_walk_states(θ, cw::CarWalks)
    xc, xe = car_walk_raw(θ, cw)
    return center_by(xc, cw.chassis.season, cw.chassis.counts), center_by(xe, cw.engine.season, cw.engine.counts)
end

# The walk states before centring within season, in either parametrisation: steps
# (z_ch, z_en; the first state is σ0·z) or levels (zl_ch, zl_en, `_wlev`; the first
# slot holds the walk's level, see `walk_level_path`).
function car_walk_raw(θ, cw::CarWalks)
    # one regulation multiplier (`_walk1`, #97) or one per family (`_walk`, the #95 fit)
    δc, δe = haskey(θ, :δ_reg) ? (θ.δ_reg, θ.δ_reg) : (θ.δ_reg_ch, θ.δ_reg_en)
    haskey(θ, :xs_en) && return θ.xs_ch, θ.xs_en               # states sampled directly (`_wst`)
    if haskey(θ, :xs_ch) && haskey(θ, :zl_en)   # chassis states directly, engine levels (`_wstl`)
        return θ.xs_ch, walk_level_path(walk_level_inputs(θ.τ_en, zero(θ.δ_own), δe, θ.zl_en, cw.engine), cw.engine.prev)
    end
    if haskey(θ, :xs_ch)        # chassis states sampled directly, engine steps (`_wstc`)
        return θ.xs_ch, walk_path(walk_steps(θ.τ_en, θ.σ0_en, zero(θ.δ_own), δe, θ.z_en, cw.engine), cw.engine.prev)
    end
    if haskey(θ, :zn_en)        # levels, the engine's non-centred (`_wlevn`)
        return walk_level_path(walk_level_inputs(θ.τ_ch, θ.δ_own, δc, θ.zl_ch, cw.chassis), cw.chassis.prev),
               walk_level_path(walk_level_inputs(θ.τ_en, zero(θ.δ_own), δe, θ.zn_en, cw.engine, θ.σ0_en), cw.engine.prev)
    end
    if haskey(θ, :zl_ch)
        return walk_level_path(walk_level_inputs(θ.τ_ch, θ.δ_own, δc, θ.zl_ch, cw.chassis), cw.chassis.prev),
               walk_level_path(walk_level_inputs(θ.τ_en, zero(θ.δ_own), δe, θ.zl_en, cw.engine), cw.engine.prev)
    end
    return walk_path(walk_steps(θ.τ_ch, θ.σ0_ch, θ.δ_own, δc, θ.z_ch, cw.chassis), cw.chassis.prev),
           walk_path(walk_steps(θ.τ_en, θ.σ0_en, zero(θ.δ_own), δe, θ.z_en, cw.engine), cw.engine.prev)
end

# Inputs of `walk_level_path` from standardised coordinates: a first slot is the
# level as sampled, times s0 (σ0 for non-centred levels, `_wlevn`); a step is
# τ·√gap·exp(δ_own·own + δ_reg·reg)·z.
function walk_level_inputs(τ, δ_own, δ_reg, z, ws::WalkStates, s0 = 1)
    f = Float64.(ws.first)
    return z .* (f .* s0 .+ (1 .- f) .* τ .* ws.rootgap .* exp.(δ_own .* ws.own .+ δ_reg .* ws.reg))
end

"""
    walk_level_prior(θ, cw::CarWalks) -> Real

Prior correction of the level parametrisation (#102): the model gives every slot
of zl_ch, zl_en a N(0, 1) prior; the first state of each walk should instead be
N(0, σ0). Adding this keeps the prior of the step form exactly (the change of
variables is linear with unit Jacobian). Non-centred engine levels (zn_en,
`_wlevn`) are σ0_en·z, whose Jacobian cancels the 1/σ0 of the N(0, σ0) density.
"""
function walk_level_prior(θ, xc, xe, cw::CarWalks)
    lp = zero(θ.σ0_ch)
    fams = haskey(θ, :zn_en) ? ((xc, θ.zl_ch, θ.σ0_ch, cw.chassis, true), (xe, θ.zn_en, θ.σ0_en, cw.engine, false)) :
           haskey(θ, :zl_ch) ? ((xc, θ.zl_ch, θ.σ0_ch, cw.chassis, true), (xe, θ.zl_en, θ.σ0_en, cw.engine, true)) :
                               ((xe, θ.zl_en, θ.σ0_en, cw.engine, true),)     # engine levels only (`_wstl`)
    for (x, z, σ0, ws, centred) in fams
        fi = findall(ws.first)
        lp += sum(normlogpdf.(x[fi] ./ σ0)) - centred * length(fi) * log(σ0) - sum(normlogpdf.(z[fi]))
    end
    return lp
end

"""
    walk_to_levels(θ, cw::CarWalks) -> NamedTuple

The same draw in the level parametrisation (`_wlev`): z_ch and z_en become zl_ch
and zl_en, whose first slots hold each walk's level (the mean of its states
before centring within season); the steps are unchanged. With `nc_en`, the
engine's become zn_en, its levels divided by σ0_en (`_wlevn`).
"""
function walk_to_levels(θ, cw::CarWalks; nc_en::Bool = false)
    xc, xe = car_walk_raw(θ, cw)
    function lev(z, x, ws)
        zl = collect(float.(z))
        w = 0
        for k in eachindex(ws.prev)
            ws.prev[k] == 0 || continue
            idx = [k]
            for j in (k + 1):length(ws.prev); ws.prev[j] == idx[end] ? push!(idx, j) : break; end
            zl[k] = mean(x[idx])
        end
        return zl
    end
    function levn(z, x, ws)
        zl = lev(z, x, ws)
        zl[ws.first] ./= θ.σ0_en
        return zl
    end
    return NamedTuple(k === :z_ch ? (:zl_ch => lev(v, xc, cw.chassis)) :
                      k === :z_en ? (nc_en ? (:zn_en => levn(v, xe, cw.engine)) : (:zl_en => lev(v, xe, cw.engine))) :
                      (k => v) for (k, v) in pairs(θ))
end

"""
    walk_diffs(x, prev) -> Vector

Each state minus its walk's previous state (the state itself for a walk's first
season): the inverse of `walk_path`. For the state parametrisation (`_wst`, #102),
where the walk states are sampled directly and their prior is on these
differences. Linear, with a hand-written rule.
"""
walk_diffs(x, prev::Vector{Int}) = [prev[k] == 0 ? x[k] : x[k] - x[prev[k]] for k in eachindex(prev)]
function ChainRulesCore.rrule(::typeof(walk_diffs), x, prev::Vector{Int})
    function pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        dx = collect(Float64, ȳ)
        for k in eachindex(prev); prev[k] == 0 || (dx[prev[k]] -= ȳ[k]); end
        return NoTangent(), dx, NoTangent()
    end
    return walk_diffs(x, prev), pullback
end
ReverseDiff.@grad_from_chainrules walk_diffs(x::ReverseDiff.TrackedArray, prev::Vector{Int})

"""
    walk_state_prior(τ, σ0, δ_own, δ_reg, x, ws::WalkStates) -> Real

Log prior of directly sampled walk states (`_wst`, #102): a walk's first state
N(0, σ0), each later state N(previous state, τ·√gap·exp(δ_own·own + δ_reg·reg)).
"""
function walk_state_prior(τ, σ0, δ_own, δ_reg, x, ws::WalkStates)
    sd = walk_steps(τ, σ0, δ_own, δ_reg, ones(length(ws.prev)), ws)
    d = walk_diffs(x, ws.prev)
    return sum(normlogpdf.(d ./ sd)) - sum(log.(sd))
end
