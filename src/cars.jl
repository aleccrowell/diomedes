# Chassis and engine walks for the race model's car effects (#95), from the
# hand-checked tables in reference/cars/ (see its README).

const CAR_DIR = joinpath(dirname(@__DIR__), "reference", "cars")

"""
    car_table(dir = CAR_DIR) -> NamedTuple

The car reference tables (#95): `entries` (each Jolpica race entry matched to
F1DB's chassis constructor and engine maker), `works` (the works that built each
constructor's cars), `engines` (the builder behind each engine name), `ownership`
(changes of a works' owner, by the first season raced under the new owner) and
`regulations` (major pace-relevant regulation changes, tagged chassis and/or
engine). Pass it to `CarWalks` or the race model (`paceloss_effects(…; walk = true)`).
"""
function car_table(dir::AbstractString = CAR_DIR)
    rd(f) = CSV.read(joinpath(dir, f), DataFrame; stringtype = String)
    return (; entries = rd("entries.csv"), works = rd("works.csv"), engines = rd("engine_builders.csv"),
            ownership = rd("ownership_changes.csv"), regulations = rd("pace_regulations.csv"))
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
(see `WalkStates`). Each row's chassis is its F1DB constructor's works and its
engine is its engine maker's builder, for that season. A row without an F1DB
match takes its car-season's most common chassis and engine.
"""
struct CarWalks
    chassis::WalkStates
    engine::WalkStates
    row_ch::Vector{Int}
    row_en::Vector{Int}
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
    return CarWalks(ch, en, [ich[(w, y)] for (w, y) in zip(wk, rows.season)],
                    [ien[(b, y)] for (b, y) in zip(bd, rows.season)])
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
both families: with one each, they were indistinguishable (#95).
"""
function car_walk_states(θ, cw::CarWalks)
    xc = walk_path(walk_steps(θ.τ_ch, θ.σ0_ch, θ.δ_own, θ.δ_reg, θ.z_ch, cw.chassis), cw.chassis.prev)
    xe = walk_path(walk_steps(θ.τ_en, θ.σ0_en, zero(θ.δ_own), θ.δ_reg, θ.z_en, cw.engine), cw.engine.prev)
    return center_by(xc, cw.chassis.season, cw.chassis.counts), center_by(xe, cw.engine.season, cw.engine.counts)
end
