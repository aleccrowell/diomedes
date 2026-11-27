# The joint race + lap model (#128): one driver ability shared by the race model
# (`paceloss_effects`) and the lap model (`lap_effects`), with model-specific deviations.
#
#     a_race = a + δ_race,      a_lap = a + δ_lap,
#
# - **a:** for every driver in either dataset, a ~ N(0, σ_a), in race units (% of race time,
#   average-season).
# - **Deviations:** only for drivers in both, δ_race ~ N(0, τ_r) and δ_lap ~ N(0, τ_l). A driver
#   in one dataset only has that model's effect equal to the shared one.
# - **One scale:** the lap part uses every lap (`prepare_laps(…; alllaps = true)`), with driver
#   and car effects on every lap, as the race model's apply to the whole race time. Both effects
#   are then a share of total time, so no scale factor links them (#128).
# - **No race counted twice:** a finishing gap is the sum of a driver's laps, so the two datasets
#   must not cover the same races. The joint fit uses the race model on the seasons before the
#   lap data (e.g. 1950–2020 against 2021–2026), and the shared ability links drivers across the
#   boundary.
# - **Everything else** in each model stays, so cars, walks and conditions adjust to the pooled
#   drivers. Both submodels are the usual models with their drivers given from outside
#   (`ext.drivers`, `drv`), and their parameters carry the prefixes `race.` and `lap.`.

"""
    JointDrivers(g::GapData, d::LapData)

Every driver in either dataset (`ids`: the race data's, then the lap data's not in it), with
the selection matrices from those to each dataset's drivers (`Sr`, `Sl`), and from the drivers
in both to each dataset's (`Pr`, `Pl`). `both` lists the drivers in both, as indices into `ids`.
"""
struct JointDrivers
    ids::Vector{String}
    race_idx::Vector{Int}       # each race driver's index in ids
    lap_idx::Vector{Int}        # each lap driver's index in ids
    both::Vector{Int}           # indices in ids of the drivers in both
    Sr::Matrix{Float64}         # n_race × n_ids
    Sl::Matrix{Float64}         # n_lap × n_ids
    Pr::Matrix{Float64}         # n_race × n_both
    Pl::Matrix{Float64}         # n_lap × n_both
end
function JointDrivers(g::GapData, d::LapData)
    ids = vcat(g.competitors, filter(c -> c ∉ Set(g.competitors), d.competitors))
    pos = Dict(c => i for (i, c) in enumerate(ids))
    ri, li = [pos[c] for c in g.competitors], [pos[c] for c in d.competitors]
    both = sort(intersect(ri, li))
    sel(idx, n) = (S = zeros(length(idx), n); for (r, i) in enumerate(idx); S[r, i] = 1.0; end; S)
    Sr, Sl = sel(ri, length(ids)), sel(li, length(ids))
    return JointDrivers(ids, ri, li, both, Sr, Sl, Sr[:, both], Sl[:, both])
end
Base.show(io::IO, jd::JointDrivers) = print(io, "JointDrivers($(length(jd.ids)) drivers: $(length(jd.race_idx)) race, ",
                                              "$(length(jd.lap_idx)) lap, $(length(jd.both)) in both)")

# the submodels with their drivers given: rebuilt from the stored arguments (a DynamicPPL
# Model can't be updated in place)
race_with_drivers(rm, a) = paceloss_effects(values(merge(rm.args, (; ext = merge(rm.args.ext, (; drivers = a)))))...)
lap_with_drivers(lm, a) = lap_effects(values(merge(lm.args, (; drv = a)))...)

"""
    joint_effects(rm, lm, jd::JointDrivers)

The joint model (see the file header). `rm` is a race model from `paceloss_effects(g; …)` and
`lm` a lap model from `lap_effects(d; …)`, each built as for its own fit, on seasons that
don't overlap; their own driver terms are replaced by the shared ones.

Priors: σ_a ~ half-N(0, 2s), with s the race data's gap scale (as the race model's σ_comp);
τ_r, τ_l ~ half-N(0, 0.5).
"""
@model function joint_effects(rm, lm, jd::JointDrivers)
    nb = length(jd.both)
    σ_a ~ truncated(Normal(0, 2rm.args.ps.s); lower = 0)
    z_a ~ StdNormalVec(length(jd.ids))
    τ_r ~ truncated(Normal(0, 0.5); lower = 0)
    u_r ~ StdNormalVec(nb)
    τ_l ~ truncated(Normal(0, 0.5); lower = 0)
    u_l ~ StdNormalVec(nb)
    a = σ_a .* z_a
    a_race = jd.Sr * a .+ jd.Pr * (τ_r .* u_r)
    a_lap = jd.Sl * a .+ jd.Pl * (τ_l .* u_l)
    race ~ to_submodel(race_with_drivers(rm, a_race))
    lap ~ to_submodel(lap_with_drivers(lm, a_lap))
    return (; a, a_race, a_lap)
end

# a chain's draw `i` as a NamedTuple of its parameters (vectors as vectors)
function draw_params(chain, i::Int = size(chain[:σ], 1), c::Int = 1)
    ps = Turing.FlexiChains.parameters(chain)
    return NamedTuple(Symbol(string(vn)) => chain[vn][i, c] for vn in ps)
end

"""
    joint_init(race_draw, lap_draw, jd::JointDrivers)

Starting values for `joint_effects` from a race draw and a lap draw (NamedTuples of each
model's parameters: a fit's draw via `draw_params`, or `paceloss_init_values`).
- **Race drivers:** the race drivers' effects become their shared ability, with `σ_a` the race
  σ_comp.
- **Lap-only drivers:** their lap effect.
- **Drivers in both:** the lap deviation makes a + δ_lap the lap draw's own effect, with τ_l
  its spread.
"""
joint_init(race_draw::NamedTuple, lap_draw::NamedTuple, jd::JointDrivers) =
    InitFromParams(joint_init_values(race_draw, lap_draw, jd))
"The values behind `joint_init`, as a nested NamedTuple."
function joint_init_values(race_draw::NamedTuple, lap_draw::NamedTuple, jd::JointDrivers)
    σ_a = race_draw.σ_comp
    ar = σ_a .* (race_draw.z_comp .- mean(race_draw.z_comp))
    al = lap_draw.a_comp .- mean(lap_draw.a_comp)
    a = zeros(length(jd.ids))
    a[jd.lap_idx] .= al
    a[jd.race_idx] .= ar
    lap_of = Dict(i => r for (r, i) in enumerate(jd.lap_idx))
    dl = [al[lap_of[i]] - a[i] for i in jd.both]
    τ_l = isempty(dl) || length(dl) < 2 ? 0.1 : max(std(dl), 0.02)
    race = NamedTuple(k => v for (k, v) in pairs(race_draw) if k ∉ (:σ_comp, :z_comp, :a_comp))
    lap = NamedTuple(k => v for (k, v) in pairs(lap_draw) if k ∉ (:σ_comp, :a_comp))
    return (; σ_a, z_a = a ./ σ_a, τ_r = 0.1, u_r = zeros(length(jd.both)), τ_l, u_l = dl ./ τ_l, race, lap)
end
