# The joint race + lap model (#128): one driver ability shared by the race model
# (`paceloss_effects`) and the lap model (`lap_effects`), with model-specific deviations.
#
#     a_race = a + δ_race,      a_lap = λ·a + δ_lap,
#
# a ~ N(0, σ_a) for every driver in the race data (race units: % of race time, average-season);
# δ_race ~ N(0, τ_r) and δ_lap ~ N(0, τ_l) for the drivers in both. λ maps race units to lap
# time. Clean laps leave out much of what separates drivers in races, so λ is well under 1
# (lap effects run at about 0.45× the race model's). A driver only in the race data keeps
# a_race = a, as in the race model alone. Each model keeps every other term of its own, so
# the cars, walks and conditions adjust to the pooled drivers. Both submodels are the usual
# models with their drivers given from outside (`ext.drivers`, `drv`), and their parameters
# carry the prefixes `race.` and `lap.`.

"""
    JointDrivers(g::GapData, d::LapData)

The lap data's drivers as indices into the race data's (`lap_in_race`), by driver id.
Every lap driver must be in the race data.
"""
struct JointDrivers
    lap_in_race::Vector{Int}
    n_race::Int
end
function JointDrivers(g::GapData, d::LapData)
    idx = Dict(c => i for (i, c) in enumerate(g.competitors))
    miss = filter(c -> !haskey(idx, c), d.competitors)
    isempty(miss) || throw(ArgumentError("lap drivers not in the race data: $(join(miss, ", "))"))
    return JointDrivers([idx[c] for c in d.competitors], length(g.competitors))
end

# each lap driver's race deviation placed at its race index (an n_race × n_lap 0/1 matrix product,
# so the tracked arrays stay arrays under ReverseDiff)
race_deviation_matrix(jd::JointDrivers) =
    (P = zeros(jd.n_race, length(jd.lap_in_race)); for (j, i) in enumerate(jd.lap_in_race); P[i, j] = 1.0; end; P)

# the submodels with their drivers given: rebuilt from the stored arguments (a DynamicPPL
# Model can't be updated in place)
race_with_drivers(rm, a) = paceloss_effects(values(merge(rm.args, (; ext = merge(rm.args.ext, (; drivers = a)))))...)
lap_with_drivers(lm, a) = lap_effects(values(merge(lm.args, (; drv = a)))...)

"""
    joint_effects(rm, lm, jd::JointDrivers)

The joint model (see the file header). `rm` is a race model from `paceloss_effects(g; …)` and
`lm` a lap model from `lap_effects(d; …)`, each built as for its own fit; their own driver
terms are replaced by the shared ones. Priors: σ_a ~ half-N(0, 2s), with s the race data's
gap scale (as the race model's σ_comp); τ_r, τ_l ~ half-N(0, 0.5); λ ~ N(0.5, 0.5) truncated at 0.
"""
@model function joint_effects(rm, lm, jd::JointDrivers, P::Matrix{Float64})
    nl = length(jd.lap_in_race)
    σ_a ~ truncated(Normal(0, 2rm.args.ps.s); lower = 0)
    z_a ~ StdNormalVec(jd.n_race)
    τ_r ~ truncated(Normal(0, 0.5); lower = 0)
    u_r ~ StdNormalVec(nl)
    λ ~ truncated(Normal(0.5, 0.5); lower = 0)
    τ_l ~ truncated(Normal(0, 0.5); lower = 0)
    u_l ~ StdNormalVec(nl)
    a = σ_a .* z_a
    a_race = a .+ P * (τ_r .* u_r)
    a_lap = λ .* (P' * a) .+ τ_l .* u_l
    race ~ to_submodel(race_with_drivers(rm, a_race))
    lap ~ to_submodel(lap_with_drivers(lm, a_lap))
    return (; a, a_race, a_lap)
end
joint_effects(rm, lm, jd::JointDrivers) = joint_effects(rm, lm, jd, race_deviation_matrix(jd))

# a chain's draw `i` as a NamedTuple of its parameters (vectors as vectors)
function draw_params(chain, i::Int = size(chain[:σ], 1), c::Int = 1)
    ps = Turing.FlexiChains.parameters(chain)
    return NamedTuple(Symbol(string(vn)) => chain[vn][i, c] for vn in ps)
end

"""
    joint_init(race_draw, lap_draw, jd::JointDrivers; λ = 0.45)

Starting values for `joint_effects` from a draw of a race fit and one of a lap fit of the
same models (`draw_params`). The race drivers' effects become the shared ability.
- **Race drivers:** with `σ_a` = the race σ_comp, the race model's drivers are unchanged at
  the start.
- **Lap drivers:** their deviations are set so that λ·a + δ_lap is the lap fit's own
  effects, with τ_l their spread.
"""
joint_init(race_draw::NamedTuple, lap_draw::NamedTuple, jd::JointDrivers; λ::Real = 0.45) =
    InitFromParams(joint_init_values(race_draw, lap_draw, jd; λ))
"The values behind `joint_init`, as a nested NamedTuple."
function joint_init_values(race_draw::NamedTuple, lap_draw::NamedTuple, jd::JointDrivers; λ::Real = 0.45)
    σ_a = race_draw.σ_comp
    z = race_draw.z_comp .- mean(race_draw.z_comp)
    a = σ_a .* z
    al = lap_draw.a_comp .- mean(lap_draw.a_comp)
    dl = al .- λ .* a[jd.lap_in_race]
    dl .-= mean(dl)
    τ_l = max(std(dl), 0.02)
    race = NamedTuple(k => v for (k, v) in pairs(race_draw) if k ∉ (:σ_comp, :z_comp, :a_comp))
    lap = NamedTuple(k => v for (k, v) in pairs(lap_draw) if k ∉ (:σ_comp, :a_comp))
    return (; σ_a, z_a = z, τ_r = 0.1, u_r = zeros(length(jd.lap_in_race)), λ = Float64(λ), τ_l, u_l = dl ./ τ_l, race, lap)
end
