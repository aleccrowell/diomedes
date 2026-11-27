# The joint race + lap model (#128): one driver ability shared by the race model
# (`paceloss_effects`) and the lap model (`lap_effects`), with model-specific deviations.
#
#     a_race = a + δ_race,      a_lap = a + δ_lap,
#
# - **a:** for every driver in either dataset, a ~ N(0, σ_a) (centred), in race units (% of race
#   time, average-season).
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

"""
    joint_ages(rm, lm, jd::JointDrivers) -> (rm, lm, lapage)

The two submodels with their drivers' ages centred on one reference age per driver, for the
shared career slopes (#128).
- **Reference:** each driver's mean age, weighted by races, over their race results and their
  lap seasons.
- **Why:** each model alone centres on its own data (the race model on the driver's race
  results, the lap model on their laps), and a shared slope with two centres would give the
  level two meanings.
- **`lapage`:** each lap age bin as an index into the race model's age bins, so the lap part
  uses the race part's age curve.

Both models must have the age curve and career slopes on.
"""
function joint_ages(rm, lm, jd::JointDrivers)
    g, d = rm.args.g, lm.args.d
    cr = rm.args.career_rows
    (cr !== nothing && cr.slopes && rm.args.age_basis !== nothing) ||
        throw(ArgumentError("the joint model's race part needs the age curve and career slopes (age, slopes)"))
    (lm.args.slopes && lm.args.age) || throw(ArgumentError("the joint model's lap part needs the age curve and career slopes (age, slopes)"))
    rows = cr.rows
    n = length(jd.ids)
    s, w = zeros(n), zeros(n)
    yrs = [Float64(g.age_years[b]) for b in rows.bin]
    for k in eachindex(rows.comp)            # race results: one per race
        rows.known[k] > 0 || continue
        i = jd.race_idx[rows.comp[k]]
        s[i] += yrs[k]; w[i] += 1
    end
    nraces = zeros(length(d.competitors))    # lap seasons: weighted by the driver's races with laps
    for (c, r) in unique(zip(d.comp, d.race)); nraces[c] += 1; end
    for c in eachindex(d.competitors)
        isnan(d.comp_age[c]) && continue
        i = jd.lap_idx[c]
        s[i] += nraces[c] * d.comp_age[c]; w[i] += nraces[c]
    end
    ref = s ./ max.(w, 1)
    agec_r = [rows.known[k] * (yrs[k] - ref[jd.race_idx[rows.comp[k]]]) / 10 for k in eachindex(rows.comp)]
    rm2 = paceloss_effects(values(merge(rm.args, (; career_rows = (; rows = CareerRows(rows.bin, rows.known, rows.comp, agec_r),
                                                                    slopes = cr.slopes))))...)
    # the lap model's agec is (season age - its own centre) / 10: moved to the shared centre where the age is known
    agec_l = [d.abin[j] > 0 ? d.agec[j] + (d.comp_age[d.comp[j]] - ref[jd.lap_idx[d.comp[j]]]) / 10 : 0.0 for j in eachindex(d.y)]
    lm2 = lap_effects(values(merge(lm.args, (; d = @set(d.agec = agec_l))))...)
    pos = Dict(y => k for (k, y) in enumerate(g.age_years))
    miss = filter(y -> !haskey(pos, y), d.age_years)
    isempty(miss) || throw(ArgumentError("lap ages outside the race data's age bins: $(join(miss, ", "))"))
    return rm2, lm2, [pos[y] for y in d.age_years]
end

# the submodels with their driver terms given: rebuilt from the stored arguments (a DynamicPPL
# Model can't be updated in place)
race_given(rm, a, fage, sl) = paceloss_effects(values(merge(rm.args, (; ext = merge(rm.args.ext, (; drivers = a, fage, slopes = sl)))))...)
lap_given(lm, a, sl, fage) = lap_effects(values(merge(lm.args, (; drv = a, sl_ext = sl, fage_ext = fage)))...)

"""
    joint_effects(rm, lm, jd::JointDrivers)

The joint model (see the file header). `rm` is a race model from `paceloss_effects(g; …)` and
`lm` a lap model from `lap_effects(d; …)`, each built as for its own fit, on seasons that
don't overlap, both with the age curve and career slopes on.

Shared:
- **The driver's level**, with deviations for drivers in both. Priors: σ_a ~ half-N(0, 2s),
  with s the race data's gap scale (as the race model's σ_comp); τ_r, τ_l ~ half-N(0, 0.5).
- **The career slope**, one per driver, on a common age centre (`joint_ages`); prior as the race
  model's.
- **The age curve**, on the race data's age bins; prior as the race model's.

Their own versions of these terms are replaced by the shared ones.
"""
function joint_effects(rm, lm, jd::JointDrivers)
    rm2, lm2, lapage = joint_ages(rm, lm, jd)
    return joint_model(rm2, lm2, jd, lapage)
end

@model function joint_model(rm, lm, jd::JointDrivers, lapage::Vector{Int})
    nb = length(jd.both)
    ps = rm.args.ps
    σ_a ~ truncated(Normal(0, 2ps.s); lower = 0)
    # centred, as the lap model's drivers (#28): non-centred, a = σ·z ties σ to every well-measured z
    # (a funnel), and the joint smoke runs diverged on every iteration (#128). σ_a is pinned by every
    # driver, so the few-result drivers don't funnel it the other way.
    a ~ IsoNormalVec(0.0, σ_a, length(jd.ids))
    τ_r ~ truncated(Normal(0, 0.5); lower = 0)
    δ_r ~ IsoNormalVec(0.0, τ_r, nb)
    τ_l ~ truncated(Normal(0, 0.5); lower = 0)
    δ_l ~ IsoNormalVec(0.0, τ_l, nb)
    # shared age curve (the race data's bins) and career slopes (one per driver, centred across them)
    τ_age ~ truncated(Normal(0, 0.2ps.s); lower = 0)
    e_age ~ StdNormalVec(length(rm.args.g.age_years) - 1)
    fage = age_curve(τ_age, e_age, rm.args.age_basis)
    τ_slope ~ truncated(Normal(0, 0.5ps.s); lower = 0)
    u_slope ~ StdNormalVec(length(jd.ids))
    sl = career_slopes(τ_slope, u_slope)
    a_race = jd.Sr * a .+ jd.Pr * δ_r
    a_lap = jd.Sl * a .+ jd.Pl * δ_l
    race ~ to_submodel(race_given(rm, a_race, fage, jd.Sr * sl))
    lap ~ to_submodel(lap_given(lm, a_lap, jd.Sl * sl, fage[lapage]))
    return (; a, a_race, a_lap, fage, sl)
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
- **Age curve and career slopes:** from the race draw, with lap-only drivers' slopes at 0.
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
    shared = (:σ_comp, :z_comp, :a_comp, :τ_slope, :u_slope, :τ_age, :e_age)
    race = NamedTuple(k => v for (k, v) in pairs(race_draw) if k ∉ shared)
    lap = NamedTuple(k => v for (k, v) in pairs(lap_draw) if k ∉ shared)
    # the age curve from the race draw; each driver's slope from the race draw, 0 for lap-only drivers
    u = zeros(length(jd.ids))
    u[jd.race_idx] .= race_draw.u_slope .- mean(race_draw.u_slope)
    return (; σ_a, a, τ_r = 0.1, δ_r = zeros(length(jd.both)), τ_l, δ_l = dl, τ_age = race_draw.τ_age,
            e_age = race_draw.e_age, τ_slope = race_draw.τ_slope, u_slope = u, race, lap)
end

# name => range of each variable in a model's sampler (linked) vector
function varranges(model)
    DPPL = Turing.DynamicPPL
    vi = DPPL.link!!(DPPL.VarInfo(model), model)
    out, pos = Dict{String,UnitRange{Int}}(), 0
    for vn in keys(vi)
        n = length(DPPL.getindex_internal(vi, vn))
        out[string(vn)] = (pos + 1):(pos + n); pos += n
    end
    return out, pos
end

"""
    joint_warm_metric(jm, lm_std, lap_Minv; default = 1.0, other = 0.01) -> Vector

A diagonal inverse metric for the joint model `jm`, for warm-up to start from:
- **Lap part:** each `lap.` variable takes the lap fit's adapted variances (`lap_Minv`, in the
  layout of `lm_std`, the lap model as that fit had it), matched by name. Without them the
  lap part's posterior, concentrated by about 130k laps, forces a tiny step and maximum tree
  depth from the start (the #128 smoke run).
- **Other laps' terms** (not in the lap fit): `other`.
- **Race part and shared terms:** `default`, as a race fit's cold start.
"""
function joint_warm_metric(jm, lm_std, lap_Minv::AbstractVector; default::Real = 1.0, other::Real = 0.01)
    jr, n = varranges(jm)
    lr, nl = varranges(lm_std)
    nl == length(lap_Minv) || throw(DimensionMismatch("lap_Minv has $(length(lap_Minv)) entries, the lap model $nl"))
    M = fill(Float64(default), n)
    for (name, r) in jr
        startswith(name, "lap.") || continue
        k = name[5:end]
        if haskey(lr, k) && length(lr[k]) == length(r)
            M[r] .= lap_Minv[lr[k]]
        elseif Symbol(k) in OTHER_PARAMS
            M[r] .= other
        end
    end
    return M
end
