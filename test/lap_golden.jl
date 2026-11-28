# Golden values for the standalone lap model's log density and gradient (#128).
#
# The fixture is deterministic (no RNG): 6 drivers in 6 cars over 2 seasons × 2 races of 24 laps,
# with a pit stop each, a slow lap now and then, grid positions and birth dates. `lap_golden_values`
# runs on both the original code and the code that folds the incident term into the lap likelihood,
# so test/fixtures/lap_golden.tsv (written by scripts/lap_golden.jl from the code before #128's
# integration) pins the standalone model's value and gradient.

using DataFrames, Dates

function lap_golden_data()
    drivers = ["d$i" for i in 1:6]
    rows = DataFrame(series = String[], season = Int[], event_id = String[], competitor_id = String[],
                     machine_id = String[], grid = Int[], competitor_birth = Date[], event_date = Date[])
    lt = DataFrame(series = String[], season = Int[], event_id = String[], competitor_id = String[], lap = Int[],
                   time_ms = Float64[], position = Union{Missing,Int}[])
    for (si, s) in enumerate((2021, 2022)), r in 1:2
        e = "$s-0$r"
        for (i, c) in enumerate(drivers)
            g = mod1(i + 2r + si, 6)
            push!(rows, ("f1", s, e, c, "m$(mod1(i + si - 1, 6))", g, Date(1985 + 2i, i, 10), Date(s, 3 + 4r, 1)))
            pit = 8 + mod(3i + r + si, 9)
            for l in 1:24
                x = 90_000 + 400i + 150 * sin(1.3i + 0.7l + r) + 60 * (l - pit) * (l > pit) +
                    35 * l * (l <= pit) - 800 * (si - 1) + 300r
                l == 1 && (x += 6_000 + 250g)
                l == pit && (x += 5_000)
                l == pit + 1 && (x += 18_000)
                mod(7i + 5l + 3r + si, 37) == 0 && (x += 2_500 + 40 * mod(i * l, 13))   # a mistake
                push!(lt, ("f1", s, e, c, l, x, missing))
            end
        end
    end
    return lt, rows
end

lap_golden_x(n) = [0.3 * sin(0.37k + 0.2) + 0.05 * cos(1.1k) for k in 1:n]

# lap conditions: the second race of each season wet from lap 10 for drivers 1–3, rain on lap 12
function lap_golden_conditions(d, lt)
    cond = DataFrame(event_id = String[], driver_number = Int[], lap = Int[], compound = Union{Missing,String}[],
                     lap_time_ms = Union{Missing,Float64}[], track_temp = Union{Missing,Float64}[],
                     rainfall = Union{Missing,Float64}[])
    nums = Dict{Tuple{String,Int},String}()
    for r in eachrow(lt)
        i = parse(Int, r.competitor_id[2:end])
        nums[(r.event_id, 10 + i)] = r.competitor_id
        wet = endswith(r.event_id, "02") && i <= 3 && r.lap >= 10
        comp = wet ? (i == 1 ? "WET" : "INTERMEDIATE") : (r.lap > 12 ? "HARD" : "MEDIUM")
        push!(cond, (r.event_id, 10 + i, r.lap, comp, r.time_ms, 30.0 + 3 * sin(r.lap + i), r.lap == 12 ? 1.0 : 0.0))
    end
    return LapConditions(d, cond, nums; wetdev = true, wetcar = true, wetmin = 1, wetglobal = true, wetrace = true,
                         racedeg = true, racedegc = true, tyreage = true)
end

# the variants: (name, model flags, context)
function lap_golden_variants(d, lt)
    c = LapContext(d, lt; traffic = true)
    fp = [0.4 * sin(0.9j) for j in eachindex(d.y)]
    cs = LapContext(c.tbin, c.tnames, fp, true, true, true)
    return [("plain", (;), LapContext()),
            ("big_dinc", (; big = true, dinc = true), LapContext()),
            ("starts_slopes_age", (; starts = true, slopes = true, age = true, tyre_mgmt = true, lead = true), LapContext()),
            ("traffic", (; dinc = true, starts = true), c),
            ("short_fpace", (; starts = true, slopes = true), cs),
            ("short_cond", (; starts = true, slopes = true, dinc = true, cond = lap_golden_conditions(d, lt)), cs),
            ("short_cond_big", (; starts = true, dinc = true, big = true, cond = lap_golden_conditions(d, lt)), cs)]
end

# (name, dimension, log density, gradient) for each variant
function lap_golden_values()
    lt, rows = lap_golden_data()
    d = prepare_laps(lt, rows; pits = :inferred)
    out = Tuple{String,Int,Float64,Vector{Float64}}[]
    for (name, flags, cx) in lap_golden_variants(d, lt)
        m = lap_effects(d; flags..., ctx = cx)
        f = Turing.DynamicPPL.LogDensityFunction(m; adtype = Diomedes.gap_sampler().adtype)
        n = Turing.LogDensityProblems.dimension(f)
        x = lap_golden_x(n)
        for (k, r) in pairs(NamedTuple(getfield(f, :_varname_ranges)))     # scales are unlinked: keep them positive
            nm = string(k)
            (startswith(nm, "σ") || startswith(nm, "τ") || k in (:δ_λ2, :δ_own, :δ_reg)) && (x[r.range] .= 0.3 .+ abs.(x[r.range]))
        end
        v, g = Turing.LogDensityProblems.logdensity_and_gradient(f, x)
        push!(out, (name, n, v, collect(g)))
    end
    return out
end

function write_lap_golden(path, vals)
    open(path, "w") do io
        for (name, n, v, g) in vals
            println(io, join((name, n, repr(v), join(repr.(g), ",")), '\t'))
        end
    end
end

function read_lap_golden(path)
    map(readlines(path)) do line
        name, n, v, g = split(line, '\t')
        (String(name), parse(Int, n), parse(Float64, v), parse.(Float64, split(g, ',')))
    end
end
