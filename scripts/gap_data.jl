# The race data of a `gap_fit.jl` mode, shared by the scripts that read its fits.
#
#   include("gap_data.jl"); res, g = gap_data(mode)
#
# `jolpica<first>-<last>[+partial|+timed]`: F1 from the Jolpica API (#55), all rows
# (lapped cars as intervals, or at their exact gap where they have their own time),
# `+partial` adding retirements as calibrated partial gaps, `+timed` keeping only
# lead-lap finishers. `wrc<first>-<last>`: WRC stage times in the top two tiers
# (#10). Returns the results table (with any partial rows) and its GapData.
# `data_arg` and `gap_chains` are for the scripts that read saved fits.

using Diomedes, DataFrames, Serialization

# the data of the published race fit
const DEFAULT_GAP_DATA = "jolpica1950-2025+partial"

"""
    data_arg(args) -> (mode, rest)

The data mode given as `data=<mode>` anywhere in `args` (default
`DEFAULT_GAP_DATA`), and the other arguments.
"""
function data_arg(args)
    i = findfirst(a -> startswith(a, "data="), args)
    i === nothing && return DEFAULT_GAP_DATA, args
    return String(split(args[i], "="; limit = 2)[2]), args[setdiff(eachindex(args), i)]
end

"""
    gap_chains(mode, model) -> FlexiChain

The saved chains of `gap_fit.jl <mode> <model>` (output/gap_<mode>_<model>_chain<k>.jls).
"""
function gap_chains(mode, model)
    return reduce(hcat, gap_chain_list(mode, model))
end

"""
    gap_chain_list(mode, model) -> Vector

Each chain of `gap_fit.jl <mode> <model>` on its own: a whole-chain file
(output/gap_<mode>_<model>_chain<k>.jls) or, for fits run in blocks (`block=`, #137),
its saved blocks concatenated, trimmed to the shortest chain so they can be joined.
"""
function gap_chain_list(mode, model)
    fit = "gap_$(mode)_$(model)"
    files = filter(isfile, ["output/$(fit)_chain$(k).jls" for k in 1:8])
    isempty(files) || return rehash_chain!.(deserialize.(files))
    chs = []
    for k in 1:8
        bs = sort(filter(f -> occursin(Regex("^\\Q$(fit)_chain$(k)_block\\E\\d+\\.jls\$"), f), readdir("output")),
                  by = f -> parse(Int, match(r"_block(\d+)\.jls$", f)[1]))
        isempty(bs) || push!(chs, reduce(vcat, [Diomedes.block_chain(joinpath("output", f)) for f in bs]))
    end
    isempty(chs) && error("no saved chains or blocks for $fit")
    n = minimum(size(c[:σ], 1) for c in chs)
    return [c[iter = 1:n] for c in chs]
end

function gap_data(mode; io = stdout)
    wrc = match(r"^wrc(\d{4})-(\d{4})$", mode)
    jol = match(r"^jolpica(\d{4})-(\d{4})(\+partial|\+timed)?$", mode)
    if jol !== nothing
        # F1 from the Jolpica API (responses cached by scripts/fetch_jolpica.jl), all
        # rows; lapped cars with their own time (2023 on) enter at their exact gap
        yrs = parse(Int, jol[1]):parse(Int, jol[2])
        res = fetch_results(JolpicaF1(), yrs)
        if jol[3] == "+timed"
            g = prepare_gaps(res; include_lapped = false)
            return res, g
        end
        if jol[3] == "+partial"
            # retirements as calibrated partial gaps (#13), taken 3 laps before the car
            # stopped (leaving out laps it may have limped through), from Jolpica's lap
            # times (1996 on; cached by fetch_jolpica.jl)
            cum = cumulative_laps(fetch_laps(JolpicaF1(), filter(>=(1996), yrs)))
            pg = partial_gaps(res, cum; min_frac = 0.25, trim = 3)
            res = with_partial_rows(res, pg)
            println(io, "partial-gap rows (retirements): $(nrow(pg))"); flush(io)
        end
        g = prepare_gaps(res)
        println(io, "lapped cars at exact gaps (own time): $(count(g.rows.lapped_exact))"); flush(io)
    elseif wrc !== nothing
        res = wrc_top_tiers(fetch_results(WRCTiming(), parse(Int, wrc[1]):parse(Int, wrc[2])))
        g = prepare_gaps(res; machine_key = wrc_machine_key)
    else
        throw(ArgumentError("unknown data mode $mode: jolpica<a>-<b>[+partial|+timed] or wrc<a>-<b> " *
                            "(the Ergast-dump modes timed and all are now jolpica1950-2019+timed and jolpica1950-2019)"))
    end
    return res, g
end
