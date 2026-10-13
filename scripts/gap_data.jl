# The race data of a `gap_fit.jl` mode, shared by the scripts that read its fits.
#
#   include("gap_data.jl"); res, g = gap_data(mode; data_dir)
#
# `timed` / `all`: the local Ergast dump (1950–2019), lead-lap finishers only / plus
# lapped cars as intervals. `jolpica<first>-<last>[+partial]`: all rows from the
# Jolpica API (#55), optionally with retirements as calibrated partial gaps.
# `wrc<first>-<last>`: WRC stage times in the top two tiers (#10). Returns the
# results table (with any partial rows) and its GapData.

using Diomedes, DataFrames

function gap_data(mode; data_dir = get(ENV, "DIOMEDES_ERGAST_DIR", joinpath(@__DIR__, "..", "data")), io = stdout)
    wrc = match(r"^wrc(\d{4})-(\d{4})$", mode)
    jol = match(r"^jolpica(\d{4})-(\d{4})(\+partial)?$", mode)
    if jol !== nothing
        # F1 from the Jolpica API (responses cached by scripts/fetch_jolpica.jl), all
        # rows; lapped cars with their own time (2023 on) enter at their exact gap
        yrs = parse(Int, jol[1]):parse(Int, jol[2])
        res = fetch_results(JolpicaF1(), yrs)
        if jol[3] !== nothing
            # retirements as calibrated partial gaps (#13), taken 3 laps before the car
            # stopped (leaving out laps it may have limped through), from lap times:
            # the Ergast dump to 2019, Jolpica after (cached by fetch_jolpica.jl)
            cum = merge(ergast_cumulative_laps(data_dir), cumulative_laps(fetch_laps(JolpicaF1(), filter(>(2019), yrs))))
            pg = partial_gaps(res, cum; min_frac = 0.25, trim = 3)
            res = with_partial_rows(res, pg)
            println(io, "partial-gap rows (retirements): $(nrow(pg))"); flush(io)
        end
        g = prepare_gaps(res)
        println(io, "lapped cars at exact gaps (own time): $(count(g.rows.lapped_exact))"); flush(io)
    elseif wrc !== nothing
        res = wrc_top_tiers(fetch_results(WRCTiming(), parse(Int, wrc[1]):parse(Int, wrc[2])))
        g = prepare_gaps(res; machine_key = wrc_machine_key)
    elseif mode in ("timed", "all")
        res = fetch_results(ErgastCSV(data_dir), 1950:2100)
        g = prepare_gaps(res; include_lapped = mode == "all")
    else
        throw(ArgumentError("unknown data mode $mode: timed, all, jolpica<a>-<b>[+partial] or wrc<a>-<b>"))
    end
    return res, g
end
