# Writes test/fixtures/lap_golden.tsv: the standalone lap model's log density and gradient on the
# fixture in test/lap_golden.jl, from whichever Diomedes is active. Run it from the code you want
# to pin:  julia --project=<checkout> scripts/lap_golden.jl <out.tsv>
using Diomedes, Turing, ReverseDiff
include(joinpath(@__DIR__, "..", "test", "lap_golden.jl"))
vals = lap_golden_values()
write_lap_golden(ARGS[1], vals)
for (name, n, v, g) in vals
    println(rpad(name, 20), " dim ", n, "  logp ", v, "  |grad| ", sqrt(sum(abs2, g)))
end
