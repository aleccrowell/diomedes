# The directions on which a gap / pace + loss fit's chains disagree most (#132),
# read-only on its saved chains.
#
#   julia --project scripts/slow_directions.jl <data> <model> [n=4] [groups=a,b,…] [k=3] [top=8] [nperm=100]
#
# <data> and <model> as in scripts/gap_fit.jl; reads output/gap_<data>_<model>_chain<k>.jls.
# Over the identified quantities (`identified_draws`: drivers, cars, walks, slopes,
# races, scalars, …; `groups=` picks some), standardised by their posterior sd, it
# reports the leading slow directions (`slow_directions`): each one's share of the
# between-chain spread, its R-hat and ESS, and the quantities that load on it most.
# Runs once over all chosen groups together, then within each group.
using Diomedes, Serialization, Printf

opt(key, default) = (i = findfirst(a -> startswith(a, key * "="), ARGS); i === nothing ? default : split(ARGS[i], "="; limit = 2)[2])
args = filter(a -> !occursin('=', a), ARGS)
mode, model = args[1], args[2]
n = parse(Int, opt("n", "4")); k = parse(Int, opt("k", "3")); top = parse(Int, opt("top", "8"))
include("gap_data.jl")
res, g = gap_data(mode)
chain = reduce(hcat, [rehash_chain!(deserialize("output/gap_$(mode)_$(model)_chain$(c).jls")) for c in 1:n])
cols, label = identified_draws(chain, g, model)
groups = opt("groups", nothing) === nothing ? sort(collect(keys(cols))) : Symbol.(split(opt("groups", ""), ","))

function report(title, A, labels)
    println("\n== $title ($(size(A, 3)) quantities)")
    for (j, d) in enumerate(slow_directions(A, labels; k, top))
        @printf("direction %d: %.0f %% of the between-chain spread, R-hat %.3f, ESS %.0f\n", j, 100d.share, d.rhat, d.ess)
        println("   ", join([@sprintf("%s %+.2f", l, w) for (l, w) in d.loadings], ", "))
    end
end
report("all of " * join(groups, ", "), cat((cols[gr] for gr in groups)...; dims = 3),
       vcat(([string(gr, ": ", label(gr, i)) for i in axes(cols[gr], 3)] for gr in groups)...))
for gr in groups
    size(cols[gr], 3) > 1 && report(string(gr), cols[gr], [label(gr, i) for i in axes(cols[gr], 3)])
end

println("\n== slow trends over all of " * join(groups, ", ") * " (permutation test, LIMBR-style)")
A = cat((cols[gr] for gr in groups)...; dims = 3)
labs = vcat(([string(gr, ": ", label(gr, i)) for i in axes(cols[gr], 3)] for gr in groups)...)
tr = slow_trends(A, labs; nperm = parse(Int, opt("nperm", "100")))
isempty(tr) && println("no significant trends")
for (j, t) in enumerate(tr)
    @printf("trend %d: p %.2f, %.1f %% of the segment variance, %d members; between chains %.0f %% (p %.2f), lag-1 %.2f (p %.2f)%s\n",
            j, t.p, 100t.share, length(t.members), 100t.between, t.p_between, t.lag1, t.p_drift,
            min(t.p_between, t.p_drift) < 0.05 ? "  SLOW" : "")
    println("   ", join([@sprintf("%s %+.2f", l, r) for (l, r) in t.members[1:min(top, end)]], ", "))
    grp = Dict{String,Int}(); for (l, _) in t.members; g = split(l, ":")[1]; grp[g] = get(grp, g, 0) + 1; end
    println("   by group: ", join(["$g $n" for (g, n) in sort(collect(grp); by = last, rev = true)], ", "))
end
