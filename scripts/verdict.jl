# The verdict page of the benchmark, from the result files of the full-mode runs of the other
# scripts. Run from the repository root:
#
#   julia --project scripts/verdict.jl --results-dir <r>... --out <file>
#
# Each <r> is a results directory whose path names the commit that wrote it, in a component that
# ends in -<commit>. Together they hold linear_baselines.csv, autoencoders.csv and dlra.csv, each
# with its cost file and a full_order_cost.toml beside it, autoencoders_hamiltonian.toml, and
# particle_frontends.csv with particle_pod_modes.csv and particle_cost.toml. The script writes
# one Markdown page to <file> and prints its last line, the answer to whether the Vlasov half of
# the plan goes on. A missing input file stops it with a message that names the file.

using VlasovReductionBenchmark: write_verdict

const USAGE = "usage: julia --project scripts/verdict.jl --results-dir <r>... --out <file>"

# The keyword arguments of `write_verdict` from the command line `args`.
function parse_options(args)
    results_dirs, out = String[], nothing
    i = 1
    while i ≤ length(args)
        if args[i] == "--results-dir"
            i += 1
            start = i
            while i ≤ length(args) && !startswith(args[i], "--")
                push!(results_dirs, args[i])
                i += 1
            end
            i > start || throw(ArgumentError(USAGE))
        elseif args[i] == "--out" && i < length(args)
            out = args[i + 1]
            i += 2
        else
            throw(ArgumentError(USAGE))
        end
    end
    (isempty(results_dirs) || out === nothing) && throw(ArgumentError(USAGE))

    return (results_dirs = results_dirs, out = out)
end

main(args) = println(write_verdict(; parse_options(args)...))

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
