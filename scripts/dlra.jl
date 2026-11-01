# The DLRA solver at the ranks r ∈ {4, 8, 16, 32} (smoke: {4, 8}) on the full-order runs of
# scripts/full_order.jl. Run from the repository root, after scripts/full_order.jl with the same
# directories:
#
#   julia --project scripts/dlra.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It first checks the solver: at full rank in x it agrees with its full-rank solver, and the
# full-rank solver and the run of rank 16 meet the rate check of linear Landau damping; a check
# that fails stops the script. It then reads the run files from <d> and <r>/full_order_cost.toml,
# and writes one row per case and rank, and per test amplitude of the sweep and rank, to
# <r>/dlra.csv, with the online cost per step, measured in a fresh process, in <r>/dlra_cost.toml.
# A DLRA run that blows up is a row with its status.

using VlasovReductionBenchmark: run_dlra_benchmark

const USAGE = "usage: julia --project scripts/dlra.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The keyword arguments of `run_dlra_benchmark` from the command line `args`.
function parse_options(args)
    runs_dir = results_dir = nothing
    smoke = false
    i = 1
    while i ≤ length(args)
        if args[i] == "--smoke"
            smoke = true
            i += 1
        elseif args[i] in ("--runs-dir", "--results-dir") && i < length(args)
            args[i] == "--runs-dir" ? (runs_dir = args[i + 1]) : (results_dir = args[i + 1])
            i += 2
        else
            throw(ArgumentError(USAGE))
        end
    end
    (runs_dir === nothing || results_dir === nothing) && throw(ArgumentError(USAGE))

    return (runs_dir = runs_dir, results_dir = results_dir, smoke = smoke)
end

main(args) = run_dlra_benchmark(; parse_options(args)...)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
