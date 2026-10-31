# The full-order data of the benchmark: linear Landau damping at the five amplitudes of the sweep,
# nonlinear Landau damping and the two-stream instability, on 64 × 128 (smoke: 32 × 64). Run from
# the repository root:
#
#   julia --project scripts/full_order.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It writes one HDF5 file per run to <d>, checks the linear rates against the dispersion
# relation, and writes the cost per step to <r>/full_order_cost.toml. It throws when a run breaks
# an invariant or a rate check fails.

using VlasovReductionBenchmark: run_full_order

const USAGE = "usage: julia --project scripts/full_order.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The keyword arguments of `run_full_order` from the command line `args`.
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

main(args) = run_full_order(; parse_options(args)...)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
