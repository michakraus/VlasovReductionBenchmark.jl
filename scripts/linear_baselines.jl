# The POD and PSD Galerkin baselines of the full-order runs of scripts/full_order.jl, at the latent
# dimensions n ∈ {4, 8, 16, 32} (smoke: {4, 8}). Run from the repository root, after
# scripts/full_order.jl with the same directories:
#
#   julia --project scripts/linear_baselines.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It reads the run files from <d> and <r>/full_order_cost.toml, and writes one row per method,
# case and n, and per method, n and test amplitude of the sweep, to <r>/linear_baselines.csv, with
# the online cost of each reduced model, measured in a fresh process, in
# <r>/linear_baselines_cost.toml. A reduced run that stops is a row with its status; the script
# throws when the POD optimality check fails.

using VlasovReductionBenchmark: run_linear_baselines

const USAGE = "usage: julia --project scripts/linear_baselines.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The keyword arguments of `run_linear_baselines` from the command line `args`.
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

main(args) = run_linear_baselines(; parse_options(args)...)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
