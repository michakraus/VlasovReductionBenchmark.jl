# The reconstruction probes: the reconstruction error of POD, PSD, the untied and the
# weight-tied autoencoder and the rank-r truncated SVD, with no reduced dynamics, on five
# datasets: the passive advection of a bump over half and over one recurrence time of the grid,
# and the long runs of the two-stream instability (the v²-scaled Maxwellian and two beams) and of
# nonlinear Landau damping to t = 100. Each is measured in sample and in extrapolation in time
# (train on t ≤ T/2, test on the rest), at n ∈ {4, 8, 16, 32}, with three seeds of the untied
# network, one of the weight-tied network and 2000 training steps, on 64 × 128 (smoke: 32 × 64,
# n ∈ {4, 8}, long runs to t = 40, one seed, 100 steps). Run from the repository root:
#
#   julia --project scripts/reconstruction.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It writes the long runs to <d>, one row per dataset, split, method, n and seed to
# <r>/reconstruction.csv, the singular values and the mode counts of each dataset to
# <r>/singular_values.csv, and the page of both to <r>/reconstruction.md. A training whose loss
# becomes non-finite or does not fall is a row with its status; the script goes on and counts it.

using VlasovReductionBenchmark: run_reconstruction

const USAGE = "usage: julia --project scripts/reconstruction.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The keyword arguments of `run_reconstruction` from the command line `args`.
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

main(args) = run_reconstruction(; parse_options(args)...)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
