# The input channels of the multi-modal network: the reconstruction error of the weight-tied
# network on the five datasets of the reconstruction probes, in sample. Through the grid
# channel, the frontend error, the Stage I error and the trained error at every split
# (m_x, m_v) of n ∈ {4, 8, 16, 32}; through the DLRA channel, the truncated SVD of each snapshot
# at the input ranks {4, 8, 16} encoded into the trained core; and through a network of both
# channels trained on both, without and with the cross terms that train its translations, each
# channel and each translation between them. One seed, 2000 training steps, on 64 × 128 (smoke:
# 32 × 64, n ∈ {4, 8}, input rank 4, long runs to t = 40, 100 steps). Run from the repository
# root:
#
#   julia --project scripts/channels.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It reads the long runs from <d>, or runs and writes them there, and writes one row per
# dataset, n and split to <r>/channel_splits.csv, per dataset, n and input rank to
# <r>/channel_dlra.csv, per dataset, n and cross to <r>/channel_joint.csv, and the page of the
# three to <r>/channels.md. A training whose loss becomes non-finite or does not fall is a row with
# its status; the script goes on and counts it.

using VlasovReductionBenchmark: run_channels

const USAGE = "usage: julia --project scripts/channels.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The keyword arguments of `run_channels` from the command line `args`.
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

main(args) = run_channels(; parse_options(args)...)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
