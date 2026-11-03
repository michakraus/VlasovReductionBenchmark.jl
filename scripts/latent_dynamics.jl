# The latent dynamics of the reductions: PSD, PSD without the x-average, dual PSD and the
# weight-tied network, each built on the snapshots with t ≤ 50 of the three long runs to t = 100,
# at n ∈ {4, 8, 16, 32}. Each has a G-SympNet per seed (42, 43, 44; 2000 epochs) trained on its
# latent trajectory and rolled out over the whole run, and, except dual PSD, its Galerkin model
# (one per seed for the network). Each row holds the errors in sample and in extrapolation and
# the drifts of the energy, L¹, L² and mass. On 64 × 128 (smoke: 32 × 64, the long runs to
# t = 40, n = 4, seed 42, 20 epochs, 100 training steps of the network). Run from the
# repository root:
#
#   julia --project scripts/latent_dynamics.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It reads the long runs from <d>, or runs and writes them there, and writes one row per run, n,
# reduction, dynamics and seed to <r>/latent_dynamics.csv and the page of it to
# <r>/latent_dynamics.md. A rollout that diverges, a Galerkin run that fails, a training that
# fails and a dual PSD whose pairing block is singular or round-off are rows with their status;
# the script goes on.

using VlasovReductionBenchmark: LATENT_COLUMNS, latent_comparison_mode,
                                latent_comparison_rows, write_csv, write_latent_page

const USAGE = "usage: julia --project scripts/latent_dynamics.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The directories and the mode from the command line `args`.
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

function main(args)
    options = parse_options(args)
    mode = latent_comparison_mode(options.smoke)
    println("latent dynamics on $(mode.grid[1]) × $(mode.grid[2])",
        options.smoke ? " (smoke)" : "", ", n ∈ $(mode.ns), seeds $(mode.seeds)")
    flush(stdout)
    rows = latent_comparison_rows(options.runs_dir; mode)
    mkpath(options.results_dir)
    path = write_csv(joinpath(options.results_dir, "latent_dynamics.csv"), LATENT_COLUMNS, rows)
    write_latent_page(options.results_dir)
    println("latent_dynamics.csv: $(length(rows)) rows, ",
        count(r -> r["status"] != "ok", rows), " not ok, written to $path")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
