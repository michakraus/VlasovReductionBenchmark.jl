# The untied and the weight-tied symplectic autoencoder of the full-order runs of
# scripts/full_order.jl, at the latent dimensions n ∈ {4, 8, 16, 32}, three seeds and 2000 training
# steps (smoke: n ∈ {4, 8}, one seed, 100 steps). Run from the repository root, after
# scripts/full_order.jl with the same directories:
#
#   julia --project scripts/autoencoders.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It reads the run files from <d> and <r>/full_order_cost.toml, and writes one row per network,
# case, n and seed, and per network, n, seed and test amplitude of the sweep, to
# <r>/autoencoders.csv, with the online cost of each reduced model, measured in a fresh process,
# in <r>/autoencoders_cost.toml, and the comparison of the canonical flow of the energy with the
# Vlasov field in <r>/autoencoders_hamiltonian.toml. A training whose loss becomes non-finite or
# does not fall, and a reduced run that stops, are rows with their status; the script goes on and
# counts them.

using VlasovReductionBenchmark: run_autoencoders

const USAGE = "usage: julia --project scripts/autoencoders.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The keyword arguments of `run_autoencoders` from the command line `args`.
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

main(args) = run_autoencoders(; parse_options(args)...)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
