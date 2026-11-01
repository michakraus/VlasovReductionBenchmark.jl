# The particle half of the benchmark: a PIC run of nonlinear Landau damping with 4096 particles
# over [0, 40] (smoke: 512 over [0, 10]), the POD mode counts of its labelled and sorted
# snapshots, and the particle frontends of SymmetricSymplecticAutoencoder trained on 256 of its
# particles (smoke: 64), beside the POD of the same particles. Run from the repository root:
#
#   julia --project scripts/particles.jl --runs-dir <d> --results-dir <r> [--smoke]
#
# It writes the run to <d>/particles_nonlinear_landau.h5, and particle_pod_modes.csv,
# particle_frontends.csv and particle_cost.toml to <r>. It throws when the energy of the run
# drifts by more than 1e-3 relative.

using VlasovReductionBenchmark: run_particles

const USAGE = "usage: julia --project scripts/particles.jl --runs-dir <d> --results-dir <r> [--smoke]"

# The keyword arguments of `run_particles` from the command line `args`.
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

main(args) = run_particles(; parse_options(args)...)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
