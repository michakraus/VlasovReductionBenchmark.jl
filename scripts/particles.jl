# The particle half of the benchmark: a PIC run of nonlinear Landau damping with 4096 particles
# over [0, 40] (smoke: 512 over [0, 10]), the POD mode counts of its labelled and sorted
# snapshots, and the particle frontends of SymmetricSymplecticAutoencoder trained on 256 of its
# particles (smoke: 64), beside the POD of the same particles, sorted in each snapshot and with
# their labels. Run from the repository root:
#
#   julia --project scripts/particles.jl --runs-dir <d> --results-dir <r> [--smoke] [--labelled-pod]
#
# It writes the run to <d>/particles_nonlinear_landau.h5, and particle_pod_modes.csv,
# particle_frontends.csv and particle_cost.toml to <r>. It throws when the energy of the run
# drifts by more than 1e-3 relative. With --labelled-pod it runs nothing and trains nothing: it
# writes the rows of the POD of the labelled particles into the particle_frontends.csv in <r>,
# from the run stored in <d>.

using VlasovReductionBenchmark: add_labelled_pod, run_particles

const USAGE = "usage: julia --project scripts/particles.jl --runs-dir <d> --results-dir <r> [--smoke] [--labelled-pod]"

# The options of the command line `args`.
function parse_options(args)
    runs_dir = results_dir = nothing
    smoke = labelled_pod = false
    i = 1
    while i ≤ length(args)
        if args[i] == "--smoke"
            smoke = true
            i += 1
        elseif args[i] == "--labelled-pod"
            labelled_pod = true
            i += 1
        elseif args[i] in ("--runs-dir", "--results-dir") && i < length(args)
            args[i] == "--runs-dir" ? (runs_dir = args[i + 1]) : (results_dir = args[i + 1])
            i += 2
        else
            throw(ArgumentError(USAGE))
        end
    end
    (runs_dir === nothing || results_dir === nothing) && throw(ArgumentError(USAGE))

    return (runs_dir = runs_dir, results_dir = results_dir, smoke = smoke,
        labelled_pod = labelled_pod)
end

function main(args)
    (; runs_dir, results_dir, smoke, labelled_pod) = parse_options(args)
    labelled_pod ? add_labelled_pod(; runs_dir, results_dir, smoke) :
    run_particles(; runs_dir, results_dir, smoke)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
