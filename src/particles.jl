"""
The settings of the particle half: the number of `particles` of the PIC run, the
`train_particles` of the subsample that every frontend trains on, the end time `tend` of the
run, the latent dimensions `latent_dims`, the training `steps` and the `seeds` of the networks.
"""
const PARTICLE_FULL_MODE = (particles = 4096, train_particles = 256, tend = 40.0,
    latent_dims = (4, 8, 16, 32), steps = 2000, seeds = (42, 43, 44))

"The settings of [`PARTICLE_FULL_MODE`](@ref) for the smoke mode."
const PARTICLE_SMOKE_MODE = (particles = 512, train_particles = 64, tend = 10.0,
    latent_dims = (4, 8), steps = 100, seeds = (42,))

"""
    particle_mode(smoke)

The settings of the particle half: [`PARTICLE_SMOKE_MODE`](@ref) with `smoke`, else
[`PARTICLE_FULL_MODE`](@ref).
"""
particle_mode(smoke::Bool) = smoke ? PARTICLE_SMOKE_MODE : PARTICLE_FULL_MODE

"The largest relative drift of the energy of the PIC run that the script accepts."
const PIC_ENERGY_TOLERANCE = 1e-3

"The number of cells and the order (degree + 1) of the periodic B-spline basis of the potential."
const PIC_BASIS = (cells = 32, order = 4)

"The seed of the generator that draws the particles."
const PARTICLE_SEED = 42

"The step size of the training of a frontend."
const TRAINING_STEP_SIZE = 1e-3

"The particle frontends of `SymmetricSymplecticAutoencoder`, in the order of the results."
const FRONTENDS = (:sort, :slot, :pairwise)

"The fractions of the snapshot energy that the POD mode counts reach."
const POD_THRESHOLDS = (0.99, 0.9999)

"""
    inverse_cdf(u, α, k)

The position ``x ∈ [0, L)``, ``L = 2π/k``, at which the cumulative distribution
``F(x) = (x + α \\sin(kx)/k)/L`` of the density ``(1 + α \\cos kx)/L`` equals `u ∈ [0, 1)`,
found by bisection. `F` is increasing for ``|α| < 1``.
"""
function inverse_cdf(u::Real, α::Real, k::Real)
    L = 2π / k
    F(x) = (x + α * sin(k * x) / k) / L
    a, b = 0.0, L
    while true
        m = (a + b) / 2
        (m == a || m == b) && return a
        F(m) ≤ u ? (a = m) : (b = m)
    end
end

"""
    sample_particles(M, α, k; seed = PARTICLE_SEED)

`M` particles of the density ``(1 + α \\cos kx)\\, e^{-v^2/2}/\\sqrt{2π}`` on ``[0, 2π/k)``, from
one `Random.Xoshiro(seed)`: first the `M` positions by inverse-CDF sampling of
``1 + α \\cos kx`` ([`inverse_cdf`](@ref)), then the `M` velocities from the standard normal.
It returns the positions, the velocities and the weights, all equal to ``L/M``.
"""
function sample_particles(M::Integer, α::Real, k::Real; seed::Integer = PARTICLE_SEED)
    rng = Xoshiro(seed)
    x = [inverse_cdf(rand(rng), α, k) for _ in 1:M]
    v = randn(rng, M)

    return x, v, fill(2π / k / M, M)
end

"""
    particle_E1(x, w, k)

The first Fourier mode ``|E_1| = (2/(kL))\\, |\\sum_i w_i e^{-ikx_i}|`` of the density of the
particles at `x` with weights `w`, ``L = 2π/k``.
"""
function particle_E1(x::AbstractVector, w::AbstractVector, k::Real)
    L = 2π / k

    return 2 / (k * L) * abs(sum(w[i] * cis(-k * x[i]) for i in eachindex(x, w)))
end

"""
    wrap_position(x, L)

`x` reduced into ``[0, L)``. `mod(x, L)` gives `L` itself for a small negative `x`, which this
maps to `0`.
"""
function wrap_position(x::Real, L::Real)
    y = mod(x, L)

    return y < L ? y : zero(y)
end

"""
    sorted_snapshots(X, V)

The snapshots `X` (positions) and `V` (velocities), one particle per row and one snapshot per
column, with each column's particles reordered by position by a stable sort, the velocities
carried along.
"""
function sorted_snapshots(X::AbstractMatrix, V::AbstractMatrix)
    Xs, Vs = similar(X), similar(V)
    for j in axes(X, 2)
        p = sortperm(X[:, j])
        Xs[:, j] = X[p, j]
        Vs[:, j] = V[p, j]
    end

    return Xs, Vs
end

"""
    pod_modes(S, τ)

The smallest number of POD modes of the snapshot matrix `S` whose squared singular values hold
at least the fraction `τ ≤ 1` of the snapshot energy ``\\sum σ_i^2``.
"""
function pod_modes(S::AbstractMatrix, τ::Real)
    captured = cumsum(svdvals(S) .^ 2)

    return findfirst(c -> c ≥ τ * captured[end], captured)
end

"""
    pod_error(S, n)

The relative error ``\\|S - Π_n S\\|_F / \\|S\\|_F = (\\sum_{i > n} σ_i^2 / \\sum σ_i^2)^{1/2}`` of
the projection of the snapshot matrix `S` onto its first `n` POD modes.
"""
function pod_error(S::AbstractMatrix, n::Integer)
    σ² = svdvals(S) .^ 2

    return sqrt(sum(σ²[(n + 1):end]; init = 0.0) / sum(σ²))
end

"""
    check_pic_energy(drift)

Throw an error naming the first step whose relative energy drift `drift[s + 1]` exceeds
[`PIC_ENERGY_TOLERANCE`](@ref) in magnitude, or is not a number.
"""
function check_pic_energy(drift::AbstractVector)
    for (i, d) in enumerate(drift)
        abs(d) ≤ PIC_ENERGY_TOLERANCE || error(
            "nonlinear Landau damping with particles: the energy drifted by $(abs(d)) " *
            "relative at step $(i - 1), above $PIC_ENERGY_TOLERANCE")
    end

    return nothing
end

"""
    pic_method(x, v, w, k, tend)

VlasovMethods' `SplittingMethod` over ``[0, t_\\mathrm{end}]`` with step [`TIMESTEP`](@ref) for
the particles at `x` with velocities `v` and weights `w` on ``[0, 2π/k)``, in the potential on
the periodic B-spline basis of [`PIC_BASIS`](@ref).
"""
function pic_method(x, v, w, k, tend)
    dist = ParticleDistribution(1, 1, length(x))
    dist.particles.x .= x'
    dist.particles.v .= v'
    dist.particles.w .= w'
    potential = Potential(PeriodicBasisSpline((0.0, 2π / k), PIC_BASIS.order, PIC_BASIS.cells))

    return SplittingMethod(VlasovPoisson(dist, potential), (0.0, tend), TIMESTEP)
end

"""
    run_pic(M, tend; runs_dir = nothing, seed = PARTICLE_SEED)

Nonlinear Landau damping ([`nonlinear_landau`](@ref)) with `M` particles from
[`sample_particles`](@ref), stepped by VlasovMethods' `SplittingMethod` with `run!` over
``[0, t_\\mathrm{end}]``. `run!` does not wrap the positions, and the potential reduces them
itself; the stored snapshots are wrapped into ``[0, L)`` by [`wrap_position`](@ref).

It keeps the positions and the velocities every [`SNAPSHOT_INTERVAL`](@ref) (`M × N_t`), and
every step ``|E_1|`` ([`particle_E1`](@ref)), the relative drift of the energy (VlasovMethods'
`VlasovHamiltonian`) and the drift ``P(t) - P(0)`` of the momentum ``P = \\sum_i w_i v_i``. It
throws by [`check_pic_energy`](@ref) when the energy drifts too far. With `runs_dir` it writes
the run by [`write_particle_run`](@ref). It returns the run as a `NamedTuple`.
"""
function run_pic(M::Integer, tend::Real; runs_dir = nothing, seed::Integer = PARTICLE_SEED)
    case = nonlinear_landau()
    k, L = case.k, 2π / case.k
    x, v, w = sample_particles(M, case.α, k; seed)
    method = pic_method(x, v, w, k, tend)
    hamiltonian = VlasovMethods.VlasovHamiltonian(method.model)
    params = method.equation.parameters

    z = mktempdir() do dir
        file = joinpath(dir, "trajectory.h5")
        run!(method, file)
        h5open(f -> read(f, "z"), file)
    end
    nsteps = size(z, 3) - 1
    times = [s * TIMESTEP for s in 0:nsteps]
    stride = round(Int, SNAPSHOT_INTERVAL / TIMESTEP)
    snapshot_steps = 0:stride:nsteps

    energy = [hamiltonian(times[s + 1], z[:, :, s + 1], params) for s in 0:nsteps]
    momentum = [dot(w, z[2, :, s + 1]) for s in 0:nsteps]
    run = (case = case, particles = M, seed = seed, tend = Float64(tend), times = times,
        snapshot_times = times[snapshot_steps .+ 1],
        positions = wrap_position.(z[1, :, snapshot_steps .+ 1], L),
        velocities = z[2, :, snapshot_steps .+ 1], weights = w,
        E1 = [particle_E1(view(z, 1, :, s + 1), w, k) for s in 0:nsteps],
        energy_drift = (energy .- energy[1]) ./ energy[1],
        momentum_drift = momentum .- momentum[1])
    check_pic_energy(run.energy_drift)
    runs_dir === nothing || write_particle_run(runs_dir, run)

    return run
end

"""
    write_particle_run(runs_dir, run)

Write `run`, as [`run_pic`](@ref) returns it, to `particles_<case>.h5` in `runs_dir`, over any
file of that name. The datasets are `positions` and `velocities` (`M × N_t`), `snapshot_times`,
`weights`, `times` (every step), `E1`, `energy_drift` and `momentum_drift`; the attributes are the
case, `alpha`, `k`, `xdomain`, `particles`, `dt`, `tend`, `ncells`, `order` and `seed`.
"""
function write_particle_run(runs_dir::AbstractString, run)
    mkpath(runs_dir)
    path = joinpath(runs_dir, "particles_$(run.case.name).h5")
    h5open(path, "w") do file
        for name in (:positions, :velocities, :snapshot_times, :weights, :times, :E1,
            :energy_drift, :momentum_drift)
            file[string(name)] = getproperty(run, name)
        end
        a = attrs(file)
        a["case"] = run.case.name
        a["alpha"] = run.case.α
        a["k"] = run.case.k
        a["xdomain"] = [0.0, 2π / run.case.k]
        a["particles"] = run.particles
        a["dt"] = TIMESTEP
        a["tend"] = run.tend
        a["ncells"] = PIC_BASIS.cells
        a["order"] = PIC_BASIS.order
        a["seed"] = run.seed
    end

    return path
end

"""
    charbonnier_relative(Ẑ, Z; ε = 1e-3)

The Charbonnier relative error ``(\\sqrt{\\|Ẑ - Z\\|_F^2 + ε^2} - ε) / \\sqrt{\\|Z\\|_F^2 + ε^2}``
of the reconstruction `Ẑ` of `Z`: the reconstruction term of GeometricMachineLearning's
`CharbonnierRelativeLoss`, which has methods only for the multi-modal network.
"""
function charbonnier_relative(Ẑ, Z; ε = 1e-3)
    return (sqrt(sum(abs2, Ẑ .- Z) + ε^2) - ε) / sqrt(sum(abs2, Z) + ε^2)
end

"""
    training_step!(ps, optimizer, loss)

One `optimization_step!` of `optimizer` on the network parameters `ps` along the Zygote gradient
of `loss`. It returns `true`, or `false` with `ps` unchanged when the loss or an entry of its
gradient is not finite: `FactoredStiefelAdam` refuses such a gradient with an `ArgumentError`.
"""
function training_step!(ps, optimizer, loss)
    l, gradient = Zygote.withgradient(loss, ps)
    finite = isfinite(l) &&
             all(layer -> all(a -> all(isfinite, a), values(layer)), gradient[1])
    finite && optimization_step!(ps, optimizer, gradient[1])

    return finite
end

"""
    train_frontend(frontend, Z, n; seed, steps)

Train `SymmetricSymplecticAutoencoder(M, 1, n; frontend)` on the snapshots `Z`, one flat vector
``[q; p]`` of length ``2M`` per column, from `NeuralNetwork(…; rng = Xoshiro(seed))`: `steps`
[`training_step!`](@ref)s on the gradient of [`charbonnier_relative`](@ref), with
`CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam())` at step size
[`TRAINING_STEP_SIZE`](@ref).

It returns the number `M` of `particles` it trained on, the relative reconstruction error
``\\|Ẑ - Z\\|_F / \\|Z\\|_F`` after the training, the `status`, and the wall time per step
taken, the compilation of the first step included. A loss or a gradient that is not finite
after `s` steps, or a loss that is not finite after the last step, ends the training with
status `"failed at step s"`, and the error is that of the network after those `s` steps.
"""
function train_frontend(frontend::Symbol, Z::AbstractMatrix, n::Integer; seed::Integer,
        steps::Integer)
    M = size(Z, 1) ÷ 2
    nn = NeuralNetwork(SymmetricSymplecticAutoencoder(M, 1, n; frontend), Float64;
        rng = Xoshiro(seed))
    ps = nn.params
    optimizer = TrainingOptimizer(ps;
        algorithm = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam()),
        linesearch = TRAINING_STEP_SIZE)
    loss(p) = charbonnier_relative(nn.model(Z, p), Z)

    taken = 0
    elapsed = @elapsed while taken < steps && training_step!(ps, optimizer, loss)
        taken += 1
    end
    status = taken == steps && isfinite(loss(ps)) ? "ok" : "failed at step $taken"

    return (particles = M, relative_error = norm(nn.model(Z, ps) .- Z) / norm(Z),
        status = status, seconds_per_step = elapsed / max(taken, 1))
end

"""
    train_subsample(positions, velocities, M)

The training snapshots of the `M` particles of smallest initial position, the first `M` of the
initial sample sorted by position, with their labels kept in time: one column ``[x; v]`` per
snapshot.
"""
function train_subsample(positions::AbstractMatrix, velocities::AbstractMatrix, M::Integer)
    labels = sortperm(positions[:, 1])[1:M]

    return vcat(positions[labels, :], velocities[labels, :])
end

"""
    frontend_rows(positions, velocities; train_particles, latent_dims, seeds, steps)

The rows of `particle_frontends.csv`: for each frontend of [`FRONTENDS`](@ref), latent dimension
in `latent_dims` and seed in `seeds`, [`train_frontend`](@ref) on the
[`train_subsample`](@ref) of `train_particles` particles; then for each latent dimension `n` the
baseline, [`pod_error`](@ref) of `n` modes of the same particles sorted in each snapshot. Each
row records the number of particles it was computed on.
"""
function frontend_rows(positions::AbstractMatrix, velocities::AbstractMatrix;
        train_particles::Integer, latent_dims, seeds, steps::Integer)
    Z = train_subsample(positions, velocities, train_particles)
    M = size(Z, 1) ÷ 2
    rows = NamedTuple[]
    for frontend in FRONTENDS, n in latent_dims, seed in seeds
        r = train_frontend(frontend, Z, n; seed, steps)
        @printf("  %s, n = %d, seed %d, %d particles: relative error %.4e, %s, %.3e s per step\n",
            frontend, n, seed, r.particles, r.relative_error, r.status, r.seconds_per_step)
        flush(stdout)
        push!(rows,
            (method = string(frontend), n = n,
                seed = string(seed), particles = r.particles,
                steps = steps, status = r.status, relative_error = r.relative_error,
                seconds_per_step = r.seconds_per_step))
    end
    sorted = vcat(sorted_snapshots(Z[1:M, :], Z[(M + 1):end, :])...)
    for n in latent_dims
        push!(rows,
            (method = "pod", n = n, seed = "none",
                particles = M, steps = 0, status = "ok",
                relative_error = pod_error(sorted, n), seconds_per_step = 0.0))
    end

    return rows
end

"""
    pic_step_time(M; steps = 100, warmup = 20)

The wall time in seconds of one step of the PIC run of [`run_pic`](@ref) with `M` particles:
the time of a `run!` of `steps` steps, after a warm-up `run!` of `warmup` steps, divided by
`steps`. It includes the HDF5 write of each step by `run!`.
"""
function pic_step_time(M::Integer; steps::Integer = 100, warmup::Integer = 20)
    case = nonlinear_landau()
    x, v, w = sample_particles(M, case.α, case.k)
    mktempdir() do dir
        run!(pic_method(x, v, w, case.k, warmup * TIMESTEP), joinpath(dir, "warmup.h5"))
        method = pic_method(x, v, w, case.k, steps * TIMESTEP)

        return @elapsed(run!(method, joinpath(dir, "timed.h5"))) / steps
    end
end

"""
    particle_cost(results_dir, M; steps = 100)

Measure [`pic_step_time`](@ref) in a fresh Julia process on the active project, and write it
with the machine's `uptime`, the id `pid` and the number of `threads` of the measuring process to
`<results_dir>/particle_cost.toml`.
"""
function particle_cost(results_dir::AbstractString, M::Integer; steps::Integer = 100)
    code = "using VlasovReductionBenchmark; println(getpid(), ' ', Threads.nthreads(), ' ', " *
           "VlasovReductionBenchmark.pic_step_time($M; steps = $steps))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    pid, threads, seconds = split(last(readlines(cmd)))
    pid, threads, seconds = parse(Int, pid), parse(Int, threads), parse(Float64, seconds)
    uptime = Sys.isunix() ? readchomp(`uptime`) :
             @sprintf("up %.0f s, load averages %s", Sys.uptime(), join(Sys.loadavg(), " "))

    mkpath(results_dir)
    path = joinpath(results_dir, "particle_cost.toml")
    open(path, "w") do io
        TOML.print(io,
            Dict("particles" => M, "steps" => steps, "dt" => TIMESTEP,
                "method" => "SplittingMethod (Strang)", "seconds_per_step" => seconds,
                "note" => "the time of run!, which includes its HDF5 write of each step",
                "uptime" => uptime, "julia" => string(VERSION), "threads" => threads,
                "pid" => pid))
    end
    @printf("PIC cost with %d particles: %.3e s per step (%s)\n", M, seconds, uptime)
    flush(stdout)

    return path
end

"""
    failed_runs(rows)

The number of training runs among the rows of [`frontend_rows`](@ref) whose status is not
`"ok"`, and the number of training runs; the rows of the POD baseline are no training runs.
"""
function failed_runs(rows)
    trained = filter(r -> r.method != "pod", rows)

    return count(r -> r.status != "ok", trained), length(trained)
end

"Write the rows `rows`, `NamedTuple`s with the same keys, to the CSV file `path`."
function write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(keys(first(rows)), ','))
        for row in rows
            println(io, join(values(row), ','))
        end
    end

    return path
end

"""
    run_particles(; runs_dir, results_dir, smoke = false)

The particle half of the benchmark, with the settings of [`PARTICLE_FULL_MODE`](@ref), or
[`PARTICLE_SMOKE_MODE`](@ref) with `smoke = true`: the PIC run of [`run_pic`](@ref), written to
`runs_dir`, which asserts the energy and prints the largest momentum drift; the POD mode counts
([`pod_modes`](@ref)) of the labelled and of the sorted ([`sorted_snapshots`](@ref)) snapshots of
all particles at the thresholds of [`POD_THRESHOLDS`](@ref), in
`<results_dir>/particle_pod_modes.csv`; the frontends and the POD baseline of
[`frontend_rows`](@ref), in `<results_dir>/particle_frontends.csv`, with the count of the
failed training runs printed; and the cost per step of [`particle_cost`](@ref).
"""
function run_particles(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false)
    mode = particle_mode(smoke)
    M = mode.particles
    @printf("PIC run of nonlinear Landau damping with %d particles over [0, %g]%s\n", M,
        mode.tend, smoke ? " (smoke)" : "")
    flush(stdout)

    elapsed = @elapsed run = run_pic(M, mode.tend; runs_dir)
    @printf("  %d steps in %.1f s; largest energy drift %.4e (asserted at most %g), largest momentum drift %.4e, not asserted\n",
        length(run.times) - 1, elapsed, maximum(abs, run.energy_drift),
        PIC_ENERGY_TOLERANCE, maximum(abs, run.momentum_drift))
    flush(stdout)

    mkpath(results_dir)
    labelled = vcat(run.positions, run.velocities)
    sorted = vcat(sorted_snapshots(run.positions, run.velocities)...)
    pod = [(snapshots = name, particles = M, threshold = τ, modes = pod_modes(S, τ))
           for (name, S) in (("labelled", labelled), ("sorted", sorted)),
    τ in POD_THRESHOLDS]
    write_csv(joinpath(results_dir, "particle_pod_modes.csv"), vec(permutedims(pod)))

    rows = frontend_rows(run.positions, run.velocities;
        train_particles = mode.train_particles, latent_dims = mode.latent_dims,
        seeds = mode.seeds, steps = mode.steps)
    write_csv(joinpath(results_dir, "particle_frontends.csv"), rows)
    @printf("%d of %d training runs failed\n", failed_runs(rows)...)
    flush(stdout)

    particle_cost(results_dir, M)

    return nothing
end
