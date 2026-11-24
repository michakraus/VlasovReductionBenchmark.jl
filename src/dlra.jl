"""
The time step of every DLRA run, with a classical RK4 step per substep: the
[`TIMESTEP`](@ref) of the full-order runs, so that a run at full rank reproduces the full-order
run.
"""
const DLRA_TIMESTEP = TIMESTEP

"The number of DLRA steps between two stored snapshots, the [`SNAPSHOT_INTERVAL`](@ref) of the
full-order runs."
const DLRA_STRIDE = round(Int, SNAPSHOT_INTERVAL / DLRA_TIMESTEP)

"The largest relative difference of a DLRA run at ``r = n_x`` from the full-order run."
const FULL_RANK_TOLERANCE = 1e-10

"The number of steps over which a DLRA run at ``r = n_x`` is compared with the full-order run."
const AGREEMENT_STEPS = 20

"The rank of the DLRA run whose damping rate of linear Landau damping is checked."
const CHECK_RANK = 16

"""
The columns of `dlra.csv`, in order. The column `n` holds the rank ``r``. The column
`full_rank_l2_error_max` holds the largest relative ``L^2`` error of the reference of the row
against the full-order run, and `max_l2_error_fluct_full_rank` the largest error of the run of
rank ``r`` against that reference, relative to its fluctuation. The reference is the full-order
run itself, so the first is `0.0` and the second equals `max_l2_error_fluct`.
"""
const DLRA_COLUMNS = ("method", "kind", "case", "alpha", "n", "state_size", "status",
    "l2_error_max", "l2_errors", "max_l2_error_fluct", "l2_error_fluct", "rate", "rate_status",
    "rate_full_order", "energy_drift", "l1_drift", "l2_drift", "entropy_drift", "mass_drift",
    "nonpositive_nodes", "full_rank_l2_error_max", "max_l2_error_fluct_full_rank",
    "offline_seconds", "online_seconds_per_step", "online_ratio")

"""
    dlra_mode(smoke)

The settings of [`FULL_MODE`](@ref), or of [`SMOKE_MODE`](@ref) for `smoke = true`, with the
DLRA `ranks` of the mode: ``r ∈ \\{4, 8, 16, 32\\}``, or ``\\{4, 8\\}`` in the smoke mode.
"""
function dlra_mode(smoke::Bool)
    smoke ? merge(SMOKE_MODE, (ranks = (4, 8),)) :
    merge(FULL_MODE, (ranks = (4, 8, 16, 32),))
end

"""
    dlra_state_size(r, nx, nv)

The number of values of the factored state ``(X, S, V)`` of rank `r` on the `nx × nv` grid,
``r (n_x + n_v) + r^2``.
"""
dlra_state_size(r::Integer, nx::Integer, nv::Integer) = r * (nx + nv) + r^2

"""
    spectral_derivative(nx, k)

The `nx × nx` matrix of the spectral derivative on `nx` equidistant nodes of the periodic
interval ``[0, 2π/k)``: ``∂_x e^{imkx} = imk\\, e^{imkx}`` for ``|m| < n_x/2``, and zero for
the Nyquist mode ``m = n_x/2``.
"""
function spectral_derivative(nx::Integer, k::Real)
    return [-2k / nx *
            sum(m * sinpi(2m * (a - b) / nx) for m in 1:(cld(nx, 2) - 1); init = 0.0)
            for a in 1:nx, b in 1:nx]
end

"""
    dlra_fields(model, Y)

VlasovMethods' workspace `LowRankFields1D1V` of the fields of the low-rank state `Y` of the
grid model `model` of [`vlasov_model`](@ref): its Green's matrix `model.work`, and the default
periodic velocity closure, the closure of the full-order runs.
"""
dlra_fields(model, Y) = VlasovMethods.LowRankFields1D1V(Y, model.work)

"""
    dlra_step!(Y, field, t)

One DLRA step of the low-rank state `Y` in place, from the time `t`: one step of length
[`DLRA_TIMESTEP`](@ref) of VlasovMethods' `AugmentedBUG()`, truncated back to the rank of `Y`,
with one classical RK4 step per substep, on the workspace `field` of [`dlra_fields`](@ref). A
step that leaves a non-finite core does not throw; `all(isfinite, Y.S)` is its status.
"""
dlra_step!(Y, field, t) = lowrank_step!(Y, AugmentedBUG(), field, t, DLRA_TIMESTEP)

"""
    full_order_stepper(case, nx, nv, steps)

The integrator of [`run_case`](@ref) for `case` on the `nx × nv` grid over `steps` steps of
[`TIMESTEP`](@ref), as a function of the step number: called with `s = 1, 2, …` in turn, it
takes step `s` and returns the `nx × nv` state after it. The returned array is the integrator's
own state, which the next call overwrites.
"""
function full_order_stepper(case::VlasovCase, nx::Integer, nv::Integer, steps::Integer)
    model = vlasov_model(case, nx, nv)
    int = VlasovMethods.GeometricIntegrator(model, (0.0, steps * TIMESTEP), TIMESTEP, RK4())
    solstep = solutionstep(int.integrator, initialstate(int.equation))
    return function (s)
        reset!(solstep, s * TIMESTEP)
        integrate!(solstep, int.integrator)
        return reshape(solstep.q, nx, nv)
    end
end

"""
    full_rank_agreement(case, nx, nv, r, steps; perturb = nothing)

The largest relative difference ``\\|f_r - f\\|_2 / \\|f\\|_2`` over `steps` steps between the
DLRA run of rank `r` of `case` on the `nx × nv` grid, by [`dlra_step!`](@ref), and the
full-order run of [`full_order_stepper`](@ref), both from the initial condition. It is `NaN`
when either run leaves a non-finite value. `perturb(step, S)`, where given, is called on the
core ``S`` of the DLRA run after each step and before this check; it lets a test break a run.
"""
function full_rank_agreement(case::VlasovCase, nx::Integer, nv::Integer, r::Integer,
        steps::Integer; perturb = nothing)
    model = vlasov_model(case, nx, nv)
    Y = LowRankDistribution(model.distribution, r)
    field = dlra_fields(model, Y)
    reference! = full_order_stepper(case, nx, nv, steps)
    err = 0.0
    for s in 1:steps
        dlra_step!(Y, field, (s - 1) * DLRA_TIMESTEP)
        perturb === nothing || perturb(s, Y.S)
        f = reference!(s)
        all(isfinite, Y.S) && all(isfinite, f) || return NaN
        err = max(err, norm(GridDistribution(Y).values .- f) / norm(f))
    end
    return err
end

"""
    check_full_rank_agreement(case, nx, nv; perturb = nothing)

Print the [`full_rank_agreement`](@ref) of `case` at ``r = n_x`` over
[`AGREEMENT_STEPS`](@ref) steps, with `perturb`, and throw an error when it is above
[`FULL_RANK_TOLERANCE`](@ref) or not finite.
"""
function check_full_rank_agreement(case::VlasovCase, nx::Integer, nv::Integer;
        perturb = nothing)
    err = full_rank_agreement(case, nx, nv, nx, AGREEMENT_STEPS; perturb)
    @printf("full-rank check %s on %d × %d: relative difference %.2e\n", case.name, nx, nv,
        err)
    flush(stdout)
    err ≤ FULL_RANK_TOLERANCE || error(
        "full-rank check $(case.name) on $nx × $nv: the DLRA run at r = $nx differs from " *
        "the full-rank solver by $err relative over $AGREEMENT_STEPS steps, above " *
        "$FULL_RANK_TOLERANCE")
    return err
end

"""
    run_dlra(case, nx, nv, r; perturb = nothing)

The DLRA run of rank `r` of `case` on the `nx × nv` grid: [`dlra_step!`](@ref) with step
[`DLRA_TIMESTEP`](@ref) over ``[0, t_\\mathrm{end}]``, from VlasovMethods' factorisation
`LowRankDistribution(dist, r)` of the distribution `dist` of [`vlasov_model`](@ref), with no
fixed modes. It returns the `times` of every step, the decoded `snapshots` every
[`DLRA_STRIDE`](@ref) steps, the final low-rank state `factors`, a `LowRankDistribution`, the
`offline_seconds` of the initial factorisation, and every step the `mass`, `energy`, `E1`, `l1`,
`l2`, `entropy` and `nonpositive` count of [`invariants`](@ref) of the decoded state.

A step that leaves a non-finite ``S`` ends the run: its `status` is `"failed at step <s>"` and
`failed_step` is `s`; otherwise the status is `"ok"` and `failed_step` is `nothing`.
`perturb(step, S)`, where given, is called on ``S`` after each step and before this check; it
lets a test break a run. It throws an `ArgumentError` unless ``1 ≤ r ≤ \\min(n_x, n_v)``.
"""
function run_dlra(case::VlasovCase, nx::Integer, nv::Integer, r::Integer; perturb = nothing)
    1 ≤ r ≤ min(nx, nv) || throw(ArgumentError(
        "the rank $r is not in 1:$(min(nx, nv)) for the $nx × $nv grid"))
    model = vlasov_model(case, nx, nv)
    x, v = model.distribution.x, model.distribution.v
    hx, hv = step(x), step(v)
    timed = @timed LowRankDistribution(model.distribution, r)
    Y = timed.value
    field = dlra_fields(model, Y)

    nsteps = round(Int, case.tend / DLRA_TIMESTEP)
    times = [s * DLRA_TIMESTEP for s in 0:nsteps]
    snapshots = zeros(nx, nv, nsteps ÷ DLRA_STRIDE + 1)
    series = (mass = Float64[], energy = Float64[], E1 = Float64[], l1 = Float64[],
        l2 = Float64[], entropy = Float64[], nonpositive = Int[])
    status, failed_step = "ok", nothing
    for s in 0:nsteps
        if s > 0
            dlra_step!(Y, field, times[s])
            perturb === nothing || perturb(s, Y.S)
            if !all(isfinite, Y.S)
                status, failed_step = "failed at step $s", s
                break
            end
        end
        f = GridDistribution(Y).values
        i = invariants(f, model.work, case.k, x, v, hx, hv)
        for name in keys(series)
            push!(series[name], i[name])
        end
        s % DLRA_STRIDE == 0 && (snapshots[:, :, s ÷ DLRA_STRIDE + 1] .= f)
    end

    return (
        status = status, failed_step = failed_step, times = times, snapshots = snapshots,
        factors = Y, offline_seconds = timed.time, series...)
end

"""
    check_dlra_rates(mode; perturb = nothing)

The checks of the damping rate of linear Landau damping at ``α = 0.01`` on the grid of `mode`,
a [`dlra_mode`](@ref), fitted by [`landau_rate_fit`](@ref) over its `landau_window` against
[`landau_root`](@ref) with its `landau_rtol`: first the full-order run of [`run_case`](@ref),
so that a failure of the DLRA run is the rank's and not the discretisation's, then the DLRA run
of rank [`CHECK_RANK`](@ref), with `perturb`. Each throws the error of [`check_rate`](@ref), and
the second an error of its own when the DLRA run stops.
"""
function check_dlra_rates(mode; perturb = nothing)
    case, (nx, nv) = linear_landau(), mode.grid
    oracle = imag(landau_root(case.k))
    full = run_case(case, nx, nv)
    check_rate("full-order run linear Landau γ",
        landau_rate_fit(full.times, log.(full.E1), mode.landau_window).γ, oracle,
        mode.landau_rtol)
    dlra = run_dlra(case, nx, nv, CHECK_RANK; perturb)
    dlra.status == "ok" ||
        error("the DLRA run of rank $CHECK_RANK of linear Landau damping " *
              "on $nx × $nv stopped: $(dlra.status)")
    check_rate("DLRA r = $CHECK_RANK linear Landau γ",
        landau_rate_fit(dlra.times, log.(dlra.E1), mode.landau_window).γ, oracle,
        mode.landau_rtol)
    return nothing
end

"A row of `dlra.csv` with the given columns, every other column empty."
function dlra_row(; kwargs...)
    merge(Dict(c => "" for c in DLRA_COLUMNS),
        Dict(string(k) => string(v) for (k, v) in kwargs))
end

"""
    dlra_data_row(run, kind, r, mode; perturb = nothing)

The row of the DLRA run of rank `r` of the full-order `run`, as [`read_run`](@ref) returns it:
the [`run_dlra`](@ref), with `perturb`, and its [`trajectory_metrics`](@ref) against the run's
snapshots, around the background of the run's first snapshot, the rate and its status of
[`reduced_rate`](@ref), the rate of [`fitted_rate`](@ref) of the run, the drift of the mass, the
[`dlra_state_size`](@ref) and the `kind`, `"data"` or `"sweep"`. The reference of the two
full-rank columns of [`DLRA_COLUMNS`](@ref) is the run's own snapshots: the row holds their
largest relative ``L^2`` error against the run's snapshots, `0.0`, and the largest error of the
DLRA run against them, relative to their fluctuation around the background of their first
snapshot. A run that stops leaves every metric of the trajectory empty.
"""
function dlra_data_row(run, kind, r, mode; perturb = nothing)
    case = run.case
    reference = run.snapshots
    dlra = run_dlra(case, run.nx, run.nv, r; perturb)
    common = (method = "DLRA", kind = kind, case = case.name, alpha = case.α, n = r,
        state_size = dlra_state_size(r, run.nx, run.nv), status = dlra.status,
        full_rank_l2_error_max = maximum(relative_l2_errors(reference, run.snapshots)),
        offline_seconds = dlra.offline_seconds)
    dlra.status == "ok" || return dlra_row(; common...)

    m = trajectory_metrics(dlra.snapshots, run.snapshots, dlra,
        background(run.snapshots[:, :, 1]))
    own = relative_l2_errors(dlra.snapshots, reference, background(reference[:, :, 1]))
    rate, rate_status = reduced_rate(case.name, dlra.times, dlra.E1, mode.landau_window)
    rate_full = fitted_rate(case.name, run.times, run.E1, mode.landau_window)

    return dlra_row(; common..., l2_error_max = m.l2_error_max,
        l2_errors = join(m.l2_errors, ';'), max_l2_error_fluct = m.l2_error_fluct_max,
        l2_error_fluct = join(m.l2_errors_fluct, ';'), rate = something(rate, ""),
        rate_status = rate_status, rate_full_order = something(rate_full, ""),
        energy_drift = m.energy_drift, l1_drift = m.l1_drift, l2_drift = m.l2_drift,
        entropy_drift = m.entropy_drift, mass_drift = max_drift(dlra.mass),
        nonpositive_nodes = m.nonpositive_max, max_l2_error_fluct_full_rank = maximum(own))
end

"""
    dlra_step_time(case, nx, nv, r; steps = 100, warmup = 10)

The wall time in seconds of one [`dlra_step!`](@ref) of rank `r` of `case` on the `nx × nv`
grid, averaged over `steps` steps after `warmup` steps from the factorisation of
[`run_dlra`](@ref).
"""
function dlra_step_time(case::VlasovCase, nx::Integer, nv::Integer, r::Integer;
        steps::Integer = 100, warmup::Integer = 10)
    model = vlasov_model(case, nx, nv)
    Y = LowRankDistribution(model.distribution, r)
    field = dlra_fields(model, Y)
    timed_bug_steps(Y, field, warmup)
    return timed_bug_steps(Y, field, steps) / steps
end

"""
    timed_bug_steps(Y, field, n)

The wall time of `n` steps of [`dlra_step!`](@ref), which advance the low-rank state `Y` in
place. It is a function barrier: the types of `Y` and `field` are concrete inside it.
"""
function timed_bug_steps(Y, field, n::Integer)
    return @elapsed for s in 1:n
        dlra_step!(Y, field, (s - 1) * DLRA_TIMESTEP)
    end
end

"""
    dlra_step_times(runs_dir, specs)

Print the id and the thread count of this process, then for each `(file, r)` of `specs` the
[`dlra_step_time`](@ref) of rank `r` of the case of that run file in `runs_dir`.
[`dlra_costs`](@ref) calls it in a fresh process.
"""
function dlra_step_times(runs_dir::AbstractString, specs)
    println(getpid(), ' ', Threads.nthreads())
    for (file, r) in specs
        run = read_run(joinpath(runs_dir, file))
        println(file, ' ', r, ' ', dlra_step_time(run.case, run.nx, run.nv, r))
    end
    return nothing
end

"""
    dlra_costs(results_dir, runs_dir, specs, full_order_seconds)

Measure [`dlra_step_times`](@ref) of `specs` in a fresh Julia process on the active project, and
write them with the machine's `uptime`, the id `pid` and the `threads` of the measuring process
and the full-order cost `full_order_seconds` per step to `<results_dir>/dlra_cost.toml`. It
returns the seconds per step by `(file, r)`.
"""
function dlra_costs(results_dir::AbstractString, runs_dir::AbstractString, specs,
        full_order_seconds::Real)
    code = "using VlasovReductionBenchmark; VlasovReductionBenchmark.dlra_step_times(" *
           "$(repr(runs_dir)), $(repr(collect(specs))))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    lines = readlines(cmd)
    pid, threads = parse.(Int, split(lines[end - length(specs)]))
    seconds = Dict{Tuple{String, Int}, Float64}()
    for line in lines[(end - length(specs) + 1):end]
        file, r, s = split(line)
        seconds[(file, parse(Int, r))] = parse(Float64, s)
    end
    uptime = Sys.isunix() ? readchomp(`uptime`) :
             @sprintf("up %.0f s, load averages %s", Sys.uptime(), join(Sys.loadavg(), " "))

    open(joinpath(results_dir, "dlra_cost.toml"), "w") do io
        TOML.print(io,
            Dict("steps" => 100, "warmup" => 10, "dt" => DLRA_TIMESTEP,
                "method" => "augmented BUG, RK4 substeps", "uptime" => uptime, "pid" => pid,
                "threads" => threads, "julia" => string(VERSION),
                "full_order_seconds_per_step" => full_order_seconds,
                "rows" => [Dict("file" => f, "n" => r, "seconds_per_step" => s)
                 for ((f, r), s) in sort(collect(seconds))]))
    end

    return seconds
end

"""
    write_dlra(path, rows)

Write the `rows` of [`dlra_row`](@ref) to the CSV file `path`, with the header
[`DLRA_COLUMNS`](@ref).
"""
function write_dlra(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(DLRA_COLUMNS, ','))
        for row in rows
            println(io, join((csv_field(row[c]) for c in DLRA_COLUMNS), ','))
        end
    end
    return path
end

"""
    run_dlra_benchmark(; runs_dir, results_dir, smoke = false, perturb = nothing)

The DLRA runs of the full-order runs in `runs_dir`, as `scripts/full_order.jl` writes them, with
the settings of [`dlra_mode`](@ref): the three cases of [`DATA_RUNS`](@ref), each at the amplitude
of its run, and linear Landau damping at the [`TEST_AMPLITUDES`](@ref) of the sweep, for each rank
of the mode.

First it checks the DLRA runs: [`check_full_rank_agreement`](@ref) of the three cases on the
smoke grid ``32 × 64``, and [`check_dlra_rates`](@ref) on the grid of the mode. Then for each run
it writes a [`dlra_data_row`](@ref) of kind `"data"` (the cases) or `"sweep"` (the test
amplitudes) per rank, against the run's own snapshots. The online cost of each row that
completes comes from [`dlra_costs`](@ref), as a ratio to `<results_dir>/full_order_cost.toml`;
the ratio is per step, and a DLRA step and a full-order step are both [`TIMESTEP`](@ref) long.
It writes `<results_dir>/dlra.csv`, prints the number of rows and of failed runs, and returns
the rows. `perturb` goes to every [`run_dlra`](@ref) of a row; it lets a test break the runs.

It throws an error when a check fails, or when the grid of `full_order_cost.toml` or of a run file
is not the grid of the mode.
"""
function run_dlra_benchmark(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false, perturb = nothing)
    mode = dlra_mode(smoke)
    full_cost = TOML.parsefile(joinpath(results_dir, "full_order_cost.toml"))
    (full_cost["nx"], full_cost["nv"]) == mode.grid || error(
        "the full-order cost is measured on $(full_cost["nx"]) × $(full_cost["nv"]), " *
        "not on the $(mode.grid[1]) × $(mode.grid[2]) grid of this mode")
    @printf("DLRA on %d × %d%s, r ∈ %s\n", mode.grid..., smoke ? " (smoke)" : "",
        mode.ranks)
    flush(stdout)

    for case in (linear_landau(), nonlinear_landau(), two_stream())
        check_full_rank_agreement(case, SMOKE_MODE.grid...)
    end
    check_dlra_rates(mode)

    files = vcat([file => "data" for file in DATA_RUNS],
        [run_file(linear_landau(α)) => "sweep" for α in TEST_AMPLITUDES])
    rows, row_files = Dict{String, String}[], String[]
    for (file, kind) in files
        run = read_run(joinpath(runs_dir, file))
        (run.nx, run.nv) == mode.grid || error(
            "$file holds a $(run.nx) × $(run.nv) run, not the $(mode.grid[1]) × " *
            "$(mode.grid[2]) grid of this mode")
        for r in mode.ranks
            push!(rows, dlra_data_row(run, kind, r, mode; perturb))
            push!(row_files, file)
            @printf("  DLRA, %s, r = %d: %s\n", file, r, rows[end]["status"])
            flush(stdout)
        end
    end

    specs = [(row_files[i], parse(Int, rows[i]["n"]))
             for i in eachindex(rows) if rows[i]["status"] == "ok"]
    seconds = dlra_costs(results_dir, runs_dir, specs, full_cost["seconds_per_step"])
    for (i, file) in enumerate(row_files)
        key = (file, parse(Int, rows[i]["n"]))
        haskey(seconds, key) || continue
        rows[i]["online_seconds_per_step"] = string(seconds[key])
        rows[i]["online_ratio"] = string(seconds[key] / full_cost["seconds_per_step"])
    end

    path = write_dlra(joinpath(results_dir, "dlra.csv"), rows)
    nfailed = count(r -> r["status"] != "ok", rows)
    @printf("DLRA: %d rows, %d failed, written to %s\n", length(rows), nfailed, path)
    flush(stdout)

    return rows
end
