"The latent dimensions of the full mode and of the smoke mode."
const BASELINE_SIZES = (full = (4, 8, 16, 32), smoke = (4, 8))

"The run files of the three data cases, each at the amplitude of its full-order run."
const DATA_RUNS = ("linear_landau_alpha0.01.h5", "nonlinear_landau_alpha0.5.h5",
    "two_stream_alpha0.05.h5")

"The linear baselines, by name, with the function that builds each reduction."
const LINEAR_METHODS = ("POD" => pod_reduction, "PSD" => psd_reduction)

"The columns of `linear_baselines.csv`, in order."
const BASELINE_COLUMNS = ("method", "kind", "case", "alpha", "n", "state_size", "status",
    "l2_error_max", "l2_errors", "max_l2_error_fluct", "l2_error_fluct",
    "rate", "rate_status", "rate_full_order", "energy_drift",
    "l1_drift",
    "l2_drift", "entropy_drift", "nonpositive_nodes", "projection_error", "offline_seconds",
    "online_seconds_per_step", "online_ratio")

"The velocity profile of each case name of a run file."
const PROFILES = Dict("linear_landau" => maxwellian, "nonlinear_landau" => maxwellian,
    "two_stream" => two_stream_profile)

"""
    read_run(path)

The run file at `path`, as [`write_run`](@ref) writes it: its `case`, rebuilt from the
attributes, the grid `nx`, `nv`, its `label`, the `snapshots`, and the `times` and `E1` of every
step.
"""
function read_run(path::AbstractString)
    return h5open(path) do file
        a = attrs(file)
        case = VlasovCase(a["case"], PROFILES[a["case"]], a["alpha"], a["k"], a["tend"])
        (case = case, nx = a["nx"], nv = a["nv"], label = a["label"],
            snapshots = read(file, "snapshots"), times = read(file, "times"),
            E1 = read(file, "E1"))
    end
end

"A row of `linear_baselines.csv` with the given columns, every other column empty."
function baseline_row(; kwargs...)
    merge(Dict(c => "" for c in BASELINE_COLUMNS),
        Dict(string(k) => string(v) for (k, v) in kwargs))
end

"""
    reduced_rate(name, t, E1, landau_window)

The [`fitted_rate`](@ref) of a reduced run and its status: `"ok"`, `"none"` for a case without a
rate, or the message of a Landau fit that finds fewer than four maxima, with the rate `nothing`.
A reduced model can lose the oscillation of ``E_1``, and the row records that rather than end
the script.
"""
function reduced_rate(name::AbstractString, t, E1, landau_window)
    rate = try
        fitted_rate(name, t, E1, landau_window)
    catch e
        e isa ErrorException || rethrow()
        return nothing, "fit failed: " * e.msg
    end
    return rate, rate === nothing ? "none" : "ok"
end

"""
    data_row(method, run, n, r, offline, projection, mode; perturb = nothing)

The row of the reduction `r` of the data `run` at latent dimension `n`: the reduced run of
[`run_reduced`](@ref), with `perturb`, and its [`trajectory_metrics`](@ref) against the run,
around the background of the run's first snapshot, the rate and its status of [`reduced_rate`](@ref), the rate of [`fitted_rate`](@ref) of the
run, the snapshot `projection` error and the `offline` seconds. A reduced run that stops
leaves every metric empty.
"""
function data_row(method, run, n, r, offline, projection, mode; perturb = nothing)
    case = run.case
    reduced = run_reduced(r, case, run.nx, run.nv, run.snapshots[:, :, 1]; perturb)
    common = (method = method, kind = "data", case = case.name, alpha = case.α, n = n,
        state_size = n, status = reduced.status, projection_error = projection,
        offline_seconds = offline)
    reduced.status == "ok" || return baseline_row(; common...)

    m = trajectory_metrics(reduced.snapshots, run.snapshots, reduced,
        background(run.snapshots[:, :, 1]))
    rate, rate_status = reduced_rate(case.name, reduced.times, reduced.E1, mode.landau_window)
    rate_full = fitted_rate(case.name, run.times, run.E1, mode.landau_window)

    return baseline_row(; common..., l2_error_max = m.l2_error_max,
        l2_errors = join(m.l2_errors, ';'), max_l2_error_fluct = m.l2_error_fluct_max,
        l2_error_fluct = join(m.l2_errors_fluct, ';'), rate = something(rate, ""),
        rate_status = rate_status, rate_full_order = something(rate_full, ""),
        energy_drift = m.energy_drift,
        l1_drift = m.l1_drift, l2_drift = m.l2_drift, entropy_drift = m.entropy_drift,
        nonpositive_nodes = m.nonpositive_max)
end

"""
    reduced_step_times(runs_dir, specs)

Print the id and the thread count of this process, then for each `(file, method, n)` of `specs`
the wall time per step of [`reduced_step_time`](@ref) of that reduction of that run file in
`runs_dir`. [`reduced_costs`](@ref) calls it in a fresh process.
"""
function reduced_step_times(runs_dir::AbstractString, specs)
    println(getpid(), ' ', Threads.nthreads())
    methods = Dict(LINEAR_METHODS)
    for (file, method, n) in specs
        run = read_run(joinpath(runs_dir, file))
        S = run.snapshots
        r = methods[method](S, background(S[:, :, 1]), n)
        println(file, ' ', method, ' ', n, ' ', reduced_step_time(r, run.case, S[:, :, 1]))
    end
    return nothing
end

"""
    reduced_costs(results_dir, runs_dir, specs, full_order_seconds)

Measure [`reduced_step_times`](@ref) of `specs` in a fresh Julia process on the active project,
and write them with the machine's `uptime`, the id `pid` and the `threads` of the measuring
process and the full-order cost `full_order_seconds` per step to
`<results_dir>/linear_baselines_cost.toml`. It returns the seconds per step by
`(file, method, n)`.
"""
function reduced_costs(results_dir::AbstractString, runs_dir::AbstractString, specs,
        full_order_seconds::Real)
    code = "using VlasovReductionBenchmark; VlasovReductionBenchmark.reduced_step_times(" *
           "$(repr(runs_dir)), $(repr(collect(specs))))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    lines = readlines(cmd)
    pid, threads = parse.(Int, split(lines[end - length(specs)]))
    seconds = Dict{Tuple{String, String, Int}, Float64}()
    for line in lines[(end - length(specs) + 1):end]
        file, method, n, s = split(line)
        seconds[(file, method, parse(Int, n))] = parse(Float64, s)
    end
    uptime = Sys.isunix() ? readchomp(`uptime`) :
             @sprintf("up %.0f s, load averages %s", Sys.uptime(), join(Sys.loadavg(), " "))

    open(joinpath(results_dir, "linear_baselines_cost.toml"), "w") do io
        TOML.print(io,
            Dict("steps" => 100, "warmup" => 10, "dt" => TIMESTEP,
                "method" => "ImplicitMidpoint", "uptime" => uptime, "pid" => pid,
                "threads" => threads, "julia" => string(VERSION),
                "full_order_seconds_per_step" => full_order_seconds,
                "rows" => [Dict("file" => f, "method" => m, "n" => n,
                     "seconds_per_step" => s)
                 for ((f, m, n), s) in sort(collect(seconds))]))
    end

    return seconds
end

"The field `s` as a CSV field, quoted where it holds a comma or a quote."
function csv_field(s::AbstractString)
    occursin(r"[,\"]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

"""
    write_baselines(path, rows)

Write the `rows` of [`baseline_row`](@ref) to the CSV file `path`, with the header
[`BASELINE_COLUMNS`](@ref).
"""
function write_baselines(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(BASELINE_COLUMNS, ','))
        for row in rows
            println(io, join((csv_field(row[c]) for c in BASELINE_COLUMNS), ','))
        end
    end
    return path
end

"""
    run_linear_baselines(; runs_dir, results_dir, smoke = false, perturb = nothing)

POD and PSD Galerkin models of the full-order runs in `runs_dir`, as `scripts/full_order.jl`
writes them, with the grid and the fit window of [`FULL_MODE`](@ref), or [`SMOKE_MODE`](@ref)
with `smoke = true`, at the latent dimensions of [`BASELINE_SIZES`](@ref).

For each run of [`DATA_RUNS`](@ref) it asserts the POD optimality of
[`checked_projection_errors`](@ref), and for each method and `n` it builds the basis from that
run, integrates the reduced model over the run's time span and writes a [`data_row`](@ref). For
the amplitude sweep it builds each basis from the three training runs together, around the
background of the first, and writes the largest relative error at each test amplitude, relative
to ``f`` and relative to the fluctuation around the background of the test run. The online
cost of each data row that completes comes from [`reduced_costs`](@ref), as a ratio to
`<results_dir>/full_order_cost.toml`. It writes `<results_dir>/linear_baselines.csv`, prints the
number of rows and of failed runs, and returns the rows. A reduced run that stops is a row with
its status and empty metrics, and the other runs go on. `perturb` goes to every
[`run_reduced`](@ref); it lets a test break the runs.

It throws an error when the grid of `full_order_cost.toml` or of a run file is not the grid of
the mode.
"""
function run_linear_baselines(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false, perturb = nothing)
    mode = smoke ? SMOKE_MODE : FULL_MODE
    ns = smoke ? BASELINE_SIZES.smoke : BASELINE_SIZES.full
    full_cost = TOML.parsefile(joinpath(results_dir, "full_order_cost.toml"))
    (full_cost["nx"], full_cost["nv"]) == mode.grid || error(
        "the full-order cost is measured on $(full_cost["nx"]) × $(full_cost["nv"]), " *
        "not on the $(mode.grid[1]) × $(mode.grid[2]) grid of this mode")
    @printf("linear baselines on %d × %d%s, n ∈ %s\n",
        mode.grid..., smoke ? " (smoke)" : "",
        ns)
    flush(stdout)

    rows, files = Dict{String, String}[], String[]
    for file in DATA_RUNS
        run = read_run(joinpath(runs_dir, file))
        (run.nx, run.nv) == mode.grid || error(
            "$file holds a $(run.nx) × $(run.nv) run, not the $(mode.grid[1]) × " *
            "$(mode.grid[2]) grid of this mode")
        S = run.snapshots
        f0 = background(S[:, :, 1])
        errors = checked_projection_errors(run.case.name, S, f0, ns)
        for (method, build) in LINEAR_METHODS, (i, n) in enumerate(ns)

            timed = @timed build(S, f0, n)
            r, offline = timed.value, timed.time
            projection = method == "POD" ? errors.pod[i] : errors.psd[i]
            push!(rows, data_row(method, run, n, r, offline, projection, mode; perturb))
            push!(files, file)
            @printf("  %s, %s, n = %d: %s\n", method, file, n, rows[end]["status"])
            flush(stdout)
        end
    end

    train = [read_run(joinpath(runs_dir, run_file(linear_landau(α))))
             for α in TRAIN_AMPLITUDES]
    test = [read_run(joinpath(runs_dir, run_file(linear_landau(α))))
            for α in TEST_AMPLITUDES]
    S = cat((r.snapshots for r in train)...; dims = 3)
    f0 = background(train[1].snapshots[:, :, 1])
    for (method, build) in LINEAR_METHODS, n in ns

        timed = @timed build(S, f0, n)
        r, offline = timed.value, timed.time
        for run in test
            reduced = run_reduced(r, run.case, run.nx, run.nv, run.snapshots[:, :, 1];
                perturb)
            common = (method = method, kind = "sweep", case = run.case.name,
                alpha = run.case.α, n = n, state_size = n, status = reduced.status,
                offline_seconds = offline)
            reduced.status == "ok" || (push!(rows, baseline_row(; common...)); continue)
            f0_run = background(run.snapshots[:, :, 1])
            push!(rows,
                baseline_row(; common...,
                    l2_error_max = maximum(relative_l2_errors(reduced.snapshots,
                        run.snapshots)),
                    max_l2_error_fluct = maximum(relative_l2_errors(reduced.snapshots,
                        run.snapshots, f0_run))))
        end
    end

    specs = [(files[i], rows[i]["method"], parse(Int, rows[i]["n"]))
             for i in eachindex(files) if rows[i]["status"] == "ok"]
    seconds = reduced_costs(results_dir, runs_dir, specs, full_cost["seconds_per_step"])
    for (i, file) in enumerate(files)
        key = (file, rows[i]["method"], parse(Int, rows[i]["n"]))
        haskey(seconds, key) || continue
        rows[i]["online_seconds_per_step"] = string(seconds[key])
        rows[i]["online_ratio"] = string(seconds[key] / full_cost["seconds_per_step"])
    end

    path = write_baselines(joinpath(results_dir, "linear_baselines.csv"), rows)
    nfailed = count(r -> r["status"] != "ok", rows)
    @printf("linear baselines: %d rows, %d failed, written to %s\n", length(rows), nfailed,
        path)
    flush(stdout)

    return rows
end
