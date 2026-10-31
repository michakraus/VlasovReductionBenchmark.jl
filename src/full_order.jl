"The time step of every full-order run, with `RK4()`."
const TIMESTEP = 0.05

"The time between two stored snapshots of a run."
const SNAPSHOT_INTERVAL = 0.5

"The largest relative drift of the mass ``h_x h_v \\sum f`` that a run accepts."
const MASS_TOLERANCE = 1e-12

"The largest relative drift of the discrete energy that a run accepts."
const ENERGY_TOLERANCE = 1e-4

"""
The grid `(nx, nv)` of the data runs, the window of the linear Landau fit, and the relative
tolerances of the linear Landau and the two-stream rate checks.
"""
const FULL_MODE = (grid = (64, 128), landau_window = (0.0, 40.0), landau_rtol = 0.02,
    two_stream_rtol = 0.05)

"""
The settings of [`FULL_MODE`](@ref) for the smoke mode. On its coarse grid the recurrence spoils
a linear Landau fit over ``t ∈ [0, 40]``, so its window ends at ``t = 25``.
"""
const SMOKE_MODE = (grid = (32, 64), landau_window = (0.0, 25.0), landau_rtol = 0.10,
    two_stream_rtol = 0.10)

"The amplitudes of the linear Landau sweep that the training uses."
const TRAIN_AMPLITUDES = (0.005, 0.01, 0.02)

"The amplitudes of the linear Landau sweep that the training does not see."
const TEST_AMPLITUDES = (0.015, 0.03)

@doc raw"""
    diagnostics(q, G, k, x, v, hx, hv)

The diagnostics of the state `q = vec(f)` on the grid with nodes `x`, `v` and Green's matrix `G`:
the mass ``h_x h_v \sum f``, the discrete energy
``h_x h_v \sum f v^2/2 + (h_x/2)\, ρ^{\mathsf T} G ρ`` with ``ρ_i = h_v \sum_j f_{ij}``, its field
part ``(h_x/2)\, ρ^{\mathsf T} G ρ``, the first Fourier mode
``|E_1| = k\, (2/n_x)\, |\sum_i φ_i e^{-ikx_i}|`` of ``φ = Gρ``, the ``L^2`` norm
``(h_x h_v \sum f^2)^{1/2}``, the entropy ``-h_x h_v \sum_{f > 0} f \log f`` and the number of
nodes where ``f ≤ 0``.
"""
function diagnostics(q, G, k, x, v, hx, hv)
    nx, nv = length(x), length(v)
    f = reshape(q, nx, nv)
    ρ = hv .* vec(sum(f; dims = 2))
    φ = G * ρ
    field = hx / 2 * dot(ρ, φ)
    kinetic = hx * hv * sum(f[i, j] * v[j]^2 / 2 for j in 1:nv, i in 1:nx)
    E1 = k * (2 / nx) * abs(sum(φ[i] * exp(-im * k * x[i]) for i in 1:nx))
    entropy = -hx * hv * sum(fij * log(fij) for fij in q if fij > 0; init = 0.0)

    return (
        mass = hx * hv * sum(q), energy = kinetic + field, field_energy = field, E1 = E1,
        l2 = sqrt(hx * hv * sum(abs2, q)), entropy = entropy,
        nonpositive = count(≤(0), q))
end

"""
    run_case(case, nx, nv; runs_dir = nothing, label = "none", perturb = nothing)

Run `case` on the `nx × nv` grid with `RK4()` and step [`TIMESTEP`](@ref), from ``t = 0`` to
`case.tend`. It keeps the state every [`SNAPSHOT_INTERVAL`](@ref), and every step the
electric-field energy, ``|E_1|``, and the relative drifts of the mass, the energy, the ``L^2``
norm and the entropy, with the count of nodes where ``f ≤ 0`` (see [`diagnostics`](@ref)).

After each step it throws an error naming the case, the amplitude and the step when the state
holds a `NaN` or an `Inf`, when the mass drifts by more than [`MASS_TOLERANCE`](@ref), or when
the energy drifts by more than [`ENERGY_TOLERANCE`](@ref), relative to their initial values.
`perturb(step, q)`, where given, is called on the state after each step and before these
checks; it lets a test break a run.

With `runs_dir`, a run that completes is written there by [`write_run`](@ref), with the
train/test `label`, and a run that throws writes nothing. It returns the run as a `NamedTuple`.
"""
function run_case(case::VlasovCase, nx::Integer, nv::Integer;
        runs_dir = nothing, label::AbstractString = "none", perturb = nothing)
    model = vlasov_model(case, nx, nv)
    dist = model.distribution
    x, v = dist.x, dist.v
    hx, hv = step(x), step(v)
    G = model.work

    nsteps = round(Int, case.tend / TIMESTEP)
    stride = round(Int, SNAPSHOT_INTERVAL / TIMESTEP)
    int = VlasovMethods.GeometricIntegrator(model, (0.0, case.tend), TIMESTEP, RK4())
    solstep = solutionstep(int.integrator, initialstate(int.equation))

    d₀ = diagnostics(solstep.q, G, case.k, x, v, hx, hv)
    times = [s * TIMESTEP for s in 0:nsteps]
    snapshot_steps = 0:stride:nsteps
    snapshots = zeros(nx, nv, length(snapshot_steps))
    series = (field_energy = zeros(nsteps + 1), E1 = zeros(nsteps + 1),
        mass_drift = zeros(nsteps + 1), energy_drift = zeros(nsteps + 1),
        l2_drift = zeros(nsteps + 1), entropy_drift = zeros(nsteps + 1),
        nonpositive_nodes = zeros(Int, nsteps + 1))

    name = "$(case.name), α = $(case.α)"
    for s in 0:nsteps
        if s > 0
            reset!(solstep, times[s + 1])
            integrate!(solstep, int.integrator)
            perturb === nothing || perturb(s, solstep.q)
        end
        q = solstep.q
        all(isfinite, q) ||
            error("$name: the state holds a non-finite value after step $s")
        d = diagnostics(q, G, case.k, x, v, hx, hv)
        mass_drift = (d.mass - d₀.mass) / d₀.mass
        energy_drift = (d.energy - d₀.energy) / d₀.energy
        abs(mass_drift) ≤ MASS_TOLERANCE || error(
            "$name: the mass drifted by $(abs(mass_drift)) relative at step $s, " *
            "above $MASS_TOLERANCE")
        abs(energy_drift) ≤ ENERGY_TOLERANCE || error(
            "$name: the energy drifted by $(abs(energy_drift)) relative at step $s, " *
            "above $ENERGY_TOLERANCE")

        series.field_energy[s + 1] = d.field_energy
        series.E1[s + 1] = d.E1
        series.mass_drift[s + 1] = mass_drift
        series.energy_drift[s + 1] = energy_drift
        series.l2_drift[s + 1] = (d.l2 - d₀.l2) / d₀.l2
        series.entropy_drift[s + 1] = (d.entropy - d₀.entropy) / abs(d₀.entropy)
        series.nonpositive_nodes[s + 1] = d.nonpositive
        s % stride == 0 && (snapshots[:, :, s ÷ stride + 1] .= reshape(q, nx, nv))
    end

    run = (case = case, nx = nx, nv = nv, times = times,
        snapshot_times = times[snapshot_steps .+ 1], snapshots = snapshots,
        state = copy(solstep.q), series...)
    runs_dir === nothing || write_run(runs_dir, run; label)

    return run
end

"""
    run_file(case)

The name of the run file of `case`, `"<name>_alpha<α>.h5"`.
"""
run_file(case::VlasovCase) = "$(case.name)_alpha$(case.α).h5"

"""
    write_run(runs_dir, run; label)

Write `run`, as [`run_case`](@ref) returns it, to the HDF5 file [`run_file`](@ref) in `runs_dir`,
over any file of that name. The datasets are `snapshots` (``n_x × n_v × n_t``), `snapshot_times`,
`times` (every step), `field_energy`, `E1`, `mass_drift`, `energy_drift`, `l2_drift`,
`entropy_drift` and `nonpositive_nodes`; the attributes are the case, `alpha`, `k`, `tend`, `dt`,
`nx`, `nv`, the domains and the train/test `label`.
"""
function write_run(runs_dir::AbstractString, run; label::AbstractString)
    mkpath(runs_dir)
    path = joinpath(runs_dir, run_file(run.case))
    h5open(path, "w") do file
        for name in (:snapshots, :snapshot_times, :times, :field_energy, :E1, :mass_drift,
            :energy_drift, :l2_drift, :entropy_drift, :nonpositive_nodes)
            file[string(name)] = getproperty(run, name)
        end
        a = attrs(file)
        a["case"] = run.case.name
        a["alpha"] = run.case.α
        a["k"] = run.case.k
        a["tend"] = run.case.tend
        a["dt"] = TIMESTEP
        a["nx"] = run.nx
        a["nv"] = run.nv
        a["xdomain"] = [0.0, 2π / run.case.k]
        a["vdomain"] = collect(VDOMAIN)
        a["label"] = label
    end

    return path
end

"""
    landau_rate_fit(t, y, window)

The damping rate and the frequency of a damped oscillation from `y = log|E₁|` at the times `t`:
the slope `γ` of the least-squares line through `y` at its local maxima, ``y_{n-1} < y_n ≥
y_{n+1}``, with ``t_n`` in `window`, and the frequency `ω` ``= π / \\overline{Δt}`` from the mean
spacing of those maxima, two per period of ``E_1``. It throws an error for fewer than four maxima.
"""
function landau_rate_fit(t, y, window)
    peaks = [n
             for n in 2:(length(t) - 1)
             if window[1] ≤ t[n] ≤ window[2] && y[n - 1] < y[n] ≥ y[n + 1]]
    length(peaks) ≥ 4 ||
        error("the Landau fit found $(length(peaks)) maxima of log|E₁| in " *
              "[$(window[1]), $(window[2])]; it needs at least four")
    tp = t[peaks]

    return (γ = slope(tp, y[peaks]), ω = π / (sum(diff(tp)) / (length(tp) - 1)),
        npeaks = length(peaks))
end

"""
    growth_rate_fit(t, y, window)

The slope of the least-squares line through `y = log|E₁|` at every time `t` in `window`.
"""
function growth_rate_fit(t, y, window)
    w = [window[1] ≤ tn ≤ window[2] for tn in t]

    return slope(t[w], y[w])
end

"The slope of the least-squares line through the points `(t, y)`."
function slope(t, y)
    tm, ym = sum(t) / length(t), sum(y) / length(y)

    return sum((t .- tm) .* (y .- ym)) / sum(abs2, t .- tm)
end

"""
    check_rate(name, fitted, oracle, rtol)

Print `fitted` and `oracle`, and throw an error when they differ by more than `rtol` relative to
`oracle`.
"""
function check_rate(name, fitted, oracle, rtol)
    err = abs(fitted - oracle) / abs(oracle)
    @printf("rate check %s: fitted %.5f, oracle %.5f, relative error %.4f\n", name, fitted,
        oracle, err)
    flush(stdout)
    err ≤ rtol ||
        error(@sprintf("rate check %s: fitted %.5f, oracle %.5f, relative error %.4f, above %s",
            name, fitted, oracle, err, rtol))

    return nothing
end

"""
    measure_step_time(nx, nv; steps = 100, warmup = 20)

The wall time in seconds of one `RK4()` step of linear Landau damping on the `nx × nv` grid,
averaged over `steps` steps after a warm-up run of `warmup` steps.
"""
function measure_step_time(nx::Integer, nv::Integer; steps::Integer = 100,
        warmup::Integer = 20)
    case = linear_landau()
    int = VlasovMethods.GeometricIntegrator(vlasov_model(case, nx, nv),
        (0.0, (warmup + steps) * TIMESTEP), TIMESTEP, RK4())
    solstep = solutionstep(int.integrator, initialstate(int.equation))
    for s in 1:warmup
        reset!(solstep, s * TIMESTEP)
        integrate!(solstep, int.integrator)
    end
    elapsed = @elapsed for s in (warmup + 1):(warmup + steps)
        reset!(solstep, s * TIMESTEP)
        integrate!(solstep, int.integrator)
    end

    return elapsed / steps
end

"""
    full_order_cost(results_dir, nx, nv; steps = 100)

Measure [`measure_step_time`](@ref) in a fresh Julia process on the active project, and write it
with the machine's `uptime` and the id `pid` of the measuring process to
`<results_dir>/full_order_cost.toml`.
"""
function full_order_cost(results_dir::AbstractString, nx::Integer, nv::Integer;
        steps::Integer = 100)
    code = "using VlasovReductionBenchmark; println(getpid(), ' ', " *
           "VlasovReductionBenchmark.measure_step_time($nx, $nv; steps = $steps))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    pid, seconds = split(last(readlines(cmd)))
    pid, seconds = parse(Int, pid), parse(Float64, seconds)
    uptime = Sys.isunix() ? readchomp(`uptime`) :
             @sprintf("up %.0f s, load averages %s", Sys.uptime(), join(Sys.loadavg(), " "))

    mkpath(results_dir)
    path = joinpath(results_dir, "full_order_cost.toml")
    open(path, "w") do io
        TOML.print(io,
            Dict("nx" => nx, "nv" => nv, "steps" => steps, "dt" => TIMESTEP,
                "method" => "RK4", "seconds_per_step" => seconds, "uptime" => uptime,
                "julia" => string(VERSION), "threads" => Threads.nthreads(), "pid" => pid))
    end
    @printf("full-order cost on %d × %d: %.3e s per step (%s)\n", nx, nv, seconds, uptime)
    flush(stdout)

    return path
end

"""
    run_full_order(; runs_dir, results_dir, smoke = false)

The full-order data of the benchmark, with the settings of [`FULL_MODE`](@ref), or
[`SMOKE_MODE`](@ref) with `smoke = true`: linear Landau damping at the amplitudes of
[`TRAIN_AMPLITUDES`](@ref) and [`TEST_AMPLITUDES`](@ref), nonlinear Landau damping and the
two-stream instability, each written to `runs_dir` by [`run_case`](@ref).

It then checks the rates against the oracle of [`landau_root`](@ref) and
[`two_stream_root`](@ref): the damping rate and the frequency of the linear Landau run at
``α = 0.01``, fitted by [`landau_rate_fit`](@ref) over ``t ∈ [0, 40]`` to 2 % (smoke:
``t ∈ [0, 25]``, 10 %), and the growth rate of a two-stream run at ``α = 10^{-3}`` that writes
no file, fitted by [`growth_rate_fit`](@ref) over ``t ∈ [5, 25]`` to 5 % (smoke: 10 %). Last, it
writes the cost per step to `results_dir` by [`full_order_cost`](@ref).
"""
function run_full_order(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false)
    mode = smoke ? SMOKE_MODE : FULL_MODE
    nx, nv = mode.grid
    @printf("full-order runs on %d × %d%s\n", nx, nv, smoke ? " (smoke)" : "")
    flush(stdout)

    cases = vcat([linear_landau(α) => "train" for α in TRAIN_AMPLITUDES],
        [linear_landau(α) => "test" for α in TEST_AMPLITUDES],
        [nonlinear_landau() => "none", two_stream() => "none"])
    runs = map(cases) do (case, label)
        elapsed = @elapsed run = run_case(case, nx, nv; runs_dir, label)
        @printf("  %s, α = %g, %s: %d steps in %.1f s, wrote %s\n",
            case.name, case.α, label,
            length(run.times) - 1, elapsed, run_file(case))
        flush(stdout)
        run
    end

    oracle = landau_root(0.5)
    landau = runs[findfirst(r -> r.case.α == 0.01, runs)]
    fit = landau_rate_fit(landau.times, log.(landau.E1), mode.landau_window)
    check_rate("linear Landau γ", fit.γ, imag(oracle), mode.landau_rtol)
    check_rate("linear Landau ω", fit.ω, real(oracle), mode.landau_rtol)

    check = run_case(VlasovCase("two_stream", two_stream_profile, 1e-3, 0.5, 25.0), nx, nv)
    check_rate("two-stream growth rate",
        growth_rate_fit(check.times, log.(check.E1), (5.0, 25.0)),
        imag(two_stream_root(0.5)), mode.two_stream_rtol)

    full_order_cost(results_dir, nx, nv)

    return nothing
end
