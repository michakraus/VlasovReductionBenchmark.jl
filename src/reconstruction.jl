"The width ``σ`` of the bump in ``x`` of the advection probe."
const ADVECTION_WIDTH = 1.0

"The velocity ``v_0`` of each beam of [`two_stream_beams_profile`](@ref)."
const BEAM_VELOCITY = 1.5

"The thermal width ``v_t`` of each beam of [`two_stream_beams_profile`](@ref)."
const BEAM_WIDTH = 0.5

"""
A test snapshot whose fluctuation ``g`` has ``\\|g\\| ≤`` `FLUCTUATION_FLOOR` ``\\|f\\|`` has no
fluctuation to relate an error to.
"""
const FLUCTUATION_FLOOR = 1e-14

"The relative tolerance within which the per-snapshot errors reproduce the aggregate error."
const AGGREGATE_RTOL = 1e-12

"The two splits: train and test on every snapshot, or train on ``t ≤ T/2`` and test on the rest."
const RECONSTRUCTION_SPLITS = ("in-sample", "extrapolation")

"""
The methods of the reconstruction probes: the two linear reductions, the two networks, and the
truncated SVD of each test snapshot, which trains on nothing.
"""
const RECONSTRUCTION_METHODS = ("POD", "PSD", "untied", "weight-tied", "rank-r")

"The columns of `reconstruction.csv`, in order."
const RECONSTRUCTION_COLUMNS = ("dataset", "split", "method", "n", "seed", "state_size",
    "status", "error", "max_snapshot_error", "snapshot_errors", "train_snapshots",
    "test_snapshots", "offline_seconds", "loss_first", "loss_last")

"The columns of `singular_values.csv`, in order."
const SINGULAR_VALUE_COLUMNS = ("dataset", "snapshots", "modes_99", "modes_9999",
    "singular_values")

"""
    reconstruction_mode(smoke)

The settings of the full mode of the reconstruction probes, or of the smoke mode with
`smoke = true`: the `grid` `(nx, nv)`, the latent dimensions `ns`, the training `steps`, the
`seeds` of each network, the end `long_tend` of the long runs, and the `advection_fractions` of
the recurrence time that the two advection datasets span.
"""
function reconstruction_mode(smoke::Bool)
    fractions = (advection_half = 0.5, advection_full = 1.0)
    return smoke ?
           (grid = SMOKE_MODE.grid, ns = BASELINE_SIZES.smoke, steps = 100,
        seeds = (untied = (42,), weight_tied = (42,)), long_tend = 40.0,
        advection_fractions = fractions) :
           (grid = FULL_MODE.grid, ns = BASELINE_SIZES.full, steps = 2000,
        seeds = (untied = (42, 43, 44), weight_tied = (42,)), long_tend = 100.0,
        advection_fractions = fractions)
end

"""
    method_seeds(mode, method)

The seeds of `method` under the [`reconstruction_mode`](@ref) `mode`: those of the network for
`"untied"` and `"weight-tied"`, and the one entry `nothing` for a method that draws no random
numbers.
"""
function method_seeds(mode, method::AbstractString)
    method == "untied" && return mode.seeds.untied
    method == "weight-tied" && return mode.seeds.weight_tied
    return (nothing,)
end

"""
    two_stream_beams_profile(v)

The two-beam profile ``\\frac12 [M(v - v_0) + M(v + v_0)]`` of unit density, with the Maxwellian
``M(v) = e^{-v^2/(2 v_t^2)} / (\\sqrt{2π}\\, v_t)`` of the thermal width `BEAM_WIDTH` ``= v_t``
and the beam velocity `BEAM_VELOCITY` ``= v_0``.
"""
function two_stream_beams_profile(v)
    beam(u) = exp(-u^2 / (2 * BEAM_WIDTH^2)) / (sqrt(2π) * BEAM_WIDTH)
    return (beam(v - BEAM_VELOCITY) + beam(v + BEAM_VELOCITY)) / 2
end

"""
    long_run_cases(tend)

The three long runs, each to `tend`: `two_stream_long` with [`two_stream_profile`](@ref),
`two_stream_beams_long` with [`two_stream_beams_profile`](@ref), both with the amplitude and the
wave number of [`two_stream`](@ref), and `nonlinear_landau_long` with those of
[`nonlinear_landau`](@ref).
"""
function long_run_cases(tend::Real)
    ts, nl = two_stream(), nonlinear_landau()
    return (VlasovCase("two_stream_long", two_stream_profile, ts.α, ts.k, tend),
        VlasovCase("two_stream_beams_long", two_stream_beams_profile, ts.α, ts.k, tend),
        VlasovCase("nonlinear_landau_long", nl.profile, nl.α, nl.k, tend))
end

"""
    recurrence_time(nv, k)

The recurrence time ``T_R = 2π / (k\\, Δv)`` of free streaming at the wave number `k` on the `nv`
velocity nodes of [`grid`](@ref), with ``Δv = 12 / (n_v - 1)``.
"""
recurrence_time(nv::Integer, k::Real) = 2π / (k * step(range(VDOMAIN...; length = nv)))

"""
    advection_times(nv, k; fraction)

The ``m = \\operatorname{round}(T / 0.5) + 1`` evenly spaced times of ``[0, T]``, both ends
included, with ``T`` = `fraction` times the [`recurrence_time`](@ref) of the `nv` velocity nodes
at the wave number `k`.
"""
function advection_times(nv::Integer, k::Real; fraction::Real)
    T = fraction * recurrence_time(nv, k)
    return collect(range(0.0, T; length = round(Int, T / SNAPSHOT_INTERVAL) + 1))
end

"""
    advection_profile(x, v, t; σ, x0, L)

The passive advection ``f(x, v, t) = M(v)\\, g(ξ)`` of a bump of width `σ` at `x0` on the periodic
interval ``[0, L)``, with the Maxwellian ``M`` of [`maxwellian`](@ref),
``ξ = \\operatorname{mod}(x - vt - x_0 + L/2, L) - L/2`` and
``g(ξ) = \\sum_{m=-1}^{1} e^{-(ξ - mL)^2 / (2σ^2)}``. It solves
``\\partial_t f + v\\, \\partial_x f = 0`` without an electric field.
"""
function advection_profile(x, v, t; σ, x0, L)
    ξ = mod(x - v * t - x0 + L / 2, L) - L / 2
    return maxwellian(v) * sum(exp(-(ξ - m * L)^2 / (2σ^2)) for m in -1:1)
end

"""
    advection_snapshots(nx, nv, times; σ = ADVECTION_WIDTH, x0 = L/2)

The `nx × nv × length(times)` array of [`advection_profile`](@ref) at the nodes of the
[`grid`](@ref) of the cases, all at ``k = 0.5`` with ``L = 4π``, at each of the `times`.
"""
function advection_snapshots(nx::Integer, nv::Integer, times;
        σ::Real = ADVECTION_WIDTH, x0::Real = π / linear_landau().k)
    case = linear_landau()
    x, v = grid(case, nx, nv)
    L = 2π / case.k
    return [advection_profile(x[i], v[j], t; σ, x0, L) for i in 1:nx, j in 1:nv, t in times]
end

"""
    time_split(times, T)

The indices `(train = …, test = …)` of the `times` with ``t ≤ T/2`` and with ``t > T/2``; the
snapshot at ``t = T/2`` trains.
"""
function time_split(times, T::Real)
    return (train = findall(≤(T / 2), times), test = findall(>(T / 2), times))
end

"""
    split_indices(times, split)

The training and the test indices of the snapshots at `times` under `split`: every snapshot in
both for `"in-sample"`, and the [`time_split`](@ref) over ``[0, T]``, ``T`` the last time, for
`"extrapolation"`.
"""
function split_indices(times, split::AbstractString)
    split == "in-sample" && return (train = collect(eachindex(times)),
        test = collect(eachindex(times)))
    split == "extrapolation" && return time_split(times, last(times))
    throw(ArgumentError("the split is \"in-sample\" or \"extrapolation\", got \"$split\""))
end

"""
    rank_error(g, r)

The relative error ``\\|g - [g]_r\\|_F / \\|g\\|_F`` of the truncated SVD ``[g]_r`` of rank `r`
of the matrix `g`.
"""
function rank_error(g::AbstractMatrix, r::Integer)
    F = svd(g)
    gr = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    return norm(g - gr) / norm(g)
end

"""
    reconstruction_errors(dataset, r, S)
    reconstruction_errors(dataset, rank, S, f0)

The errors of the `nx × nv × m` test snapshots `S` of `dataset`: the aggregate `error`
``\\|G - R(G)\\|_F / \\|G\\|_F`` of their fluctuations ``G``, the `snapshot_errors`
``\\|g_s - R(g_s)\\| / \\|g_s\\|`` and their maximum `max_snapshot_error`.

With a `LinearReduction` or an `AutoencoderReduction` `r`, ``R`` is its reconstruction around
`r.f0`, each error is its [`relative_projection_error`](@ref), and the aggregate comes from that
function on all of `S`; it throws an error when the per-snapshot errors do not reproduce it to
[`AGGREGATE_RTOL`](@ref). With an integer `rank`, ``R`` is the truncated SVD of each
`nx × nv` fluctuation around `f0` ([`rank_error`](@ref)), and the aggregate comes from the
per-snapshot errors.

It throws an error naming `dataset` and the snapshot when a test snapshot has no fluctuation,
``\\|g_s\\| ≤`` [`FLUCTUATION_FLOOR`](@ref) ``\\|f_s\\|``.
"""
function reconstruction_errors(dataset::AbstractString, r, S::AbstractArray{<:Any, 3})
    norms = fluctuation_norms(dataset, S, r.f0)
    errors = [relative_projection_error(r, S[:, :, s:s]) for s in axes(S, 3)]
    aggregate = relative_projection_error(r, S)
    combined = combined_error(errors, norms)
    abs(combined - aggregate) ≤ AGGREGATE_RTOL * aggregate || error(
        "$dataset: the per-snapshot errors give the aggregate $combined, not the $aggregate " *
        "of relative_projection_error")
    return (
        error = aggregate, snapshot_errors = errors, max_snapshot_error = maximum(errors))
end
function reconstruction_errors(dataset::AbstractString, rank::Integer,
        S::AbstractArray{<:Any, 3}, f0::AbstractVector)
    norms = fluctuation_norms(dataset, S, f0)
    G = fluctuation(S, f0)
    errors = [rank_error(G[:, :, s], rank) for s in axes(S, 3)]
    return (error = combined_error(errors, norms), snapshot_errors = errors,
        max_snapshot_error = maximum(errors))
end

"""
    fluctuation_norms(dataset, S, f0)

The norm of the fluctuation around `f0` of each snapshot of `S`, after the check of
[`reconstruction_errors`](@ref) that each has one.
"""
function fluctuation_norms(dataset, S, f0)
    return map(axes(S, 3)) do s
        g, f = norm(fluctuation(view(S, :, :, s), f0)), norm(view(S, :, :, s))
        g > FLUCTUATION_FLOOR * f || error(
            "$dataset: test snapshot $s has no fluctuation: $g ≤ $FLUCTUATION_FLOOR times $f")
        g
    end
end

"The aggregate relative error of the per-snapshot `errors` of fluctuations of the `norms`."
combined_error(errors, norms) = sqrt(sum(abs2, errors .* norms) / sum(abs2, norms))

"""
    reconstruction_row(dataset, S, times, split, method, n; seed = nothing, steps = 0,
        step_size = TRAINING_STEP_SIZE)

The row of `reconstruction.csv` of `method` at `n` on the `nx × nv × m` snapshots `S` of
`dataset` at `times` under `split` (see [`split_indices`](@ref)). The method is built, or trained
with `seed`, `steps` and `step_size`, on the training snapshots around the
[`background`](@ref) ``f_0`` of the first of them, and its [`reconstruction_errors`](@ref) are
measured on the test snapshots: POD and PSD by [`pod_reduction`](@ref) and
[`psd_reduction`](@ref), the networks by [`train_untied`](@ref) and [`train_weight_tied`](@ref),
and `"rank-r"` as the truncated SVD of rank `n` of each test fluctuation around ``f_0``, which
trains nothing. The state size of `"rank-r"` is [`dlra_state_size`](@ref), that of every other
method `n`. A training whose status is not `"ok"` gives a row with that status and empty errors.
"""
function reconstruction_row(dataset::AbstractString, S::AbstractArray{<:Any, 3}, times,
        split::AbstractString, method::AbstractString, n::Integer; seed = nothing,
        steps::Integer = 0, step_size::Real = TRAINING_STEP_SIZE)
    nx, nv, _ = size(S)
    idx = split_indices(times, split)
    train, test = S[:, :, idx.train], S[:, :, idx.test]
    f0 = background(train[:, :, 1])
    row = Dict{String, String}(c => "" for c in RECONSTRUCTION_COLUMNS)
    merge!(row,
        Dict("dataset" => dataset, "split" => split, "method" => method, "n" => string(n),
            "seed" => string(something(seed, "")), "status" => "ok",
            "train_snapshots" => string(method == "rank-r" ? 0 : length(idx.train)),
            "test_snapshots" => string(length(idx.test)),
            "state_size" => string(method == "rank-r" ? dlra_state_size(n, nx, nv) : n)))
    if method == "rank-r"
        errors = reconstruction_errors(dataset, n, test, f0)
    elseif method in ("POD", "PSD")
        timed = @timed (method == "POD" ? pod_reduction : psd_reduction)(train, f0, n)
        row["offline_seconds"] = string(timed.time)
        errors = reconstruction_errors(dataset, timed.value, test)
    elseif method in ("untied", "weight-tied")
        build = method == "untied" ? train_untied : train_weight_tied
        timed = @timed build(train, f0, n; seed, steps, step_size)
        training = timed.value.training
        merge!(row,
            Dict("offline_seconds" => string(timed.time), "status" => training.status,
                "loss_first" => string(training.first), "loss_last" =>
                    string(training.last)))
        training.status == "ok" || return row
        errors = reconstruction_errors(dataset, timed.value.reduction, test)
    else
        throw(ArgumentError("no reconstruction method \"$method\""))
    end
    merge!(row,
        Dict("error" => string(errors.error),
            "max_snapshot_error" => string(errors.max_snapshot_error),
            "snapshot_errors" => join(errors.snapshot_errors, ';')))

    return row
end

"""
    snapshot_singular_values(S)

The singular values of the `nx nv × m` matrix of the fluctuations of the `nx × nv × m` snapshots
`S` around the [`background`](@ref) of the first, divided by the largest.
"""
function snapshot_singular_values(S::AbstractArray{<:Any, 3})
    nx, nv, m = size(S)
    σ = svdvals(reshape(fluctuation(S, background(S[:, :, 1])), nx * nv, m))
    return σ ./ σ[1]
end

"""
    mode_count(σ, fraction)

The smallest number of the leading singular values `σ` whose squares reach `fraction` of
``\\sum_i σ_i^2``.
"""
function mode_count(σ::AbstractVector, fraction::Real)
    return findfirst(≥(fraction), cumsum(σ .^ 2) ./ sum(abs2, σ))
end

"""
    write_csv(path, columns, rows)

Write the `rows`, each a `Dict` of the `columns`, to the CSV file `path`, with the header
`columns` and each field as [`csv_field`](@ref) writes it.
"""
function write_csv(path::AbstractString, columns, rows)
    open(path, "w") do io
        println(io, join(columns, ','))
        for row in rows
            println(io, join((csv_field(row[c]) for c in columns), ','))
        end
    end
    return path
end

"""
    write_reconstruction_page(results_dir)

Write `<results_dir>/reconstruction.md` from `reconstruction.csv` and `singular_values.csv` in
`results_dir`, and return its path. For each dataset and split it gives a table of the
aggregate `error` per method and `n`: for a network the median over its seeds, with the range in
brackets, and below the methods the state size of `"rank-r"`. A section after the tables gives
the mode counts of each dataset. A row whose status is not `"ok"` enters no median; the page
lists every such row with its status. Every value comes from the two CSV files: each error with
the four significant digits of [`format_value`](@ref), each count and state size as written.
"""
function write_reconstruction_page(results_dir::AbstractString)
    rows = read_csv_rows(joinpath(results_dir, "reconstruction.csv"))
    modes = read_csv_rows(joinpath(results_dir, "singular_values.csv"))
    path = joinpath(results_dir, "reconstruction.md")
    datasets = unique(r["dataset"] for r in rows)
    open(path, "w") do io
        println(io, "# Reconstruction probes\n")
        println(io, "The relative error of the reconstruction of the test fluctuations, ",
            "per method and latent dimension; a network gives the median over its seeds and ",
            "the range in brackets. The rank-r reference is the truncated SVD of each test ",
            "snapshot, whose state size is that of a factored state.\n")
        for dataset in datasets, split in RECONSTRUCTION_SPLITS

            these = filter(r -> r["dataset"] == dataset && r["split"] == split, rows)
            isempty(these) && continue
            ns = sort(unique(parse(Int, r["n"]) for r in these))
            println(io, "## $dataset, $split\n")
            println(io, "| method | ", join(("n = $n" for n in ns), " | "), " |")
            println(io, "|", repeat(":--|", length(ns) + 1))
            for method in RECONSTRUCTION_METHODS
                any(r -> r["method"] == method, these) || continue
                cells = map(ns) do n
                    ok = filter(
                        r -> r["method"] == method && r["n"] == string(n) &&
                             r["status"] == "ok", these)
                    isempty(ok) && return "—"
                    e = [parse(Float64, r["error"]) for r in ok]
                    method in ("untied", "weight-tied") || return format_value(only(e))
                    "$(format_value(median_of(e))) [$(format_value(minimum(e))), " *
                    "$(format_value(maximum(e)))]"
                end
                println(io, "| $method | ", join(cells, " | "), " |")
            end
            if any(r -> r["method"] == "rank-r", these)
                sizes = map(ns) do n
                    r = filter(r -> r["method"] == "rank-r" && r["n"] == string(n), these)
                    isempty(r) ? "—" : first(r)["state_size"]
                end
                println(io, "| rank-r state size | ", join(sizes, " | "), " |")
            end
            println(io)
        end
        println(io, "## Mode counts\n")
        for m in modes
            println(
                io, "- $(m["dataset"]): `modes_99` = $(m["modes_99"]), `modes_9999` = ",
                "$(m["modes_9999"]) of $(m["snapshots"]) snapshots")
        end
        println(io)
        failed = filter(r -> r["status"] != "ok", rows)
        println(io, "## Failed rows\n")
        isempty(failed) && println(io, "No row failed.")
        # only a network trains, so every failed row has a seed
        for r in failed
            println(
                io, "- $(r["dataset"]), $(r["split"]), $(r["method"]), n = $(r["n"]), ",
                "seed $(r["seed"]): $(r["status"])")
        end
    end
    return path
end

"""
    reconstruction_datasets(mode, runs_dir)

The five datasets on the grid of `mode`, each as `(name, snapshots, times)`: `advection_half`
and `advection_full`, the [`advection_snapshots`](@ref) at the [`advection_times`](@ref) of the
two `mode.advection_fractions`, and the snapshots of the three [`long_run_cases`](@ref) to
`mode.long_tend`, each run by [`run_case`](@ref) with its checks and written to `runs_dir`.
"""
function reconstruction_datasets(mode, runs_dir::AbstractString)
    nx, nv = mode.grid
    k = linear_landau().k
    datasets = Any[]
    for (name, fraction) in pairs(mode.advection_fractions)
        times = advection_times(nv, k; fraction)
        push!(datasets, (string(name), advection_snapshots(nx, nv, times), times))
    end
    for case in long_run_cases(mode.long_tend)
        elapsed = @elapsed run = run_case(case, nx, nv; runs_dir)
        @printf("  %s: %d steps in %.1f s, wrote %s\n", case.name, length(run.times) - 1,
            elapsed, run_file(case))
        flush(stdout)
        push!(datasets, (case.name, run.snapshots, run.snapshot_times))
    end
    return datasets
end

"""
    run_reconstruction(; runs_dir, results_dir, smoke = false)

The reconstruction probes with the settings of [`reconstruction_mode`](@ref): the
[`reconstruction_datasets`](@ref), whose long runs go to `runs_dir`, and for each dataset, split,
method, `n` and seed of a network the [`reconstruction_row`](@ref), written to
`<results_dir>/reconstruction.csv`; the [`snapshot_singular_values`](@ref) of each dataset with
their [`mode_count`](@ref)s at 99 % and 99.99 % to `<results_dir>/singular_values.csv`; and the
page of [`write_reconstruction_page`](@ref). A failed training is a row with its status, and the
script goes on. It prints the number of rows and of failed rows, and returns the rows.
"""
function run_reconstruction(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false)
    mode = reconstruction_mode(smoke)
    @printf("reconstruction probes on %d × %d%s, n ∈ %s, %d steps\n", mode.grid...,
        smoke ? " (smoke)" : "", mode.ns, mode.steps)
    flush(stdout)
    mkpath(results_dir)
    datasets = reconstruction_datasets(mode, runs_dir)

    spectra = map(datasets) do (name, S, _)
        σ = snapshot_singular_values(S)
        Dict("dataset" => name, "snapshots" => string(size(S, 3)),
            "modes_99" => string(mode_count(σ, 0.99)),
            "modes_9999" => string(mode_count(σ, 0.9999)),
            "singular_values" => join(σ, ';'))
    end
    write_csv(joinpath(results_dir, "singular_values.csv"), SINGULAR_VALUE_COLUMNS, spectra)

    rows = Dict{String, String}[]
    for (name, S, times) in datasets, split in RECONSTRUCTION_SPLITS,
        method in RECONSTRUCTION_METHODS, n in mode.ns
        for seed in method_seeds(mode, method)
            row = reconstruction_row(name, S, times, split, method, n; seed,
                steps = mode.steps)
            push!(rows, row)
            @printf("  %s, %s, %s, n = %d%s: %s\n", name, split, method, n,
                seed === nothing ? "" : ", seed $seed", row["status"])
            flush(stdout)
        end
    end
    path = write_csv(joinpath(results_dir, "reconstruction.csv"), RECONSTRUCTION_COLUMNS, rows)
    write_reconstruction_page(results_dir)
    @printf("reconstruction probes: %d rows, %d failed, written to %s\n", length(rows),
        count(r -> r["status"] != "ok", rows), path)
    flush(stdout)

    return rows
end
