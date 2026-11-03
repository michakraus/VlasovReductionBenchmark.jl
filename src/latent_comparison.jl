"The columns of `latent_dynamics.csv`, in order."
const LATENT_COLUMNS = ("dataset", "n", "reduction", "dynamics", "seed", "status",
    "error_in_sample", "error_extrapolation", "max_energy_drift", "max_l1_drift",
    "max_l2_drift", "max_mass_drift", "pairing_condition", "k0_fraction", "offline_seconds",
    "loss_first", "loss_last")

"The columns of a row of `latent_dynamics.csv` that hold an error or a drift."
const LATENT_METRIC_COLUMNS = (
    "error_in_sample", "error_extrapolation", "max_energy_drift",
    "max_l1_drift", "max_l2_drift", "max_mass_drift")

"The reductions of `latent_dynamics.csv`, in the order of the page."
const LATENT_REDUCTIONS = ("psd", "psd_nonzero", "dual_psd", "weight_tied")

"The reduced dynamics of `latent_dynamics.csv`, in the order of the page."
const LATENT_DYNAMICS = ("sympnet", "galerkin")

"""
    latent_comparison_mode(smoke)

The settings of the full mode of the latent-dynamics comparison, or of the smoke mode with
`smoke = true`: the `grid` `(nx, nv)` and the end `long_tend` of the long runs, the latent
dimensions `ns`, the `seeds`, the `epochs`, `batch_size` and `n_layers` of each SympNet, the
training steps `autoencoder_steps` of the weight-tied network, and `m_train`, the number of
snapshots with ``t ≤ t_\\mathrm{end}/2``, on which every reduction and every SympNet trains.
"""
function latent_comparison_mode(smoke::Bool)
    long_tend = smoke ? 40.0 : 100.0
    return (grid = smoke ? SMOKE_MODE.grid : FULL_MODE.grid, long_tend = long_tend,
        ns = smoke ? (4,) : (4, 8, 16, 32), seeds = smoke ? (42,) : (42, 43, 44),
        epochs = smoke ? 20 : 2000, batch_size = 32, n_layers = 4,
        autoencoder_steps = smoke ? 100 : 2000,
        m_train = round(Int, long_tend / 2 / SNAPSHOT_INTERVAL) + 1)
end

# `latent_run` reads the grid and the background of a reduction from its fields; a dual PSD
# holds them in its reduction `r0`.
function Base.getproperty(r::DualReduction, name::Symbol)
    name in (:nx, :nv, :f0) && return getfield(getfield(r, :r0), name)
    return getfield(r, name)
end

"""
    k0_fraction(S, f0)

The ratio ``\\|\\bar g\\|_F / \\|g\\|_F`` of the x-average part ``\\bar g`` of the fluctuations
``g`` of the `nx × nv × m` snapshots `S` around `f0` to the fluctuations themselves: the share of
their ``k = 0`` content.
"""
function k0_fraction(S::AbstractArray{<:Any, 3}, f0::AbstractVector)
    G = fluctuation(S, f0)
    average = sum(G; dims = 1) ./ size(G, 1)
    return norm(average .+ zero(G)) / norm(G)
end

"The root mean square of the `errors` at the indices `idx`."
rms_error(errors, idx) = sqrt(sum(abs2, errors[idx]) / length(idx))

"""
    latent_row(common, reduction, dynamics, seed, status)

A row of [`LATENT_COLUMNS`](@ref) with the fields of `common`, the `reduction`, the `dynamics`,
the `seed` (empty for `nothing`) and the `status`, and every other field empty.
"""
function latent_row(common, reduction, dynamics, seed, status)
    return merge!(Dict{String, String}(c => "" for c in LATENT_COLUMNS), common,
        Dict("reduction" => reduction, "dynamics" => dynamics,
            "seed" => string(something(seed, "")), "status" => status))
end

"""
    nan_row(common, reduction, dynamics, seed, status)

The [`latent_row`](@ref) of a row with no model to measure: `NaN` in every error, drift and time
column, and in the loss columns of a SympNet row.
"""
function nan_row(common, reduction, dynamics, seed, status)
    row = latent_row(common, reduction, dynamics, seed, status)
    for c in (LATENT_METRIC_COLUMNS..., "offline_seconds")
        row[c] = "NaN"
    end
    if dynamics == "sympnet"
        row["loss_first"] = row["loss_last"] = "NaN"
    end
    return row
end

"""
    sympnet_row(common, reduction, r, case, S, seed, mode; seconds)

The SympNet row of the reduction `r`, named `reduction`, on the `nx × nv × m` snapshots `S` of
`case`: [`train_latent_sympnet`](@ref) with `seed` and the `epochs`, `n_layers` and `batch_size`
of `mode` on the latent snapshots of `S[:, :, 1:mode.m_train]`, and [`latent_run`](@ref) of the
network over all of `S`. The row holds the status, the errors and the drifts of `latent_run`,
the first and the last loss of the training, and the time `seconds` of the build of `r` plus
the time of the training as `offline_seconds`.
"""
function sympnet_row(
        common, reduction, r, case::VlasovCase, S::AbstractArray{<:Any, 3}, seed,
        mode; seconds::Real)
    Z = latent_snapshots(r, S[:, :, 1:(mode.m_train)])
    trained = @timed train_latent_sympnet(Z, mode.m_train; epochs = mode.epochs, seed,
        n_layers = mode.n_layers, batch_size = mode.batch_size)
    history = trained.value.history
    run = latent_run(r, case, S, trained.value.network, mode.m_train)
    return merge!(latent_row(common, reduction, "sympnet", seed, run.status),
        Dict("error_in_sample" => string(run.l2_error_in_sample),
            "error_extrapolation" => string(run.l2_error_extrapolation),
            "max_energy_drift" => string(run.energy_drift),
            "max_l1_drift" => string(run.l1_drift), "max_l2_drift" => string(run.l2_drift),
            "max_mass_drift" => string(run.mass_drift),
            "offline_seconds" => string(seconds + trained.time),
            "loss_first" => string(first(history)), "loss_last" => string(last(history))))
end

"""
    galerkin_row(common, reduction, r, case, S, seed, m_train; seconds)

The Galerkin row of the reduction `r`, named `reduction`, on the `nx × nv × m` snapshots `S` of
`case`: [`run_reduced`](@ref) of `r` from `S[:, :, 1]`, with its status verbatim. Its errors are
the root mean square of the [`relative_l2_errors`](@ref) of the decoded snapshots relative to the
fluctuation around `r.f0` over the snapshots `1:m_train` and `m_train+1:m`, and its drifts the
[`max_drift`](@ref) of the energy, ``L^1``, ``L^2`` and mass of the decoded snapshots, every
[`SNAPSHOT_INTERVAL`](@ref) from ``t = 0``. A run whose status is not `"ok"` has `NaN` in each.
`offline_seconds` is `seconds`, the time of the build of `r`.
"""
function galerkin_row(
        common, reduction, r, case::VlasovCase, S::AbstractArray{<:Any, 3}, seed,
        m_train::Integer; seconds::Real)
    nx, nv, m = size(S)
    run = run_reduced(r, case, nx, nv, S[:, :, 1])
    row = latent_row(common, reduction, "galerkin", seed, run.status)
    row["offline_seconds"] = string(seconds)
    if run.status != "ok"
        foreach(c -> row[c] = "NaN", LATENT_METRIC_COLUMNS)
        return row
    end
    errors = relative_l2_errors(run.snapshots, S, r.f0)
    stride = round(Int, SNAPSHOT_INTERVAL / TIMESTEP)
    drift(series) = string(max_drift(series[1:stride:end]))
    return merge!(row,
        Dict("error_in_sample" => string(rms_error(errors, 1:m_train)),
            "error_extrapolation" => string(rms_error(errors, (m_train + 1):m)),
            "max_energy_drift" => drift(run.energy), "max_l1_drift" => drift(run.l1),
            "max_l2_drift" => drift(run.l2), "max_mass_drift" => drift(run.mass)))
end

"""
    dual_rows(common, case, S, f0, n, mode; build = dual_psd_reduction)

The SympNet rows of dual PSD on the `nx × nv × m` snapshots `S` of `case`, one per seed of
`mode`: `build(S[:, :, 1:mode.m_train], f0, n, case.k)` and the [`sympnet_row`](@ref) of the
result, with its [`pairing_condition`](@ref). Dual PSD has no Galerkin row: its subspace is that
of [`psd_nonzero_reduction`](@ref).

Where `build` throws an `ArgumentError`, each row has the status `singular`, `NaN` in every
error, drift, loss and time column, and as `pairing_condition` the condition number of the
odd–even block of [`latent_pairing`](@ref) of `psd_nonzero_reduction` of the same snapshots
(`Inf` for an exactly singular block). That pairing is computed again outside `build`, so an
`ArgumentError` of `psd_nonzero_reduction` or of `latent_pairing` propagates, and only the
guards of [`dual_psd_reduction`](@ref) on the odd–even block give a singular row. Any other
exception propagates.
"""
function dual_rows(
        common, case::VlasovCase, S::AbstractArray{<:Any, 3}, f0::AbstractVector,
        n::Integer, mode; build = dual_psd_reduction)
    train = S[:, :, 1:(mode.m_train)]
    built = try
        @timed build(train, f0, n, case.k)
    catch e
        e isa ArgumentError || rethrow()
        nothing
    end
    if built === nothing
        h = n ÷ 2
        κ = cond(latent_pairing(psd_nonzero_reduction(train, f0, n), case.k)[1:h, (h + 1):n])
        return [merge!(nan_row(common, "dual_psd", "sympnet", seed, "singular"),
                    Dict("pairing_condition" => string(κ))) for seed in mode.seeds]
    end
    κ = string(pairing_condition(built.value))
    return [merge!(
                sympnet_row(common, "dual_psd", built.value, case, S, seed, mode;
                    seconds = built.time),
                Dict("pairing_condition" => κ)) for seed in mode.seeds]
end

"""
    weight_tied_rows(common, case, S, f0, n, mode)

The rows of the weight-tied network on the `nx × nv × m` snapshots `S` of `case`: for each seed
of `mode`, [`train_weight_tied`](@ref) on `S[:, :, 1:mode.m_train]` around `f0` with
`mode.autoencoder_steps` steps, and the [`sympnet_row`](@ref) and the [`galerkin_row`](@ref) of
its reduction, each with the seed. A training whose status is not `"ok"` gives both rows that
status and `NaN` in every error, drift, loss and time column.
"""
function weight_tied_rows(common, case::VlasovCase, S::AbstractArray{<:Any, 3},
        f0::AbstractVector, n::Integer, mode)
    rows = Dict{String, String}[]
    for seed in mode.seeds
        trained = @timed train_weight_tied(S[:, :, 1:(mode.m_train)], f0, n; seed,
            steps = mode.autoencoder_steps)
        status = trained.value.training.status
        if status != "ok"
            append!(rows,
                [nan_row(common, "weight_tied", d, seed, status) for d in LATENT_DYNAMICS])
            continue
        end
        r = trained.value.reduction
        push!(rows,
            sympnet_row(
                common, "weight_tied", r, case, S, seed, mode; seconds = trained.time),
            galerkin_row(common, "weight_tied", r, case, S, seed, mode.m_train;
                seconds = trained.time))
    end
    return rows
end

"""
    latent_rows(case, S, n, mode)

The rows of the `nx × nv × m` snapshots `S` of the long run `case` at `n`. Every reduction is
built from the snapshots `1:mode.m_train` around the [`background`](@ref) `f0` of the first
snapshot: [`psd_reduction`](@ref) and [`psd_nonzero_reduction`](@ref), each with a
[`sympnet_row`](@ref) per seed and one [`galerkin_row`](@ref) with no seed, dual PSD by
[`dual_rows`](@ref), and the weight-tied network by [`weight_tied_rows`](@ref). Every row holds
the [`k0_fraction`](@ref) of the training snapshots.
"""
function latent_rows(case::VlasovCase, S::AbstractArray{<:Any, 3}, n::Integer, mode)
    train = S[:, :, 1:(mode.m_train)]
    f0 = background(S[:, :, 1])
    common = Dict("dataset" => case.name, "n" => string(n),
        "k0_fraction" => string(k0_fraction(train, f0)))
    rows = Dict{String, String}[]
    linear = (("psd", psd_reduction), ("psd_nonzero", psd_nonzero_reduction))
    for (reduction, build) in linear
        built = @timed build(train, f0, n)
        for seed in mode.seeds
            push!(rows,
                sympnet_row(common, reduction, built.value, case, S, seed, mode;
                    seconds = built.time))
        end
        push!(rows,
            galerkin_row(common, reduction, built.value, case, S, nothing, mode.m_train;
                seconds = built.time))
    end
    append!(rows, dual_rows(common, case, S, f0, n, mode),
        weight_tied_rows(common, case, S, f0, n, mode))
    return rows
end

"""
    latent_comparison_rows(runs_dir; mode)

The rows of `latent_dynamics.csv` under the [`latent_comparison_mode`](@ref) `mode`: the
[`latent_rows`](@ref) of each of the three [`long_run_cases`](@ref) to `mode.long_tend` on the
grid of `mode`, at each `n` of `mode.ns`. A long run is read from `runs_dir` where its file holds
a run of the same grid, amplitude and end time, and otherwise run and written there, as
[`channel_datasets`](@ref) does. It prints the statuses of each run and `n`.
"""
function latent_comparison_rows(runs_dir::AbstractString; mode)
    runs = channel_datasets(
        (grid = mode.grid, long_tend = mode.long_tend, advection_fractions = (;)), runs_dir)
    rows = Dict{String, String}[]
    # channel_datasets returns the long runs in the order of long_run_cases
    for (case, (_, S, _)) in zip(long_run_cases(mode.long_tend), runs), n in mode.ns

        these = latent_rows(case, S, n, mode)
        append!(rows, these)
        @printf("  %s, n = %d: %s\n", case.name, n,
            join(unique(r["status"] for r in these), ", "))
        flush(stdout)
    end
    return rows
end

"""
    latent_cell(rows)

The cell of the page for the `rows` of one run, reduction, dynamics and `n`: the medians over the
rows of status `"ok"` of the extrapolation error, the in-sample error, the largest energy drift
and the largest mass drift, each with the four significant digits of [`format_value`](@ref) and
separated by ` / `, then the number of rows whose status is neither `"ok"` nor `"singular"`
(`diverged`) and of rows of status `"singular"`. Where no row is `"ok"`, each median is
`singular` when every row is singular, and `—` otherwise.
"""
function latent_cell(rows)
    ok = filter(r -> r["status"] == "ok", rows)
    singular = count(r -> r["status"] == "singular", rows)
    diverged = length(rows) - length(ok) - singular
    medians = map(("error_extrapolation", "error_in_sample", "max_energy_drift",
        "max_mass_drift")) do c
        isempty(ok) && return singular == length(rows) ? "singular" : "—"
        format_value(median_of([parse(Float64, r[c]) for r in ok]))
    end
    return "$(join(medians, " / ")); $diverged diverged, $singular singular"
end

"""
    write_latent_page(results_dir)

Write `<results_dir>/latent_dynamics.md` from `latent_dynamics.csv` in `results_dir`, and return
its path. Per run it gives one table with a row per reduction and dynamics and a column per `n`,
whose cells are those of [`latent_cell`](@ref), then the run's `k0_fraction` and the
`pairing_condition` of every dual PSD row. A paragraph says what the comparison holds.
"""
function write_latent_page(results_dir::AbstractString)
    rows = read_csv_rows(joinpath(results_dir, "latent_dynamics.csv"))
    path = joinpath(results_dir, "latent_dynamics.md")
    open(path, "w") do io
        println(io, "# Latent dynamics of the reductions\n")
        println(io,
            "Each reduction is built on the snapshots with t ≤ T/2 of each long run, and its ",
            "latent trajectory is advanced by a SympNet trained on the same snapshots or by ",
            "the Galerkin projection of the full-order field, over the whole run. Each cell ",
            "gives the median over the seeds of the extrapolation error / the in-sample error ",
            "/ the largest energy drift / the largest mass drift, each relative and the errors ",
            "relative to the fluctuation, then the number of diverged rows (a rollout that ",
            "diverged, or a Galerkin run or a training that failed) and of singular rows (a ",
            "dual PSD whose pairing block is singular or round-off). A median is over the rows ",
            "of status `ok`; it is `singular` where every row is singular.\n")
        println(io,
            "Dual PSD is canonical only for the Vlasov bracket linearised at the initial ",
            "background, the spatial average of the first snapshot. POD is not a row of the ",
            "comparison. Dual PSD has the reconstruction error of PSD without the x-average: ",
            "it maps the latent coordinates of that reduction by an invertible matrix, so both ",
            "span the same subspace.\n")
        for dataset in unique(r["dataset"] for r in rows)
            these = filter(r -> r["dataset"] == dataset, rows)
            ns = sort(unique(parse(Int, r["n"]) for r in these))
            println(io, "## $dataset\n")
            println(io, "| reduction | dynamics | ", join(("n = $n" for n in ns), " | "), " |")
            println(io, "|", repeat(":--|", length(ns) + 2))
            for reduction in LATENT_REDUCTIONS, dynamics in LATENT_DYNAMICS

                cells = map(ns) do n
                    cell = filter(
                        r -> r["reduction"] == reduction && r["dynamics"] == dynamics &&
                             r["n"] == string(n), these)
                    isempty(cell) ? "—" : latent_cell(cell)
                end
                all(==("—"), cells) && continue
                println(io, "| $reduction | $dynamics | ", join(cells, " | "), " |")
            end
            k0 = unique(r["k0_fraction"] for r in these)
            println(io, "\n`k0_fraction`: ",
                join((format_value(parse(Float64, k)) for k in k0), ", "), "\n")
            println(io, "`pairing_condition` of the dual PSD rows:\n")
            for r in filter(r -> r["reduction"] == "dual_psd", these)
                println(io, "- n = $(r["n"]), seed $(r["seed"]): ",
                    "$(format_value(parse(Float64, r["pairing_condition"]))) ($(r["status"]))")
            end
            println(io)
        end
    end
    return path
end
