"The columns of `channel_splits.csv`, in order."
const CHANNEL_SPLIT_COLUMNS = ("dataset", "n", "mx", "mv", "seed", "parity_modes", "best",
    "status", "frontend_error", "frontend_max_snapshot_error", "stage1_error",
    "stage1_max_snapshot_error", "trained_error", "trained_max_snapshot_error",
    "offline_seconds", "loss_first", "loss_last")

"The columns of `channel_dlra.csv`, in order."
const CHANNEL_DLRA_COLUMNS = (
    "dataset", "n", "mx", "mv", "r_in", "r_eff", "seed", "status",
    "error", "max_snapshot_error", "input_error", "input_max_snapshot_error")

"The columns of `channel_joint.csv`, in order."
const CHANNEL_JOINT_COLUMNS = (
    "dataset", "n", "mx", "mv", "r_in", "r_eff", "seed", "status",
    "grid_error", "grid_max_snapshot_error", "dlra_error", "dlra_max_snapshot_error",
    "grid_to_dlra_error", "grid_to_dlra_max_snapshot_error", "dlra_to_grid_error",
    "dlra_to_grid_max_snapshot_error", "offline_seconds", "loss_first", "loss_last")

"""
    channels_mode(smoke)

The settings of the full mode of the input channels, or of the smoke mode with `smoke = true`:
the `grid` `(nx, nv)`, the latent dimensions `ns`, the training `steps`, the `seed` of every
network, the input ranks `dlra_ranks` of the DLRA channel, the input rank `joint_rank` of the
DLRA channel of the two-channel network, and the end `long_tend` of the long runs and the
`advection_fractions` of the datasets of [`reconstruction_mode`](@ref).
"""
function channels_mode(smoke::Bool)
    datasets = reconstruction_mode(smoke)
    return (grid = datasets.grid, ns = datasets.ns, steps = datasets.steps, seed = 42,
        dlra_ranks = smoke ? (4,) : (4, 8, 16), joint_rank = smoke ? 4 : 8,
        long_tend = datasets.long_tend, advection_fractions = datasets.advection_fractions)
end

"""
    grid_splits(n, nx, nv)

Every split `(m_x, m_v)` of the latent dimension `n` into the modes of a
`BilinearStiefelLayer(nx, nv, m_x, m_v)`: ``m_x m_v = n``, ``m_x ≤ n_x`` and ``m_v ≤ n_v/2``,
in the order of rising ``m_x``.
"""
function grid_splits(n::Integer, nx::Integer, nv::Integer)
    return [(mx, n ÷ mx) for mx in 1:n if n % mx == 0 && mx ≤ nx && n ÷ mx ≤ nv ÷ 2]
end

"""
    copy_parameters!(dst, src)

Copy every array of the parameters `src` into the array at the same place of `dst`, a
`NamedTuple` or a `Tuple` of the same structure, through `parent` so that a Stiefel manifold
receives its matrix.
"""
copy_parameters!(dst::AbstractArray, src::AbstractArray) = copyto!(parent(dst), parent(src))
function copy_parameters!(dst::Union{NamedTuple, Tuple}, src::Union{NamedTuple, Tuple})
    foreach(copy_parameters!, values(dst), values(src))
    return dst
end

"""
    channel_network(G, mx, mv; seed, dlra = nothing)

The network of [`weight_tied_network`](@ref) on the `nx × nv × m` fluctuation snapshots `G`
with the modes `(mx, mv)` in place of [`PARITY_MODES`](@ref): the `:grid` frontend
`BilinearStiefelLayer(nx, nv, mx, mv)` set by its `solve!` on `G`, and the core
`WeightTiedSymplecticAutoencoder(2n, n; n_ambient_layers = 0)`, ``n = m_x m_v``, whose PSD lift
is set by `PSDArch(2n, n)` `solve!` on the frontend's coordinates of `G`, built with
`Xoshiro(seed)`.

With the `nx × nv × m` dense states `dlra`, it holds a second frontend, `:dlra`, a
`LowRankStiefelLayer(nx, nv, mx, mv)`, whose bases are those of a `BilinearStiefelLayer` set by
its `solve!` on `dlra`: GeometricMachineLearning has no `solve!` for `LowRankStiefelLayer`.
"""
function channel_network(G::AbstractArray{<:Any, 3}, mx::Integer, mv::Integer;
        seed::Integer, dlra = nothing)
    nx, nv, _ = size(G)
    n = mx * mv
    grid_frontend = BilinearStiefelLayer(nx, nv, mx, mv)
    frontends = dlra === nothing ? (grid = grid_frontend,) :
                (grid = grid_frontend, dlra = LowRankStiefelLayer(nx, nv, mx, mv))
    core = WeightTiedSymplecticAutoencoder(2n, n; n_ambient_layers = 0)
    nn = NeuralNetwork(MultiModalSymplecticAutoencoder(frontends, core), CPU(), Float64;
        rng = Xoshiro(seed))
    ps = nn.params
    solve!(grid_frontend, ps.frontends.grid, G)
    if dlra !== nothing
        solve!(BilinearStiefelLayer(nx, nv, mx, mv), ps.frontends.dlra, dlra)
    end
    coordinates = grid_frontend(G, ps.frontends.grid)
    psd = NeuralNetwork(PSDArch(2n, n), CPU(), Float64; rng = Xoshiro(seed))
    solve!(psd, vcat(coordinates.q, coordinates.p))
    copyto!(parent(ps.core[end].weight), parent(psd.params[1].weight))

    return nn
end

"""
    dlra_network(nn)

The network of the grid network `nn` of [`channel_network`](@ref) with a second frontend,
`:dlra`, a `LowRankStiefelLayer` of the same modes: the bases of both frontends and the core
are copies of those of `nn`, so that the DLRA channel encodes into the same core.
"""
function dlra_network(nn)
    grid = nn.architecture.frontends.grid
    nx, nv = size(nn.params.frontends.grid.Φx, 1), 2 * size(nn.params.frontends.grid.Φv, 1)
    mx, mv = size(nn.params.frontends.grid.Φx, 2), size(nn.params.frontends.grid.Φv, 2)
    frontends = (grid = grid, dlra = LowRankStiefelLayer(nx, nv, mx, mv))
    two = NeuralNetwork(MultiModalSymplecticAutoencoder(frontends, nn.architecture.core),
        CPU(), Float64; rng = Xoshiro(0))
    copy_parameters!(two.params.frontends.grid, nn.params.frontends.grid)
    copy_parameters!(two.params.frontends.dlra, nn.params.frontends.grid)
    copy_parameters!(two.params.core, nn.params.core)

    return two
end

"""
    truncated_factors(G, r)

The factors `(X = …, S = …, V = …)`, `nx × r × m`, `r × r × m` and `nv × r × m`, of the truncated
SVD of rank `r` of each `nx × nv` slice of the `nx × nv × m` array `G`. Where `r` exceeds the
rank of a slice, the trailing factors of its SVD are kept.
"""
function truncated_factors(G::AbstractArray{<:Any, 3}, r::Integer)
    nx, nv, m = size(G)
    X, S, V = zeros(nx, r, m), zeros(r, r, m), zeros(nv, r, m)
    for s in 1:m
        F = svd(G[:, :, s])
        X[:, :, s] .= F.U[:, 1:r]
        S[:, :, s] .= Diagonal(F.S[1:r])
        V[:, :, s] .= F.V[:, 1:r]
    end
    return (X = X, S = S, V = V)
end

"The `nx × nv × m` dense states ``X S V^{\\mathsf T}`` of the batched factors `x`."
function dense_states(x::NamedTuple{(:X, :S, :V)})
    return stack(x.X[:, :, s] * x.S[:, :, s] * x.V[:, :, s]' for s in axes(x.X, 3))
end

"""
    relative_errors(G, R)

The aggregate relative error `error` ``\\|G - R\\|_F / \\|G\\|_F`` of the reconstruction `R` of
the `nx × nv × m` snapshots `G`, and the maximum `max_snapshot_error` of the per-snapshot errors
``\\|g_s - r_s\\| / \\|g_s\\|``.
"""
function relative_errors(G::AbstractArray{<:Any, 3}, R::AbstractArray{<:Any, 3})
    snapshot = [norm(view(R, :, :, s) - view(G, :, :, s)) / norm(view(G, :, :, s))
                for s in axes(G, 3)]
    return (error = norm(R - G) / norm(G), max_snapshot_error = maximum(snapshot))
end

"""
    channel_fluctuations(dataset, S)

The fluctuations of the `nx × nv × m` snapshots `S` of `dataset` around the
[`background`](@ref) of the first. It throws the error of [`fluctuation_norms`](@ref) when a
snapshot has no fluctuation, which would give [`relative_errors`](@ref) a `NaN` or an `Inf`.
"""
function channel_fluctuations(dataset::AbstractString, S::AbstractArray{<:Any, 3})
    f0 = background(S[:, :, 1])
    fluctuation_norms(dataset, S, f0)
    return fluctuation(S, f0)
end

"The reconstruction of `G` by the `:grid` frontend of `nn` and its inverse, without the core."
function frontend_reconstruction(nn, G::AbstractArray{<:Any, 3})
    ps = nn.params.frontends.grid
    return nn.model.inverse_frontends.grid(nn.model.frontends.grid(G, ps), ps)
end

"The reconstruction of the input `x` of `modality` by `nn`, through its core."
function channel_reconstruction(nn, modality::Symbol, x)
    return GeometricMachineLearning.decode(nn, modality,
        GeometricMachineLearning.encode(nn, modality, x))
end

"""
    joint_loss(model, ps, G, x)

The sum of GeometricMachineLearning's `CharbonnierRelativeLoss` of the `:grid` modality on the
fluctuation snapshots `G` and of the `:dlra` modality on their factors `x`.
"""
function joint_loss(model, ps, G, x)
    loss = CharbonnierRelativeLoss()
    return loss(model, ps, :grid, G) + loss(model, ps, :dlra, x)
end

"""
    error_fields(prefix, errors)

The fields `"<prefix>error"` and `"<prefix>max_snapshot_error"` of a row, from
[`relative_errors`](@ref).
"""
function error_fields(prefix::AbstractString, errors)
    return Dict(prefix * "error" => string(errors.error),
        prefix * "max_snapshot_error" => string(errors.max_snapshot_error))
end

"""
    training_fields(timed)

The fields `status`, `offline_seconds`, `loss_first` and `loss_last` of a row, from the `@timed`
result `timed` of [`train_network!`](@ref).
"""
function training_fields(timed)
    return Dict("status" => timed.value.status, "offline_seconds" => string(timed.time),
        "loss_first" => string(timed.value.first), "loss_last" => string(timed.value.last))
end

"An empty row of the `columns`, with the fields of `fields` set."
empty_row(columns, fields) = merge!(Dict{String, String}(c => "" for c in columns), fields)

"""
    split_rows(dataset, G, n; seed, steps)

The rows of `channel_splits.csv` of `dataset` at `n`, one per split of [`grid_splits`](@ref),
and the trained network at the split of [`PARITY_MODES`](@ref). Each row holds the errors of
[`relative_errors`](@ref) of the fluctuation snapshots `G`: the frontend error of
[`frontend_reconstruction`](@ref), and the Stage I error of the untrained
[`channel_network`](@ref). The split of [`PARITY_MODES`](@ref) (`parity_modes = yes`) and the
split of the smallest Stage I error (`best = yes`, the first of equal ones) are trained with
[`train_network!`](@ref) for `steps` steps on `CharbonnierRelativeLoss` of the `:grid` modality,
and their rows hold the trained error. A training whose status is not `"ok"` gives its row that
status and an empty trained error. The trained network is `nothing` where that training fails.
"""
function split_rows(dataset::AbstractString, G::AbstractArray{<:Any, 3}, n::Integer;
        seed::Integer, steps::Integer)
    nx, nv, _ = size(G)
    rows = map(grid_splits(n, nx, nv)) do (mx, mv)
        nn = channel_network(G, mx, mv; seed)
        fields = Dict("dataset" => dataset, "n" => string(n), "mx" => string(mx),
            "mv" => string(mv), "seed" => string(seed), "status" => "ok",
            "parity_modes" => (mx, mv) == PARITY_MODES[n] ? "yes" : "no")
        merge!(fields,
            error_fields("frontend_", relative_errors(G,
                frontend_reconstruction(nn, G))),
            error_fields("stage1_", relative_errors(G, channel_reconstruction(nn, :grid, G))))
        empty_row(CHANNEL_SPLIT_COLUMNS, fields)
    end
    best = argmin([parse(Float64, r["stage1_error"]) for r in rows])
    foreach(((i, r),) -> r["best"] = i == best ? "yes" : "no", enumerate(rows))

    trained = nothing
    loss = CharbonnierRelativeLoss()
    for r in rows
        r["parity_modes"] == "yes" || r["best"] == "yes" || continue
        nn = channel_network(G, parse(Int, r["mx"]), parse(Int, r["mv"]); seed)
        timed = @timed train_network!(ps -> loss(nn.model, ps, :grid, G), nn.params, steps)
        merge!(r, training_fields(timed))
        timed.value.status == "ok" || continue
        merge!(r, error_fields("trained_", relative_errors(G,
            channel_reconstruction(nn, :grid, G))))
        r["parity_modes"] == "yes" && (trained = nn)
    end

    return (rows = rows, network = trained)
end

"""
    dlra_row(dataset, G, n, r_in, nn, status; seed)

The row of `channel_dlra.csv` of `dataset` at `n` and the input rank `r_in`: the
[`truncated_factors`](@ref) of rank `r_in` of each fluctuation snapshot of `G`, encoded by the
`:dlra` channel of [`dlra_network`](@ref) of the trained grid network `nn` and decoded by it to
factors of rank `r_eff`. It holds the [`relative_errors`](@ref) of the dense decoded states
against `G` (`error`) and against the dense rank-`r_in` input (`input_error`). With a `status`
of the training of `nn` that is not `"ok"`, the row has that status and empty errors.
"""
function dlra_row(dataset::AbstractString, G::AbstractArray{<:Any, 3}, n::Integer,
        r_in::Integer, nn, status::AbstractString; seed::Integer)
    mx, mv = PARITY_MODES[n]
    row = empty_row(CHANNEL_DLRA_COLUMNS,
        Dict(
            "dataset" => dataset, "n" => string(n), "mx" => string(mx), "mv" => string(mv),
            "r_in" => string(r_in), "seed" => string(seed), "status" => status))
    status == "ok" || return row
    x = truncated_factors(G, r_in)
    decoded = channel_reconstruction(dlra_network(nn), :dlra, x)
    R = dense_states(decoded)
    row["r_eff"] = string(size(decoded.X, 2))
    merge!(row, error_fields("", relative_errors(G, R)),
        error_fields("input_", relative_errors(dense_states(x), R)))

    return row
end

"""
    joint_row(dataset, G, n, r_in; seed, steps)

The row of `channel_joint.csv` of `dataset` at `n`: the [`channel_network`](@ref) at the split
of [`PARITY_MODES`](@ref) with the `:dlra` frontend set from the dense states of the
[`truncated_factors`](@ref) `x` of rank `r_in` of the fluctuation snapshots `G`, trained with
[`train_network!`](@ref) for `steps` steps on the [`joint_loss`](@ref) of `G` and `x`. It holds
the [`relative_errors`](@ref) against `G` of each channel's reconstruction and of each
translation, `translate(nn, :grid, :dlra, G)` and `translate(nn, :dlra, :grid, x)`, with the
decoded factors as dense states. A training whose status is not `"ok"` gives a row with that
status and empty errors.
"""
function joint_row(dataset::AbstractString, G::AbstractArray{<:Any, 3}, n::Integer,
        r_in::Integer; seed::Integer, steps::Integer)
    mx, mv = PARITY_MODES[n]
    x = truncated_factors(G, r_in)
    nn = channel_network(G, mx, mv; seed, dlra = dense_states(x))
    timed = @timed train_network!(ps -> joint_loss(nn.model, ps, G, x), nn.params, steps)
    row = empty_row(CHANNEL_JOINT_COLUMNS,
        merge(
            Dict("dataset" => dataset, "n" => string(n), "mx" => string(mx),
                "mv" => string(mv), "r_in" => string(r_in), "seed" => string(seed)),
            training_fields(timed)))
    timed.value.status == "ok" || return row
    decoded = channel_reconstruction(nn, :dlra, x)
    to_dlra = GeometricMachineLearning.translate(nn, :grid, :dlra, G)
    row["r_eff"] = string(size(decoded.X, 2))
    merge!(row,
        error_fields("grid_", relative_errors(G, channel_reconstruction(nn, :grid, G))),
        error_fields("dlra_", relative_errors(G, dense_states(decoded))),
        error_fields("grid_to_dlra_", relative_errors(G, dense_states(to_dlra))),
        error_fields("dlra_to_grid_",
            relative_errors(G,
                GeometricMachineLearning.translate(nn, :dlra, :grid, x))))

    return row
end

"""
    channel_rows(dataset, G, n, mode)

The rows of `dataset` at `n` on its fluctuation snapshots `G`, with the `seed`, `steps`,
`dlra_ranks` and `joint_rank` of the [`channels_mode`](@ref) `mode`: the `splits` of
[`split_rows`](@ref), a [`dlra_row`](@ref) per input rank from the trained network at the split
of [`PARITY_MODES`](@ref), with the status of that training, and the `joint` row of
[`joint_row`](@ref).
"""
function channel_rows(dataset::AbstractString, G::AbstractArray{<:Any, 3}, n::Integer, mode)
    result = split_rows(dataset, G, n; mode.seed, mode.steps)
    status = only(r for r in result.rows if r["parity_modes"] == "yes")["status"]
    dlra = [dlra_row(dataset, G, n, r_in, result.network, status; mode.seed)
            for r_in in mode.dlra_ranks]
    return (splits = result.rows, dlra = dlra,
        joint = joint_row(dataset, G, n, mode.joint_rank; mode.seed, mode.steps))
end

"""
    channel_datasets(mode, runs_dir)

The five datasets of [`reconstruction_datasets`](@ref) on the grid of `mode`, each as
`(name, snapshots, times)`. A long run is read from its [`run_file`](@ref) in `runs_dir` where
that file holds a run of the same grid, amplitude and end time; otherwise [`run_case`](@ref)
runs it with its checks and writes it there.
"""
function channel_datasets(mode, runs_dir::AbstractString)
    nx, nv = mode.grid
    k = linear_landau().k
    datasets = Tuple{String, Array{Float64, 3}, Vector{Float64}}[]
    for (name, fraction) in pairs(mode.advection_fractions)
        times = advection_times(nv, k; fraction)
        push!(datasets, (string(name), advection_snapshots(nx, nv, times), times))
    end
    for case in long_run_cases(mode.long_tend)
        path = joinpath(runs_dir, run_file(case))
        stored = isfile(path) && h5open(path) do file
            a = attrs(file)
            (a["nx"], a["nv"], a["alpha"], a["tend"]) == (nx, nv, case.α, case.tend)
        end
        if stored
            S, times = h5open(f -> (read(f, "snapshots"), read(f, "snapshot_times")), path)
            @printf("  %s: read %s\n", case.name, path)
        else
            run = run_case(case, nx, nv; runs_dir)
            S, times = run.snapshots, run.snapshot_times
            @printf("  %s: ran %d steps, wrote %s\n", case.name, length(run.times) - 1,
                path)
        end
        flush(stdout)
        push!(datasets, (case.name, S, times))
    end
    return datasets
end

"""
    write_channels_page(results_dir)

Write `<results_dir>/channels.md` from `channel_splits.csv`, `channel_dlra.csv` and
`channel_joint.csv` in `results_dir`, and return its path. Per dataset it gives three tables:
the frontend, Stage I and trained errors per `n` and split, with the split of
[`PARITY_MODES`](@ref) and the best split marked; the DLRA channel's errors per `n` and input
rank; and the two-channel network's errors per `n`. A row whose status is not `"ok"` shows `—`
for its missing errors, and the page lists it with its status. Every value comes from the three
CSV files: each error with the four significant digits of [`format_value`](@ref), each count as
written.
"""
function write_channels_page(results_dir::AbstractString)
    splits = read_csv_rows(joinpath(results_dir, "channel_splits.csv"))
    dlra = read_csv_rows(joinpath(results_dir, "channel_dlra.csv"))
    joint = read_csv_rows(joinpath(results_dir, "channel_joint.csv"))
    cell(r, c) = isempty(r[c]) ? "—" : format_value(parse(Float64, r[c]))
    path = joinpath(results_dir, "channels.md")
    open(path, "w") do io
        println(io, "# The input channels of the multi-modal network\n")
        println(
            io, "The relative error of the reconstruction of the fluctuation snapshots, ",
            "in sample. The frontend error is that of the grid frontend and its inverse alone, ",
            "the Stage I error that of the untrained network, and the trained error that of ",
            "the trained network, at the split of `PARITY_MODES` and at the best split, the ",
            "one of the smallest Stage I error. The DLRA channel encodes the truncated SVD of ",
            "each snapshot into the trained core of the grid network at the split of ",
            "`PARITY_MODES`; its errors are against the full fluctuation and against its ",
            "input. The two-channel network trains on both channels at once.\n")
        for dataset in unique(r["dataset"] for r in splits)
            println(io, "## $dataset\n")
            println(io, "### The splits of the grid channel\n")
            println(io, "| n | (m_x, m_v) | frontend | Stage I | trained | split |")
            println(io, "|:--|:--|:--|:--|:--|:--|")
            for r in filter(r -> r["dataset"] == dataset, splits)
                marks = filter(!isempty,
                    [r["parity_modes"] == "yes" ? "`PARITY_MODES`" : "",
                        r["best"] == "yes" ? "best" : ""])
                println(io, "| $(r["n"]) | ($(r["mx"]), $(r["mv"])) | ",
                    "$(cell(r, "frontend_error")) | $(cell(r, "stage1_error")) | ",
                    "$(cell(r, "trained_error")) | $(join(marks, ", ")) |")
            end
            println(io, "\n### The DLRA channel\n")
            println(io, "| n | (m_x, m_v) | r_in | r_eff | against the fluctuation | ",
                "against the input |")
            println(io, "|:--|:--|:--|:--|:--|:--|")
            for r in filter(r -> r["dataset"] == dataset, dlra)
                println(io, "| $(r["n"]) | ($(r["mx"]), $(r["mv"])) | $(r["r_in"]) | ",
                    "$(isempty(r["r_eff"]) ? "—" : r["r_eff"]) | $(cell(r, "error")) | ",
                    "$(cell(r, "input_error")) |")
            end
            println(io, "\n### The two-channel network\n")
            println(io, "| n | (m_x, m_v) | r_in | grid | DLRA | grid → DLRA | DLRA → grid |")
            println(io, "|:--|:--|:--|:--|:--|:--|:--|")
            for r in filter(r -> r["dataset"] == dataset, joint)
                println(io, "| $(r["n"]) | ($(r["mx"]), $(r["mv"])) | $(r["r_in"]) | ",
                    "$(cell(r, "grid_error")) | $(cell(r, "dlra_error")) | ",
                    "$(cell(r, "grid_to_dlra_error")) | $(cell(r, "dlra_to_grid_error")) |")
            end
            println(io)
        end
        println(io, "## Failed rows\n")
        failed = [(file, r)
                  for (file, rs) in (("channel_splits.csv", splits),
            ("channel_dlra.csv", dlra), ("channel_joint.csv", joint))
                  for r in rs if r["status"] != "ok"]
        isempty(failed) && println(io, "No row failed.")
        for (file, r) in failed
            println(io, "- `$file`: $(r["dataset"]), n = $(r["n"]), ",
                "(m_x, m_v) = ($(r["mx"]), $(r["mv"])): $(r["status"])")
        end
    end
    return path
end

"""
    run_channels(; runs_dir, results_dir, smoke = false)

The input channels of the multi-modal network with the settings of [`channels_mode`](@ref), on
the [`channel_datasets`](@ref), whose long runs are read from or written to `runs_dir`: for
each dataset and `n`, the [`channel_rows`](@ref) of its [`channel_fluctuations`](@ref),
written to `channel_splits.csv`, `channel_dlra.csv` and `channel_joint.csv` in `results_dir`,
with the page of [`write_channels_page`](@ref). A failed training is a row with its status, and
the script goes on. A snapshot with no fluctuation stops it with the error of
[`channel_fluctuations`](@ref). It prints the number of rows and of failed rows of each file,
and returns the rows.
"""
function run_channels(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false)
    mode = channels_mode(smoke)
    @printf("input channels on %d × %d%s, n ∈ %s, %d steps\n", mode.grid...,
        smoke ? " (smoke)" : "", mode.ns, mode.steps)
    flush(stdout)
    mkpath(results_dir)
    rows = (splits = Dict{String, String}[], dlra = Dict{String, String}[],
        joint = Dict{String, String}[])
    for (name, S, _) in channel_datasets(mode, runs_dir), n in mode.ns

        these = channel_rows(name, channel_fluctuations(name, S), n, mode)
        append!(rows.splits, these.splits)
        append!(rows.dlra, these.dlra)
        push!(rows.joint, these.joint)
        @printf("  %s, n = %d: %s\n", name, n,
            join(
                unique(r["status"] for r in (these.splits..., these.dlra..., these.joint)),
                ", "))
        flush(stdout)
    end
    for (file, columns, rs) in (("channel_splits.csv", CHANNEL_SPLIT_COLUMNS, rows.splits),
        ("channel_dlra.csv", CHANNEL_DLRA_COLUMNS, rows.dlra),
        ("channel_joint.csv", CHANNEL_JOINT_COLUMNS, rows.joint))
        write_csv(joinpath(results_dir, file), columns, rs)
        @printf("%s: %d rows, %d failed\n", file, length(rs),
            count(r -> r["status"] != "ok", rs))
    end
    write_channels_page(results_dir)
    @printf("input channels written to %s\n", results_dir)
    flush(stdout)

    return rows
end
