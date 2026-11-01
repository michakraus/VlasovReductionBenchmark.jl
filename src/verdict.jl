"The reduced models of the comparison: the three baselines first, then the two autoencoders."
const VERDICT_BASELINES = ("POD", "PSD", "DLRA")

"The autoencoders of `autoencoders.csv`, by the name of their `method` column."
const VERDICT_AUTOENCODERS = ("untied", "weight-tied")

"The latent dimensions ``n`` (the DLRA ranks ``r``) of the comparison."
const VERDICT_DIMS = (4, 8, 16, 32)

"""
The number of the [`VERDICT_DIMS`](@ref) at which a method must be better than the best baseline
to beat it on a criterion.
"""
const BEAT_COUNT = 3

"The data cases of the comparison, then the generalisation sweep."
const VERDICT_CASES = ("linear_landau", "nonlinear_landau", "two_stream", "sweep")

"""
The tolerance of a fitted rate relative to the rate of the full-order run, by case: I1's 2 % for
Landau damping (the sweep is linear Landau damping) and 5 % for the growth of the two-stream
instability.
"""
const RATE_TOLERANCE = Dict("linear_landau" => 0.02, "two_stream" => 0.05, "sweep" => 0.02)

"The largest drift of every invariant of a data row that the page lists as stationary."
const STATIONARY_DRIFT = 1e-12

"""
The criteria of a data case, as `(key, title)`. The rate applies to the data cases of
[`RATE_TOLERANCE`](@ref) only; the sweep has the out-of-sample error alone.
"""
const DATA_CRITERIA = (("error", "maximal error relative to the fluctuation"),
    ("rate", "rate, relative deviation from the full-order rate"),
    ("energy_drift", "energy drift"), ("l1_drift", "L¹ drift"), ("l2_drift", "L² drift"),
    ("entropy_drift", "entropy drift"),
    ("cost", "online cost per unit of simulated time, relative to the full order"))

"The criterion of the sweep: the out-of-sample error."
const SWEEP_CRITERION = ("error", "out-of-sample error relative to the fluctuation")

"""
The input files of the page, each with the file whose results directory must also hold it.
Every results directory of a CSV file of a Vlasov method holds a `full_order_cost.toml` too.
"""
const VERDICT_FILES = (
    "linear_baselines.csv" => nothing, "linear_baselines_cost.toml" => "linear_baselines.csv",
    "autoencoders.csv" => nothing, "autoencoders_cost.toml" => "autoencoders.csv",
    "autoencoders_hamiltonian.toml" => "autoencoders.csv",
    "dlra.csv" => nothing, "dlra_cost.toml" => "dlra.csv",
    "particle_frontends.csv" => nothing, "particle_pod_modes.csv" => "particle_frontends.csv",
    "particle_cost.toml" => "particle_frontends.csv")

"The sentence on what the error criterion can show, which the page states verbatim."
const ERROR_CRITERION_NOTE = "POD is the L²-optimal linear reconstruction at each n, so PSD, a " *
                             "restricted linear family, cannot beat it on the error criterion. The SAEs are nonlinear, so " *
                             "no such bound orders them against POD, but the symplectic constraint restricts them as " *
                             "PSD's restricts POD; a non-symplectic autoencoder, which the benchmark does not include, " *
                             "would separate the gain of nonlinearity from the cost of structure. The energy and " *
                             "Casimir drifts are the criteria where structure is expected to show, and the time spans " *
                             "of these cases are short."

"""
    csv_fields(line)

The fields of one CSV `line`, as [`csv_field`](@ref) writes them: a field in double quotes may
hold a comma, and a doubled quote inside it stands for one quote.
"""
function csv_fields(line::AbstractString)
    fields, field, quoted = String[], IOBuffer(), false
    chars = collect(line)
    i = 1
    while i ≤ length(chars)
        c = chars[i]
        if quoted && c == '"' && get(chars, i + 1, ' ') == '"'
            write(field, '"')
            i += 1
        elseif c == '"'
            quoted = !quoted
        elseif c == ',' && !quoted
            push!(fields, String(take!(field)))
        else
            write(field, c)
        end
        i += 1
    end
    push!(fields, String(take!(field)))

    return fields
end

"""
    read_csv_rows(path)

The rows of the CSV file `path`, each a `Dict` from the names of the header to the fields. It
throws an error naming the file and the line when a row has another number of fields.
"""
function read_csv_rows(path::AbstractString)
    lines = readlines(path)
    header = csv_fields(lines[1])
    return map(enumerate(lines[2:end])) do (i, line)
        fields = csv_fields(line)
        length(fields) == length(header) || error(
            "$path, line $(i + 1): $(length(fields)) fields, not the $(length(header)) of " *
            "the header")
        Dict(zip(header, fields))
    end
end

"""
    directory_commit(dir)

The commit that the path of the results directory `dir` names: the hexadecimal suffix of 7 to 40
digits after the last `-` of the innermost component of the path that ends in one, as in
`VlasovReductionBenchmark-I2-760c7a5/results`. It throws an error when no component names one.
"""
function directory_commit(dir::AbstractString)
    for part in reverse(splitpath(normpath(abspath(dir))))
        m = match(r"-([0-9a-f]{7,40})$", part)
        m === nothing || return String(m.captures[1])
    end
    error("the path of the results directory $dir names no commit: no component of it ends " *
          "in -<commit>")
end

"""
    verdict_inputs(results_dirs)

The path of each file of [`VERDICT_FILES`](@ref) among the directories `results_dirs`, and the
`full_order_cost.toml` beside each CSV file of a Vlasov method, with the
[`directory_commit`](@ref) of the directory of each file of [`VERDICT_FILES`](@ref). A file of [`VERDICT_FILES`](@ref) without a
partner is looked for in every directory; one with a partner in the directory of that partner.
It throws an error naming the file when a file is missing, or when more than one directory holds
a file that is looked for in every directory.
"""
function verdict_inputs(results_dirs)
    paths = Dict{String, String}()
    for (file, partner) in VERDICT_FILES
        if partner === nothing
            found = filter(d -> isfile(joinpath(d, file)), collect(results_dirs))
            isempty(found) && error(
                "missing input file $file: none of the results directories " *
                "$(join(results_dirs, ", ")) holds it")
            length(found) == 1 || error(
                "$file is in more than one results directory: $(join(found, ", "))")
            paths[file] = joinpath(only(found), file)
        else
            path = joinpath(dirname(paths[partner]), file)
            isfile(path) || error("missing input file $path")
            paths[file] = path
        end
    end
    full_order = Dict{String, String}()
    for csv in ("linear_baselines.csv", "autoencoders.csv", "dlra.csv")
        path = joinpath(dirname(paths[csv]), "full_order_cost.toml")
        isfile(path) || error("missing input file $path")
        full_order[csv] = path
    end
    commits = Dict(p => directory_commit(dirname(p)) for p in values(paths))

    return (paths = paths, full_order = full_order, commits = commits)
end

"The median of the non-empty vector `x`."
function median_of(x::AbstractVector)
    s = sort(x)
    m = length(s) ÷ 2
    return isodd(length(s)) ? s[m + 1] : (s[m] + s[m + 1]) / 2
end

"""
    error_column(method)

The column of the error criterion of `method`: for DLRA its error against its own full-rank
solver, relative to the fluctuation of that solver's run; for every other method its error
against the full-order run, relative to the fluctuation.
"""
function error_column(method::AbstractString)
    method == "DLRA" ? "max_l2_error_fluct_full_rank" : "max_l2_error_fluct"
end

"""
    rate_deviation(row)

The deviation ``|γ - γ_\\mathrm{full}| / |γ_\\mathrm{full}|`` of the rate of `row` from the rate
`rate_full_order` of the full-order run, and `Inf` for a row whose `rate_status` is not `ok`: an
empty rate counts as not better.
"""
function rate_deviation(row)
    row["rate_status"] == "ok" || return Inf
    full = parse(Float64, row["rate_full_order"])
    return abs(parse(Float64, row["rate"]) - full) / abs(full)
end

"The seed of `row`, or `\"\"` for a method with no seed."
row_seed(row) = get(row, "seed", "")

"""
    load_verdict(inputs)

The rows and files that [`verdict_inputs`](@ref) names, read: `rows`, every row of the three CSV
files of the Vlasov methods; `costs`, the online cost of each completed data row per unit of
simulated time relative to the full order, by `(method, run file, n, seed)`; `full_order`, the
`full_order_cost.toml` of each CSV file; `cost_dt`, the `dt` of each cost file; `hamiltonian`;
and the particle rows `frontends` and `pod_modes`, with `particle_cost`.
"""
function load_verdict(inputs)
    p = inputs.paths
    dlra = read_csv_rows(p["dlra.csv"])
    isempty(dlra) || haskey(first(dlra), error_column("DLRA")) ||
        error(
            "$(p["dlra.csv"]) has no column $(error_column("DLRA")): it was written before " *
            "scripts/dlra.jl wrote that column, so run it again")
    rows = vcat(
        read_csv_rows(p["linear_baselines.csv"]), read_csv_rows(p["autoencoders.csv"]),
        dlra)
    full_order = Dict(csv => TOML.parsefile(path) for (csv, path) in inputs.full_order)
    costs, cost_dt = Dict{NTuple{4, String}, Float64}(), Dict{String, Float64}()
    for (csv, cost) in ("linear_baselines.csv" => "linear_baselines_cost.toml",
        "autoencoders.csv" => "autoencoders_cost.toml", "dlra.csv" => "dlra_cost.toml")
        c = TOML.parsefile(p[cost])
        full = full_order[csv]["seconds_per_step"] / full_order[csv]["dt"]
        cost_dt[cost] = c["dt"]
        for r in get(c, "rows", [])
            method = get(r, "method", get(r, "network", "DLRA"))
            key = (method, r["file"], string(r["n"]), string(get(r, "seed", "")))
            costs[key] = r["seconds_per_step"] / c["dt"] / full
        end
    end

    return (rows = rows, costs = costs, full_order = full_order, cost_dt = cost_dt,
        hamiltonian = TOML.parsefile(p["autoencoders_hamiltonian.toml"]),
        frontends = read_csv_rows(p["particle_frontends.csv"]),
        pod_modes = read_csv_rows(p["particle_pod_modes.csv"]),
        particle_cost = TOML.parsefile(p["particle_cost.toml"]))
end

"""
    row_value(data, row, criterion)

The value of the completed data `row` on `criterion`, a key of [`DATA_CRITERIA`](@ref). The cost
comes from the cost file of the row's method; it throws an error when that file has no row for
it.
"""
function row_value(data, row, criterion)
    criterion == "error" && return parse(Float64, row[error_column(row["method"])])
    criterion == "rate" && return rate_deviation(row)
    if criterion == "cost"
        file = "$(row["case"])_alpha$(row["alpha"]).h5"
        key = (row["method"], file, row["n"], row_seed(row))
        haskey(data.costs, key) || error("no cost row for $(join(key, ", "))")
        return data.costs[key]
    end
    return parse(Float64, row[criterion])
end

"""
    criterion_values(data, case, criterion)

The values of every method of [`VERDICT_BASELINES`](@ref) and [`VERDICT_AUTOENCODERS`](@ref) on
`criterion` in `case`, by method and ``n``, one per seed. Only rows whose status is `ok` enter.
For the sweep each seed's value is its largest error over the test amplitudes, and a seed with a
failed row at any amplitude has none.
"""
function criterion_values(data, case, criterion)
    values = Dict(m => Dict(n => Float64[] for n in VERDICT_DIMS)
    for m in (VERDICT_BASELINES..., VERDICT_AUTOENCODERS...))
    if case == "sweep"
        groups = Dict{NTuple{3, String}, Vector{Any}}()
        for r in filter(r -> r["kind"] == "sweep", data.rows)
            push!(get!(groups, (r["method"], r["n"], row_seed(r)), []), r)
        end
        for ((method, n, _), group) in groups
            all(r -> r["status"] == "ok", group) || continue
            push!(values[method][parse(Int, n)],
                maximum(r -> row_value(data, r, criterion), group))
        end
    else
        for r in data.rows
            (r["kind"], r["case"], r["status"]) == ("data", case, "ok") || continue
            push!(values[r["method"]][parse(Int, r["n"])], row_value(data, r, criterion))
        end
    end

    return values
end

"""
    comparison(values)

The comparison of the [`criterion_values`](@ref) `values`, one entry per ``n`` of
[`VERDICT_DIMS`](@ref): `stats`, the median, the smallest and the largest value of each method,
or `nothing` where it has none; `best`, the baseline of the smallest median and that median, or
`nothing`; and for each autoencoder whether it is `better`, its median strictly smaller than
the best baseline's, and the `factor` of the best baseline's median over its own.
"""
function comparison(values)
    map(VERDICT_DIMS) do n
        stats = Dict(m => (isempty(v[n]) ? nothing :
                      (median = median_of(v[n]), lo = minimum(v[n]), hi = maximum(v[n])))
        for (m, v) in values)
        present = [(m, stats[m].median)
                   for m in VERDICT_BASELINES
                   if stats[m] !== nothing && !isnan(stats[m].median)]
        best = isempty(present) ? nothing : present[argmin(last.(present))]
        better = Dict(a =>
                          best !== nothing && stats[a] !== nothing &&
                          stats[a].median < best[2] for a in VERDICT_AUTOENCODERS)
        factor = Dict(a => better[a] ? best[2] / stats[a].median : nothing
        for a in VERDICT_AUTOENCODERS)
        (n = n, stats = stats, best = best, better = better, factor = factor)
    end
end

"""
    beats(cmp, autoencoder)

Whether `autoencoder` beats the best baseline in the [`comparison`](@ref) `cmp`: better at
[`BEAT_COUNT`](@ref) or more of its ``n``. It returns that, the number of ``n`` at which it is
better, and the factor of the win, the smallest factor over those ``n`` (`nothing` where it is
better at none).
"""
function beats(cmp, autoencoder)
    wins = [c.factor[autoencoder] for c in cmp if c.better[autoencoder]]
    return (beats = length(wins) ≥ BEAT_COUNT, count = length(wins),
        factor = isempty(wins) ? nothing : minimum(wins))
end

"The criteria of `case`: see [`DATA_CRITERIA`](@ref) and [`SWEEP_CRITERION`](@ref)."
function case_criteria(case)
    case == "sweep" && return (SWEEP_CRITERION,)
    return filter(c -> c[1] != "rate" || haskey(RATE_TOLERANCE, case), DATA_CRITERIA)
end

"""
    bounds(data, case, autoencoder, error_cmp)

The bounds of a win of `autoencoder` on the error criterion in `case`, whose
[`comparison`](@ref) is `error_cmp`: `A`, whether at some ``n`` at which it is better its median
[`rate_deviation`](@ref) is at most the [`RATE_TOLERANCE`](@ref) of the case (never for a case
without a rate); `C`, whether at some ``n ≤ 16`` its median error is at or below POD's at
``n = 32``.
"""
function bounds(data, case, autoencoder, error_cmp)
    A = false
    if haskey(RATE_TOLERANCE, case)
        rates = comparison(criterion_values(data, case, "rate"))
        A = any(zip(error_cmp, rates)) do (e, r)
            e.better[autoencoder] && r.stats[autoencoder] !== nothing &&
                r.stats[autoencoder].median ≤ RATE_TOLERANCE[case]
        end
    end
    pod32 = error_cmp[end].stats["POD"]
    C = pod32 !== nothing && any(error_cmp) do e
        s = e.stats[autoencoder]
        e.n ≤ 16 && s !== nothing && s.median ≤ pod32.median
    end

    return (A = A, C = C)
end

"""
    verdict_answer(data)

The last line of the page: `Yes` when some win of an autoencoder on the error criterion, as
[`beats`](@ref) gives it, meets both [`bounds`](@ref), with the win of the largest factor. It
returns `answer`, the line itself, and `wins`, each win with its case, factor and bounds. Without
a win that meets both the line names the closest win, the one with the most bounds met and then
the largest factor, and the bound it misses; without any win, the closest margin: the
autoencoder and case that are better at the most ``n``, then at the smallest ratio of their
median error to the best baseline's; without a completed row of an autoencoder, it says so.
"""
function verdict_answer(data)
    wins, margins = [], []
    for case in VERDICT_CASES, a in VERDICT_AUTOENCODERS

        cmp = comparison(criterion_values(data, case, "error"))
        b = beats(cmp, a)
        ratios = [(c.stats[a].median / c.best[2], c.n)
                  for c in cmp
                  if c.best !== nothing && c.stats[a] !== nothing]
        isempty(ratios) || push!(margins, (case = case, autoencoder = a, count = b.count,
            ratio = minimum(ratios)))
        b.beats && push!(wins, (case = case, autoencoder = a, factor = b.factor,
            bounds(data, case, a, cmp)...))
    end

    met = filter(w -> w.A && w.C, wins)
    answer = if !isempty(met)
        w = met[argmax([w.factor for w in met])]
        @sprintf("**Does part G go on?** Yes: the %s autoencoder beats the best baseline on the error criterion in %s by a factor of %s, and meets bounds A and C.",
            w.autoencoder, w.case, format_value(w.factor))
    elseif !isempty(wins)
        w = wins[argmax([(w.A + w.C, w.factor) for w in wins])]
        missed = join([name for (name, ok) in (("A", w.A), ("C", w.C)) if !ok], " and ")
        @sprintf("**Does part G go on?** No: no win meets both bounds. The closest win, the %s autoencoder in %s by a factor of %s, misses bound %s.",
            w.autoencoder, w.case, format_value(w.factor), missed)
    elseif !isempty(margins)
        m = margins[argmax([(m.count, -m.ratio[1]) for m in margins])]
        @sprintf("**Does part G go on?** No: no autoencoder beats the best baseline on the error criterion in any grid case. The closest margin: the %s autoencoder in %s is better at %d of the %d n, and its error is %s times the best baseline's at n = %d.",
            m.autoencoder, m.case, m.count, length(VERDICT_DIMS), format_value(m.ratio[1]),
            m.ratio[2])
    else
        "**Does part G go on?** No: no autoencoder has a completed row on the error criterion."
    end

    return (answer = answer, wins = wins)
end

"A value of the page: four significant digits, `∞` for `Inf`, `—` for `nothing`."
format_value(::Nothing) = "—"
format_value(x::Real) = isinf(x) ? (x > 0 ? "∞" : "-∞") : @sprintf("%.4g", x)

"The statistics `s` of a method: the value of a baseline, the median and range of a network."
function format_stats(s, method)
    s === nothing && return "—"
    method in VERDICT_AUTOENCODERS || return format_value(s.median)
    return "$(format_value(s.median)) [$(format_value(s.lo)), $(format_value(s.hi))]"
end

"The table and the summary lines of the [`comparison`](@ref) `cmp`, written to `io`."
function write_comparison(io, cmp)
    methods = (VERDICT_BASELINES..., VERDICT_AUTOENCODERS...)
    println(io, "| n | ", join(methods, " | "), " | best baseline | ",
        join(("$a better" for a in VERDICT_AUTOENCODERS), " | "), " |")
    println(io, "|", repeat(":--|", length(methods) + 2 + length(VERDICT_AUTOENCODERS)))
    for c in cmp
        best = c.best === nothing ? "—" : "$(c.best[1]) $(format_value(c.best[2]))"
        flags = (c.better[a] ? "yes (×$(format_value(c.factor[a])))" : "no"
        for a in VERDICT_AUTOENCODERS)
        println(
            io, "| $(c.n) | ", join((format_stats(c.stats[m], m) for m in methods), " | "),
            " | $best | ", join(flags, " | "), " |")
    end
    println(io)
    for a in VERDICT_AUTOENCODERS
        b = beats(cmp, a)
        println(io, "- $a: better at $(b.count) of $(length(VERDICT_DIMS)) n; ",
            b.beats ? "beats the best baseline by a factor of $(format_value(b.factor))." :
            "does not beat the best baseline.")
    end
    println(io)
end

"""
    stationary_rows(data)

The completed data rows whose drifts of the energy, the ``L^1`` and ``L^2`` norms and the
entropy are all at most [`STATIONARY_DRIFT`](@ref): rows whose reduced state does not move.
"""
function stationary_rows(data)
    drifts = ("energy_drift", "l1_drift", "l2_drift", "entropy_drift")
    return filter(data.rows) do r
        (r["kind"], r["status"]) == ("data", "ok") &&
            all(d -> abs(parse(Float64, r[d])) ≤ STATIONARY_DRIFT, drifts)
    end
end

"The label of `row`: its method, kind, case, amplitude, ``n`` and seed."
function row_label(r)
    "$(r["method"]), $(r["kind"]), $(r["case"]), α = $(r["alpha"]), n = $(r["n"])" *
    (isempty(row_seed(r)) ? "" : ", seed $(row_seed(r))")
end

"""
    write_particles(io, data)

The particle half: the POD mode counts of the labelled and the sorted snapshots, and for each
frontend and ``n`` the median and range of its error beside the error of POD of the sorted
snapshots (`pod`) and of the labelled ones (`pod_labelled`) of the same particles. A frontend
whose median is not below the sorted POD's at any ``n`` is marked as failing at its particle
count.
"""
function write_particles(io, data)
    println(io, "## The particle half\n")
    println(io, "| snapshots | particles | threshold | POD modes |\n|:--|:--|:--|:--|")
    for r in data.pod_modes
        println(io, "| $(r["snapshots"]) | $(r["particles"]) | $(r["threshold"]) | ",
            "$(r["modes"]) |")
    end
    top = maximum(parse(Float64, r["threshold"]) for r in data.pod_modes)
    counts = unique(r["modes"]
    for r in data.pod_modes
    if parse(Float64, r["threshold"]) == top)
    println(io)
    length(counts) == 1 && println(io,
        "At the threshold $top the labelled and the sorted counts are equal: the count ",
        "saturates at the number of snapshots (K16), so that column does not separate the ",
        "two.\n")

    ok = filter(r -> r["status"] == "ok", data.frontends)
    pods = Dict((r["method"], r["n"], r["particles"]) =>
                    parse(Float64, r["relative_error"])
    for r in ok if r["method"] in ("pod", "pod_labelled"))
    frontends = unique(r["method"] for r in ok if r["method"] ∉ ("pod", "pod_labelled"))
    println(io, "| frontend | n | particles | error, median [range] | POD sorted | ",
        "POD labelled | below POD sorted |\n|:--|:--|:--|:--|:--|:--|:--|")
    failing = String[]
    for f in frontends
        below_any = false
        for n in sort(unique(parse(Int, r["n"]) for r in ok if r["method"] == f))
            group = filter(r -> r["method"] == f && r["n"] == string(n), ok)
            M = only(unique(r["particles"] for r in group))
            e = [parse(Float64, r["relative_error"]) for r in group]
            med = median_of(e)
            pod = get(pods, ("pod", string(n), M), nothing)
            below = pod !== nothing && med < pod
            below_any |= below
            println(
                io, "| $f | $n | $M | $(format_value(med)) [$(format_value(minimum(e))), ",
                "$(format_value(maximum(e)))] | $(format_value(pod)) | ",
                "$(format_value(get(pods, ("pod_labelled", string(n), M), nothing))) | ",
                below ? "yes" : "no", " |")
        end
        below_any || push!(failing, f)
    end
    M = join(unique(r["particles"] for r in ok if r["method"] ∉ ("pod", "pod_labelled")),
        ", ")
    println(io)
    println(io,
        isempty(failing) ?
        "No frontend fails at M = $M: each is below POD of the " *
        "sorted snapshots at some n." :
        "Failing at M = $M (not below POD of the sorted snapshots at any n): " *
        join(failing, ", ") * ".")
    println(io,
        "\nThe PIC run costs $(format_value(data.particle_cost["seconds_per_step"])) s per ",
        "step with $(data.particle_cost["particles"]) particles ",
        "($(data.particle_cost["note"])).\n")
end

"""
    write_findings(io, data)

The findings: the norm ratio of the canonical flow of the pulled-back Hamiltonian, with its grid,
as the measured failure of decided item 10; the stationary rows; the seeds of the weight-tied
network; the failed rows; and the rows whose rate is empty.
"""
function write_findings(io, data)
    h = data.hamiltonian
    println(io, "## Findings\n")
    println(io,
        "- **The pulled-back Hamiltonian (decided item 10), a finding for part K.** The ",
        "canonical flow of the discrete energy in the parity coordinates has ",
        "$(format_value(h["norm_ratio"])) times the norm of VlasovMethods' Arakawa field on ",
        "the $(h["nx"]) × $(h["nv"]) grid ($(h["case"]), α = $(h["alpha"]), t = $(h["t"])), so ",
        "the pulled-back Hamiltonian does not generate the Vlasov dynamics; I3 took the manifold ",
        "Galerkin projection of the Arakawa field in its place. Only this norm ratio carries ",
        "information, and it scales with h_x h_v, so it compares only on one grid (K10). The ",
        "file's cosine, $(format_value(h["cosine"])), is no evidence: it is a round-off zero ",
        "both for the canonical flow and for the plain gradient (K11).")
    stationary = stationary_rows(data)
    if !isempty(stationary)
        println(io, "- **Stationary rows.** In these rows every drift is at most ",
            "$(format_value(STATIONARY_DRIFT)): the reduced state does not move.")
        foreach(r -> println(io, "  - ", row_label(r)), stationary)
        any(r -> (r["method"], r["case"]) == ("weight-tied", "two_stream"), stationary) &&
            println(io, "  The cause of the weight-tied two-stream rows (I3): the SVD ",
                "frontend keeps cos kx and no sin kx, so the projected field at the initial ",
                "state is zero to round-off.")
    end
    tied = filter(r -> r["method"] == "weight-tied" && r["status"] == "ok", data.rows)
    groups = Dict{NTuple{4, String}, Vector{Any}}()
    for r in tied
        push!(get!(groups, (r["kind"], r["case"], r["alpha"], r["n"]), []), r)
    end
    function spread(group, column)
        v = [parse(Float64, r[column]) for r in group]
        return (maximum(v) - minimum(v)) / abs(median_of(v))
    end
    largest = isempty(groups) ? nothing :
              maximum(g -> spread(g, "loss_final"), values(groups))
    println(
        io, "- **The seeds of the weight-tied network** are near copies: its Stage I is ",
        "deterministic, so their range is about zero (K12). The largest spread of `loss_final` ",
        "over the seeds of one case, amplitude and n is $(format_value(largest)) relative.")
    wide = sort([k for (k, g) in groups if spread(g, "max_l2_error_fluct") > 0.01])
    isempty(wide) || println(io, "  The range of the error is wider than 1 % in: ",
        join(("$(k[2]), $(k[1]), α = $(k[3]), n = $(k[4])" for k in wide), "; "), ".")

    failed = vcat(
        [row_label(r) * ": " * r["status"] for r in data.rows if r["status"] != "ok"],
        ["$(r["method"]), n = $(r["n"]), seed $(r["seed"]): $(r["status"])"
         for r in data.frontends if r["status"] != "ok"])
    println(io, "- **Failed rows**, which enter no median and no comparison:",
        isempty(failed) ? " none." : "")
    foreach(f -> println(io, "  - ", f), failed)
    empty = [row_label(r) * ": " * r["rate_status"]
             for r in data.rows
             if (r["kind"], r["status"]) == ("data", "ok") &&
                    haskey(RATE_TOLERANCE, r["case"]) && r["rate_status"] != "ok"]
    println(io, "- **Empty rates**, which count as not better on the rate criterion:",
        isempty(empty) ? " none." : "")
    foreach(f -> println(io, "  - ", f), empty)
    println(io)
end

"""
    write_verdict(; results_dirs, out)

Write the verdict page of the benchmark to the Markdown file `out`, from the files of the
results directories `results_dirs` that [`verdict_inputs`](@ref) finds: the inputs with their
commits, the rules of the comparison, the comparison of every method per case, criterion and
``n``, the wins with their bounds, the particle half and the findings, and as its last line the
[`verdict_answer`](@ref). It returns that line.
"""
function write_verdict(; results_dirs, out::AbstractString)
    inputs = verdict_inputs(results_dirs)
    data = load_verdict(inputs)
    v = verdict_answer(data)

    mkpath(dirname(abspath(out)))
    open(out, "w") do io
        println(io, "# The verdict of the Vlasov reduction benchmark\n")
        println(io, "## Inputs\n\n| file | commit |\n|:--|:--|")
        for path in sort(vcat(collect(values(inputs.paths)),
            collect(values(inputs.full_order))))
            println(io, "| `$path` | `$(directory_commit(dirname(path)))` |")
        end
        println(io, "\n## How the page reads the runs\n")
        println(
            io, "- The autoencoder values are the median over the seeds, with the range ",
            "in brackets. A row whose status is not `ok` enters no median and no comparison.")
        println(
            io, "- \"Better\" means strictly smaller. A method beats the best baseline ",
            "(POD, PSD or DLRA) on a criterion when it is better at $BEAT_COUNT of the ",
            "$(length(VERDICT_DIMS)) n; the factor of a win is the smallest ratio of the best ",
            "baseline's value to the method's over the n at which it is better.")
        println(io, "- DLRA's error is read against its own full-rank solver (the column ",
            "`max_l2_error_fluct_full_rank`, relative to the fluctuation of that solver's run), ",
            "because its error against the full-order runs measures the difference of the two ",
            "discretisations. Its reference differs from the other methods'. Its rate is the fit ",
            "to its own run, and its full-rank solver passes the rate check of the DLRA script.")
        println(
            io, "- The rate criterion is the deviation of a method's fitted rate from the ",
            "fitted rate of the full-order run, relative to the latter; an empty rate counts as ",
            "not better. The sweep's criterion is the out-of-sample error: per seed, the larger ",
            "of the errors at the test amplitudes.")
        dts = join(("`$f` $(format_value(dt))" for (f, dt) in sort(collect(data.cost_dt))), ", ")
        println(io,
            "- The online cost is per unit of simulated time: each cost file's seconds ",
            "per step divided by its `dt` ($dts), as a ratio to the full-order cost per unit ",
            "time of the `full_order_cost.toml` beside it. The Galerkin and manifold Galerkin ",
            "costs include the dual-number allocations of the Newton solve.\n")
        println(io, "| full-order cost file | seconds per step | dt |\n|:--|:--|:--|")
        for (csv, path) in sort(collect(inputs.full_order))
            f = data.full_order[csv]
            println(io, "| `$path` | $(format_value(f["seconds_per_step"])) | $(f["dt"]) |")
        end
        println(io, "\n## What the error criterion can show\n\n", ERROR_CRITERION_NOTE, "\n")

        println(io, "## Summary\n\n| case | criterion | ",
            join(VERDICT_AUTOENCODERS, " | "), " |\n|:--|:--|:--|:--|")
        comparisons = [(case, c, comparison(criterion_values(data, case, c[1])))
                       for case in VERDICT_CASES for c in case_criteria(case)]
        for (case, c, cmp) in comparisons
            cells = map(VERDICT_AUTOENCODERS) do a
                b = beats(cmp, a)
                b.beats ? "beats, ×$(format_value(b.factor))" :
                "$(b.count) of $(length(VERDICT_DIMS))"
            end
            println(io, "| $case | $(c[2]) | ", join(cells, " | "), " |")
        end
        println(io, "\n## The comparisons\n")
        for (case, c, cmp) in comparisons
            println(io, "### $case: $(c[2])\n")
            write_comparison(io, cmp)
        end

        println(io, "## Wins on the error criterion\n")
        println(
            io, "Bound A: the case has a rate and the autoencoder's fitted rate is within ",
            "the tolerance of the full-order run's rate ($(join(("$k $(format_value(100v)) %"
                                                                for (k, v) in sort(collect(RATE_TOLERANCE))), ", "))) at an n of the win. ",
            "Bound C: at some n ≤ 16 its error is at or below POD's at n = 32 in the same case.\n")
        if isempty(v.wins)
            println(io, "No autoencoder beats the best baseline on the error criterion.\n")
        else
            println(io, "| autoencoder | case | factor | A | C |\n|:--|:--|:--|:--|:--|")
            for w in v.wins
                println(
                    io, "| $(w.autoencoder) | $(w.case) | $(format_value(w.factor)) | ",
                    w.A ? "met" : "not met", " | ", w.C ? "met" : "not met", " |")
            end
            println(io)
        end

        write_particles(io, data)
        write_findings(io, data)
        print(io, v.answer, "\n")
    end

    return v.answer
end
