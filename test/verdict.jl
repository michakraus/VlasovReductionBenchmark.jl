using TOML
using VlasovReductionBenchmark
using VlasovReductionBenchmark: beats, comparison, criterion_values, csv_field, csv_fields,
                                directory_commit, load_verdict, verdict_answer,
                                verdict_inputs
using Test

include("helpers/thrown.jl")
include("../scripts/verdict.jl")

# The fixtures: four results directories, as the full modes of the scripts write them, with
# only the columns that the page reads. Every value is a default of `DEFAULTS`, unless the
# `over` of a fixture replaces it.

const DIMS = (4, 8, 16, 32)
const SEEDS = ("42", "43", "44")
const CASE_ALPHA = (linear_landau = "0.01", nonlinear_landau = "0.5", two_stream = "0.05")
const SWEEP_ALPHAS = ("0.015", "0.03")
const RATE_FULL = Dict("linear_landau" => -0.15, "two_stream" => 0.12)
const COLUMNS = ["method", "kind", "case", "alpha", "n", "seed", "status",
    "max_l2_error_fluct", "max_l2_error_fluct_full_rank", "rate", "rate_status",
    "rate_full_order", "energy_drift", "l1_drift", "l2_drift", "entropy_drift", "loss_final"]

# The default error relative to the fluctuation, the default relative deviation of the rate and
# the default drift of each method. DLRA's error against the full-order runs is far below every
# other error, so that a page that reads it makes DLRA the best baseline everywhere.
const DEFAULTS = Dict("POD" => (error = 1.0, rate = 0.10, drift = 1e-3),
    "PSD" => (error = 1.1, rate = 0.10, drift = 1e-3),
    "DLRA" => (error = 1.2, rate = 0.10, drift = 1e-3),
    "untied" => (error = 2.0, rate = 0.50, drift = 2e-3),
    "weight-tied" => (error = 2.0, rate = 0.50, drift = 2e-3))

# The rows of `method` with the `seeds`, each a `Dict`; `over[(method, case, n, seed)]` replaces
# columns of a data row, and `over[(method, "sweep", alpha, n, seed)]` those of a sweep row.
function method_rows(method, seeds, over)
    d = DEFAULTS[method]
    rows = Dict{String, String}[]
    for case in keys(CASE_ALPHA), n in DIMS, seed in seeds
        case = string(case)
        rated = haskey(RATE_FULL, case)
        row = Dict("method" => method, "kind" => "data", "case" => case,
            "alpha" => CASE_ALPHA[Symbol(case)], "n" => string(n), "seed" => seed,
            "status" => "ok", "rate_status" => rated ? "ok" : "none",
            "rate_full_order" => rated ? string(RATE_FULL[case]) : "",
            "rate" => rated ? string(RATE_FULL[case] * (1 + d.rate)) : "",
            "max_l2_error_fluct" => string(method == "DLRA" ? 0.1 : d.error),
            "max_l2_error_fluct_full_rank" => method == "DLRA" ? string(d.error) : "",
            "energy_drift" => string(d.drift), "l1_drift" => string(d.drift),
            "l2_drift" => string(d.drift), "entropy_drift" => string(d.drift),
            "loss_final" => "0.5")
        push!(rows, merge(row, get(over, (method, case, n, seed), Dict())))
    end
    for α in SWEEP_ALPHAS, n in DIMS, seed in seeds
        row = Dict("method" => method, "kind" => "sweep", "case" => "linear_landau",
            "alpha" => α, "n" => string(n), "seed" => seed, "status" => "ok",
            "max_l2_error_fluct" => string(method == "DLRA" ? 0.1 : d.error),
            "max_l2_error_fluct_full_rank" => method == "DLRA" ? string(d.error) : "",
            "loss_final" => "0.5")
        push!(rows, merge(row, get(over, (method, "sweep", α, n, seed), Dict())))
    end
    return rows
end

function write_rows(path, rows, columns)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(columns, ','))
        for r in rows
            println(io, join((csv_field(get(r, c, "")) for c in columns), ','))
        end
    end
end

# The cost file of the completed data rows `rows`, each with `seconds[(method, case, n, seed)]`
# per step or 1e-3, and with the method under the key `label`: `"method"` for the baselines,
# `"network"` with the seed for the autoencoders, and none for DLRA.
function write_cost(path, rows, dt, seconds, label)
    entries = map(filter(r -> (r["kind"], r["status"]) == ("data", "ok"), rows)) do r
        e = Dict{String, Any}("file" => "$(r["case"])_alpha$(r["alpha"]).h5",
            "n" => parse(Int, r["n"]), "seconds_per_step" => get(seconds,
                (r["method"], r["case"], parse(Int, r["n"]), r["seed"]), 1e-3))
        label == "method" && (e["method"] = r["method"])
        label == "network" &&
            (e["network"] = r["method"]; e["seed"] = parse(Int, r["seed"]))
        e
    end
    open(io -> TOML.print(io, Dict("dt" => dt, "rows" => entries)), path, "w")
end

function full_order_cost(path)
    open(
        io -> TOML.print(io, Dict("seconds_per_step" => 1e-4, "dt" => 0.05, "nx" => 64,
            "nv" => 128)), path, "w")
end

# The four results directories of a fixture under `root`, whose names carry their commits.
function fixture(root; over = Dict(), seconds = Dict(), frontends = nothing,
        pod_modes = (51, 81, 47, 81))
    dirs = (baselines = joinpath(root, "I2-aaaaaaa", "results"),
        autoencoders = joinpath(root, "I3-bbbbbbb", "results"),
        dlra = joinpath(root, "I6-ccccccc", "results"),
        particles = joinpath(root, "I5-ddddddd", "results"))

    rows = vcat(method_rows("POD", ("",), over), method_rows("PSD", ("",), over))
    write_rows(joinpath(dirs.baselines, "linear_baselines.csv"), rows,
        filter(!=("seed"), COLUMNS))
    write_cost(joinpath(dirs.baselines, "linear_baselines_cost.toml"), rows, 0.05, seconds,
        "method")
    full_order_cost(joinpath(dirs.baselines, "full_order_cost.toml"))

    rows = vcat(method_rows("untied", SEEDS, over), method_rows("weight-tied", SEEDS, over))
    write_rows(joinpath(dirs.autoencoders, "autoencoders.csv"), rows, COLUMNS)
    write_cost(joinpath(dirs.autoencoders, "autoencoders_cost.toml"), rows, 0.05, seconds,
        "network")
    full_order_cost(joinpath(dirs.autoencoders, "full_order_cost.toml"))
    open(
        io -> TOML.print(io,
            Dict("t" => 0.0, "alpha" => 0.01, "cosine" => 1.1e-17, "nx" => 64, "nv" => 128,
                "norm_ratio" => 159.48, "case" => "linear_landau")),
        joinpath(dirs.autoencoders, "autoencoders_hamiltonian.toml"), "w")

    rows = method_rows("DLRA", ("",), over)
    write_rows(joinpath(dirs.dlra, "dlra.csv"), rows, filter(!=("seed"), COLUMNS))
    write_cost(joinpath(dirs.dlra, "dlra_cost.toml"), rows, 0.025, seconds, "none")
    full_order_cost(joinpath(dirs.dlra, "full_order_cost.toml"))

    # the particle half: `:sort` below the sorted POD at every n, `:pairwise` at none
    frontends = something(frontends,
        vcat([("sort", n, s, "ok", 0.1) for n in DIMS for s in SEEDS],
            [("pairwise", n, s, "ok", 0.3) for n in DIMS for s in SEEDS],
            [("pod", n, "none", "ok", 0.2) for n in DIMS],
            [("pod_labelled", n, "none", "ok", 0.4) for n in DIMS]))
    mkpath(dirs.particles)
    open(joinpath(dirs.particles, "particle_frontends.csv"), "w") do io
        println(io, "method,n,seed,particles,steps,status,relative_error,seconds_per_step")
        foreach(f -> println(io, "$(f[1]),$(f[2]),$(f[3]),256,2000,$(f[4]),$(f[5]),0.03"),
            frontends)
    end
    open(joinpath(dirs.particles, "particle_pod_modes.csv"), "w") do io
        println(io, "snapshots,particles,threshold,modes")
        for ((snapshots, threshold), modes) in zip(
            (("labelled", 0.99), ("labelled", 0.9999),
                ("sorted", 0.99), ("sorted", 0.9999)),
            pod_modes)
            println(io, "$snapshots,4096,$threshold,$modes")
        end
    end
    open(
        io -> TOML.print(io,
            Dict("seconds_per_step" => 5.5e-4, "particles" => 4096, "dt" => 0.05,
                "note" => "the time of run!")),
        joinpath(dirs.particles, "particle_cost.toml"), "w")

    return collect(dirs)
end

loaded(dirs) = load_verdict(verdict_inputs(dirs))
function compared(dirs, case, criterion)
    comparison(criterion_values(loaded(dirs), case, criterion))
end
# The rows of every n of `method` in `case` with the columns `columns`, for each seed of `seeds`.
function everywhere(method, case, seeds, columns)
    Dict((method, case, n, s) => columns for n in DIMS for s in seeds)
end

@testset "CSV fields and commits" begin
    @test csv_fields("a,\"b, c\",,\"d \"\"e\"\"\"") == ["a", "b, c", "", "d \"e\""]
    @test csv_fields(csv_field("fit failed: [0.0, 40.0]") * ",x") ==
          ["fit failed: [0.0, 40.0]", "x"]
    @test directory_commit("/a/VlasovReductionBenchmark-I2-760c7a5/results") == "760c7a5"
    # the innermost component that names one
    @test directory_commit("/a/x-1111111/y-2222222/results") == "2222222"
    @test directory_commit("/a/run-0123456789abcdef0123456789abcdef01234567") ==
          "0123456789abcdef0123456789abcdef01234567"
    e = thrown(() -> directory_commit("/a/results-xyz/results"))
    @test e isa ErrorException
    @test occursin(
        r"^the path of the results directory /a/results-xyz/results names no commit: no component of it ends in -<commit>$",
        e.msg)
end

@testset "The inputs" begin
    mktempdir() do root
        dirs = fixture(root)
        inputs = verdict_inputs(dirs)
        @test inputs.paths["dlra.csv"] == joinpath(dirs[3], "dlra.csv")
        @test inputs.paths["autoencoders_hamiltonian.toml"] ==
              joinpath(dirs[2], "autoencoders_hamiltonian.toml")
        @test inputs.full_order["autoencoders.csv"] ==
              joinpath(dirs[2], "full_order_cost.toml")
        @test inputs.commits[inputs.paths["particle_cost.toml"]] == "ddddddd"

        # a missing input file stops the script with a message naming the file
        e = thrown(() -> verdict_inputs(dirs[[1, 2, 4]]))
        @test e isa ErrorException
        @test occursin(
            r"^missing input file dlra\.csv: none of the results directories .+ holds it$",
            e.msg)
        rm(joinpath(dirs[3], "dlra_cost.toml"))
        e = thrown(() -> verdict_inputs(dirs))
        @test e.msg == "missing input file $(joinpath(dirs[3], "dlra_cost.toml"))"
    end
    mktempdir() do root
        dirs = fixture(root)
        rm(joinpath(dirs[2], "full_order_cost.toml"))
        e = thrown(() -> verdict_inputs(dirs))
        @test e.msg == "missing input file $(joinpath(dirs[2], "full_order_cost.toml"))"
        # a file in two directories is ambiguous
        cp(joinpath(dirs[3], "dlra.csv"), joinpath(dirs[1], "dlra.csv"))
        e = thrown(() -> verdict_inputs(dirs))
        @test occursin(r"^dlra\.csv is in more than one results directory: ", e.msg)
    end
    mktempdir() do root
        dirs = fixture(root)
        # a row with a field too many
        path = joinpath(dirs[1], "linear_baselines.csv")
        lines = readlines(path)
        lines[3] *= ",x"
        write(path, join(lines, '\n') * '\n')
        e = thrown(() -> loaded(dirs))
        @test e isa ErrorException
        n = length(csv_fields(lines[1]))
        @test e.msg == "$path, line 3: $(n + 1) fields, not the $n of the header"
    end
    mktempdir() do root
        # a completed row without its cost row
        dirs = fixture(root)
        open(
            io -> TOML.print(io, Dict("dt" => 0.025)), joinpath(dirs[3], "dlra_cost.toml"),
            "w")
        e = thrown(() -> criterion_values(loaded(dirs), "two_stream", "cost"))
        @test e isa ErrorException
        @test e.msg == "no cost row for DLRA, two_stream_alpha0.05.h5, 4, "
    end
    mktempdir() do root
        # a dlra.csv without the column of the error against the full-rank solver
        dirs = fixture(root)
        path = joinpath(dirs[3], "dlra.csv")
        lines = readlines(path)
        header = csv_fields(lines[1])
        keep = findall(!=("max_l2_error_fluct_full_rank"), header)
        write(path, join([join(csv_fields(l)[keep], ',') for l in lines], '\n') * '\n')
        e = thrown(() -> loaded(dirs))
        @test e isa ErrorException
        @test e.msg ==
              "$path has no column max_l2_error_fluct_full_rank: it was written before scripts/dlra.jl wrote that column, so run it again"
    end
end

@testset "The median over the seeds" begin
    # the first seed is not the median: 3, 1, 2 against POD's 2.5
    over = Dict(("untied", "linear_landau", n, s) => Dict("max_l2_error_fluct" => v)
    for n in DIMS for (s, v) in zip(SEEDS, ("3.0", "1.0", "2.0")))
    mktempdir() do root
        cmp = compared(fixture(root; over), "linear_landau", "error")
        for c in cmp
            @test c.stats["untied"] == (median = 2.0, lo = 1.0, hi = 3.0)
            @test c.best == ("POD", 1.0)
        end
        # against POD at 2.5 the median is better and the first seed is not
        over2 = merge(over,
            everywhere("POD", "linear_landau", ("",), Dict("max_l2_error_fluct" => "2.5")),
            everywhere("PSD", "linear_landau", ("",), Dict("max_l2_error_fluct" => "2.6")),
            everywhere("DLRA", "linear_landau", ("",),
                Dict("max_l2_error_fluct_full_rank" => "2.7")))
        cmp = compared(fixture(joinpath(root, "b"); over = over2), "linear_landau", "error")
        @test all(c -> c.best == ("POD", 2.5) && c.better["untied"], cmp)
        @test beats(cmp, "untied") == (beats = true, count = 4, factor = 1.25)
    end
end

@testset "A failed row enters no median" begin
    # the failed seed would make the median 3.0, below POD's 3.2; the completed ones give 3.5
    over = merge(
        Dict(("weight-tied", "two_stream", n, s) => Dict("max_l2_error_fluct" => v)
        for n in DIMS for (s, v) in zip(SEEDS, ("3.0", "4.0", "0.5"))),
        Dict(("weight-tied", "two_stream", n, "44") =>
                 Dict("status" => "failed at step 3",
                     "max_l2_error_fluct" => "0.5") for n in DIMS),
        everywhere("POD", "two_stream", ("",), Dict("max_l2_error_fluct" => "3.2")),
        everywhere("PSD", "two_stream", ("",), Dict("max_l2_error_fluct" => "3.3")),
        everywhere("DLRA", "two_stream", ("",), Dict("max_l2_error_fluct_full_rank" => "3.4")))
    mktempdir() do root
        dirs = fixture(root; over)
        cmp = compared(dirs, "two_stream", "error")
        for c in cmp
            @test c.stats["weight-tied"] == (median = 3.5, lo = 3.0, hi = 4.0)
            @test !c.better["weight-tied"]
        end
        @test beats(cmp, "weight-tied").count == 0
        # the page lists the failed rows with their statuses
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = dirs, out)
        page = read(out, String)
        for n in DIMS
            @test occursin(
                "  - weight-tied, data, two_stream, α = 0.05, n = $n, seed 44: failed at step 3\n",
                page)
        end
    end
    # a failed sweep row removes its seed from the out-of-sample error
    over = Dict(
        ("untied", "sweep", "0.03", 4, "42") => Dict("status" => "failed at step 1"),
        ("untied", "sweep", "0.015", 4, "43") => Dict("max_l2_error_fluct" => "5.0"))
    mktempdir() do root
        cmp = compared(fixture(root; over), "sweep", "error")
        @test cmp[1].stats["untied"] == (median = 3.5, lo = 2.0, hi = 5.0)
        @test cmp[2].stats["untied"] == (median = 2.0, lo = 2.0, hi = 2.0)
    end
end

@testset "The online cost per unit of simulated time" begin
    # PSD and DLRA take the same seconds per step; DLRA's step is half as long
    seconds = merge(Dict(("PSD", "linear_landau", n, "") => 1e-3 for n in DIMS),
        Dict(("DLRA", "linear_landau", n, "") => 1e-3 for n in DIMS),
        Dict(("POD", "linear_landau", n, "") => 1.5e-3 for n in DIMS))
    mktempdir() do root
        cmp = compared(fixture(root; seconds), "linear_landau", "cost")
        # (1e-3 / 0.05) / (1e-4 / 0.05) = 10 and (1e-3 / 0.025) / (1e-4 / 0.05) = 20
        for c in cmp
            @test c.stats["PSD"].median ≈ 10 rtol=8eps()
            @test c.stats["DLRA"].median ≈ 20 rtol=8eps()
            @test c.stats["POD"].median ≈ 15 rtol=8eps()
            @test c.best[1] == "PSD"
        end
    end
    # each ratio divides by the full-order cost of its own results directory
    mktempdir() do root
        dirs = fixture(root)
        open(
            io -> TOML.print(io,
                Dict("seconds_per_step" => 2e-4, "dt" => 0.05, "nx" => 64, "nv" => 128)),
            joinpath(dirs[2], "full_order_cost.toml"), "w")
        c = first(compared(dirs, "two_stream", "cost"))
        @test c.stats["untied"].median ≈ 5 rtol=8eps()
        @test c.stats["POD"].median ≈ 10 rtol=8eps()
    end
end

@testset "Three of the four n" begin
    # the untied network is better than POD at two n only: no win
    over = Dict(("untied", "nonlinear_landau", n, s) => Dict("max_l2_error_fluct" => "0.5")
    for n in (4, 8) for s in SEEDS)
    mktempdir() do root
        cmp = compared(fixture(root; over), "nonlinear_landau", "error")
        @test [c.better["untied"] for c in cmp] == [true, true, false, false]
        @test beats(cmp, "untied") == (beats = false, count = 2, factor = 2.0)
        # at three n it wins
        over3 = merge(over,
            Dict(("untied", "nonlinear_landau", 16, s) =>
                     Dict("max_l2_error_fluct" => "0.8")
            for s in SEEDS))
        cmp = compared(fixture(joinpath(root, "b"); over = over3), "nonlinear_landau",
            "error")
        @test beats(cmp, "untied") == (beats = true, count = 3, factor = 1.25)
    end
end

@testset "DLRA against its own full-rank solver" begin
    mktempdir() do root
        # DLRA's error against the full-order runs is 0.1, against its full-rank solver 1.2
        for c in compared(fixture(root), "linear_landau", "error")
            @test c.stats["DLRA"].median == 1.2
            @test c.best == ("POD", 1.0)
        end
        for c in compared(fixture(root), "sweep", "error")
            @test c.stats["DLRA"].median == 1.2
        end
    end
end

@testset "Ties and empty rates" begin
    over = merge(
        # a tie with the best baseline is not better
        everywhere("weight-tied", "linear_landau", SEEDS, Dict("max_l2_error_fluct" => "1.0")),
        # an empty rate counts as not better, even against a worse baseline rate
        Dict(("untied", "two_stream", n, s) =>
                 Dict("rate" => "", "rate_status" => "fit failed: 1 maximum, [5.0, 25.0]")
        for n in DIMS for s in SEEDS))
    mktempdir() do root
        dirs = fixture(root; over)
        @test !any(c -> c.better["weight-tied"],
            compared(dirs, "linear_landau", "error"))
        cmp = compared(dirs, "two_stream", "rate")
        for c in cmp
            @test c.stats["untied"].median == Inf
            @test c.stats["weight-tied"].median ≈ 0.5 rtol=1e-12
            @test c.best[2] ≈ 0.1 rtol=1e-12
            @test !c.better["untied"]
        end
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = dirs, out)
        page = read(out, String)
        @test occursin(
            "  - untied, data, two_stream, α = 0.05, n = 4, seed 42: fit failed: 1 maximum, [5.0, 25.0]\n",
            page)
        # nonlinear Landau damping has no rate criterion
        @test occursin("### two_stream: rate", page)
        @test !occursin("### nonlinear_landau: rate", page)
    end
    # a baseline whose error is NaN is not the best baseline
    over = everywhere("POD", "linear_landau", ("",), Dict("max_l2_error_fluct" => "NaN"))
    mktempdir() do root
        @test all(c -> c.best == ("PSD", 1.1), compared(
            fixture(root; over), "linear_landau",
            "error"))
    end
end

# A fixture in which the weight-tied network beats the best baseline on the error criterion in
# linear Landau damping by a factor of 2: error 0.5 at every n, against POD's 1.0; POD at n = 32
# is `pod32`, and the network's rate deviates by `rate` from the full-order rate.
function winning(root; pod32 = "1.0", rate = 0.01)
    over = merge(
        everywhere("weight-tied", "linear_landau", SEEDS,
            Dict("max_l2_error_fluct" => "0.5",
                "rate" => string(RATE_FULL["linear_landau"] * (1 + rate)))),
        Dict(("POD", "linear_landau", 32, "") => Dict("max_l2_error_fluct" => pod32)))
    return fixture(root; over)
end

@testset "The answer and the bounds" begin
    mktempdir() do root
        # no win: the closest margin
        v = verdict_answer(loaded(fixture(joinpath(root, "none"))))
        @test isempty(v.wins)
        @test v.answer ==
              "**Does part G go on?** No: no autoencoder beats the best baseline on the error criterion in any grid case. The closest margin: the untied autoencoder in linear_landau is better at 0 of the 4 n, and its error is 2 times the best baseline's at n = 4."

        # a win that meets both bounds
        v = verdict_answer(loaded(winning(joinpath(root, "both"))))
        @test only(v.wins) == (case = "linear_landau", autoencoder = "weight-tied",
            factor = 2.0, A = true, C = true)
        @test v.answer ==
              "**Does part G go on?** Yes: the weight-tied autoencoder beats the best baseline on the error criterion in linear_landau by a factor of 2, and meets bounds A and C."

        # bound A: the rate deviates by 3 %, above the 2 % of Landau damping
        v = verdict_answer(loaded(winning(joinpath(root, "a"); rate = 0.03)))
        @test (only(v.wins).A, only(v.wins).C) == (false, true)
        @test v.answer ==
              "**Does part G go on?** No: no win meets both bounds. The closest win, the weight-tied autoencoder in linear_landau by a factor of 2, misses bound A."
        # just below 2 % it is met
        v = verdict_answer(loaded(winning(joinpath(root, "a2"); rate = 0.0199)))
        @test only(v.wins).A

        # bound C: POD at n = 32 is below the network's error at every n ≤ 16
        v = verdict_answer(loaded(winning(joinpath(root, "c"); pod32 = "0.4")))
        @test (only(v.wins).A, only(v.wins).C) == (true, false)
        @test endswith(v.answer, "misses bound C.")
        # at POD's error exactly it is met
        v = verdict_answer(loaded(winning(joinpath(root, "c2"); pod32 = "0.5")))
        @test only(v.wins).C
        v = verdict_answer(loaded(winning(joinpath(root, "ac"); pod32 = "0.4", rate = 0.1)))
        @test endswith(v.answer, "misses bound A and C.")
    end
    # a win in a case without a rate never meets bound A
    over = everywhere("untied", "nonlinear_landau", SEEDS, Dict("max_l2_error_fluct" => "0.5"))
    mktempdir() do root
        w = only(verdict_answer(loaded(fixture(root; over))).wins)
        @test (w.case, w.A, w.C) == ("nonlinear_landau", false, true)
    end
    # bound A counts only an n of the win: here the rate is within 2 % at n = 32 alone, where
    # POD at 0.4 is better than the network
    good = string(RATE_FULL["linear_landau"] * 1.01)
    bad = string(RATE_FULL["linear_landau"] * 1.10)
    over = merge(
        Dict(("weight-tied", "linear_landau", n, s) => Dict("max_l2_error_fluct" => "0.5",
                 "rate" => n == 32 ? good : bad) for n in DIMS for s in SEEDS),
        Dict(("POD", "linear_landau", 32, "") => Dict("max_l2_error_fluct" => "0.4")))
    mktempdir() do root
        w = only(verdict_answer(loaded(fixture(root; over))).wins)
        @test (w.A, w.C) == (false, false)
    end
    # bound A reads the median rate of the seeds, neither the best nor the worst seed: with one
    # seed of three within 2 % it is missed, with two of three it is met
    for (k, met) in ((1, false), (2, true))
        over = Dict(("weight-tied", "linear_landau", n, s) => Dict(
                        "max_l2_error_fluct" => "0.5", "rate" => i ≤ k ? good : bad)
        for n in DIMS for (i, s) in enumerate(SEEDS))
        mktempdir() do root
            w = only(verdict_answer(loaded(fixture(root; over))).wins)
            @test (w.A, w.C) == (met, true)
        end
    end
    # bound C counts only n ≤ 16: below POD's n = 32 error at n = 32 alone is not enough
    over = merge(
        Dict(("weight-tied", "linear_landau", n, s) => Dict(
                 "max_l2_error_fluct" => n == 32 ? "0.3" : "0.5",
                 "rate" => good) for n in DIMS for s in SEEDS),
        Dict(("POD", "linear_landau", 32, "") => Dict("max_l2_error_fluct" => "0.4")))
    mktempdir() do root
        w = only(verdict_answer(loaded(fixture(root; over))).wins)
        @test (w.A, w.C) == (true, false)
    end
    # of two wins that miss a bound, the closest is the one that meets more bounds, before the
    # larger factor: the untied network in nonlinear Landau damping wins by 4 and meets no
    # bound; the weight-tied network in two-stream wins by 2 and meets A and not C
    over = merge(
        everywhere("untied", "nonlinear_landau", SEEDS, Dict("max_l2_error_fluct" => "0.25")),
        Dict(("weight-tied", "two_stream", n, s) => Dict("max_l2_error_fluct" => "0.5",
                 "rate" => string(RATE_FULL["two_stream"] * 1.01)) for n in DIMS
        for s in SEEDS),
        Dict(("POD", "two_stream", 32, "") => Dict("max_l2_error_fluct" => "0.4")),
        Dict(("POD", "nonlinear_landau", 32, "") => Dict("max_l2_error_fluct" => "0.2")))
    mktempdir() do root
        v = verdict_answer(loaded(fixture(root; over)))
        @test sort([(w.autoencoder, w.factor, w.A, w.C) for w in v.wins]) ==
              [("untied", 4.0, false, false), ("weight-tied", 2.0, true, false)]
        @test v.answer ==
              "**Does part G go on?** No: no win meets both bounds. The closest win, the weight-tied autoencoder in two_stream by a factor of 2, misses bound C."
    end
    # the closest margin is the one better at the most n, before the smaller ratio: the
    # weight-tied network is better in two-stream at two n by 2, the untied one in linear
    # Landau damping at one n by 10
    over = merge(
        Dict(("weight-tied", "two_stream", n, s) => Dict(
                 "max_l2_error_fluct" => n == 4 ? "0.6" : "0.5")
        for n in (4, 8) for s in SEEDS),
        Dict(("untied", "linear_landau", 4, s) => Dict("max_l2_error_fluct" => "0.1")
        for s in SEEDS))
    mktempdir() do root
        @test verdict_answer(loaded(fixture(root; over))).answer ==
              "**Does part G go on?** No: no autoencoder beats the best baseline on the error criterion in any grid case. The closest margin: the weight-tied autoencoder in two_stream is better at 2 of the 4 n, and its error is 0.5 times the best baseline's at n = 8."
    end
    # of two wins that meet both bounds, the answer names the one of the larger factor
    over = merge(
        everywhere("weight-tied", "linear_landau", SEEDS,
            Dict("max_l2_error_fluct" => "0.5",
                "rate" => string(RATE_FULL["linear_landau"] * 1.01))),
        everywhere("untied", "two_stream", SEEDS,
            Dict("max_l2_error_fluct" => "0.25",
                "rate" => string(RATE_FULL["two_stream"] * 1.01))))
    mktempdir() do root
        v = verdict_answer(loaded(fixture(root; over)))
        @test length(v.wins) == 2 && all(w -> w.A && w.C, v.wins)
        @test v.answer ==
              "**Does part G go on?** Yes: the untied autoencoder beats the best baseline on the error criterion in two_stream by a factor of 4, and meets bounds A and C."
    end
    # at the same count the smaller ratio: the untied network nearly ties POD in two-stream
    over = everywhere("untied", "two_stream", SEEDS, Dict("max_l2_error_fluct" => "1.001"))
    mktempdir() do root
        @test endswith(verdict_answer(loaded(fixture(root; over))).answer,
            "the untied autoencoder in two_stream is better at 0 of the 4 n, and its error is 1.001 times the best baseline's at n = 4.")
    end
    # with every row of the autoencoders failed there is no margin
    over = Dict{Any, Any}((a, case, n, s) => Dict("status" => "failed at step 1")
    for a in ("untied", "weight-tied")
    for case in ("linear_landau", "nonlinear_landau", "two_stream")
    for n in DIMS for s in SEEDS)
    for a in ("untied", "weight-tied"), α in SWEEP_ALPHAS, n in DIMS, s in SEEDS
        over[(a, "sweep", α, n, s)] = Dict("status" => "failed at step 1")
    end
    mktempdir() do root
        @test verdict_answer(loaded(fixture(root; over))).answer ==
              "**Does part G go on?** No: no autoencoder has a completed row on the error criterion."
    end
end

@testset "The page" begin
    over = merge(
        # stationary rows of the weight-tied network in the two-stream case
        Dict(("weight-tied", "two_stream", n, s) => Dict("energy_drift" => "0.0",
                 "l1_drift" => "1.4e-16", "l2_drift" => "0.0", "entropy_drift" => "0.0")
        for n in (4, 8) for s in SEEDS),
        # a row with one drift at zero only is not stationary
        Dict(("untied", "two_stream", 16, "42") => Dict("energy_drift" => "0.0")),
        # seeds whose errors differ by more than 1 %, and by less
        Dict(
            ("weight-tied", "two_stream", 32, "44") =>
                Dict("max_l2_error_fluct" => "25.0"),
            ("weight-tied", "linear_landau", 8, "44") =>
                Dict("max_l2_error_fluct" => "2.008")))
    mktempdir() do root
        dirs = winning(root)
        out = joinpath(root, "page", "verdict.md")
        answer = write_verdict(; results_dirs = dirs, out)
        page = read(out, String)
        lines = split(chomp(page), '\n')
        @test lines[end] == answer
        @test startswith(answer, "**Does part G go on?** Yes")
        @test occursin(VlasovReductionBenchmark.ERROR_CRITERION_NOTE, page)
        @test occursin(
            "POD is the L²-optimal linear reconstruction at each n, so PSD, a restricted linear family, cannot beat it on the error criterion. The SAEs are nonlinear, so no such bound orders them against POD, but the symplectic constraint restricts them as PSD's restricts POD; a non-symplectic autoencoder, which the benchmark does not include, would separate the gain of nonlinearity from the cost of structure. The energy and Casimir drifts are the criteria where structure is expected to show, and the time spans of these cases are short.",
            page)
        # every input file with the commit of its directory
        for (dir, commit) in zip(dirs, ("aaaaaaa", "bbbbbbb", "ccccccc", "ddddddd"))
            for file in readdir(dir)
                @test occursin("| `$(joinpath(dir, file))` | `$commit` |", page)
            end
        end
        @test occursin("| weight-tied | linear_landau | 2 | met | met |", page)
        @test occursin(
            "| linear_landau | maximal error relative to the fluctuation | 0 of 4 | beats, ×2 |",
            page)
        @test occursin(
            "| 4 | 1 | 1.1 | 1.2 | 2 [2, 2] | 0.5 [0.5, 0.5] | POD 1 | no | yes (×2) |",
            page)
        @test occursin("does not generate the Vlasov dynamics", page)
        @test occursin(
            "159.5 times the norm of VlasovMethods' Arakawa field on the 64 × 128 grid",
            page)
        @test occursin("DLRA's error is read against its own full-rank solver", page)
        @test occursin("include the dual-number allocations of the Newton solve", page)
        @test occursin("`dlra_cost.toml` 0.025", page)
        # the particle half: `:pairwise` is not below the sorted POD at any n
        @test occursin("| pairwise | 8 | 256 | 0.3 [0.3, 0.3] | 0.2 | 0.4 | no |", page)
        @test occursin("| sort | 8 | 256 | 0.1 [0.1, 0.1] | 0.2 | 0.4 | yes |", page)
        @test occursin(
            "Failing at M = 256 (not below POD of the sorted snapshots at any n): pairwise.",
            page)
        @test occursin(
            "At the threshold 0.9999 the labelled and the sorted counts are equal",
            page)
        @test occursin(
            "- **Failed rows**, which enter no median and no comparison: none.\n",
            page)
        # the sweep rows of the fixture hold no rate, so bound A cannot hold in the sweep
        @test occursin(
            "- **Empty rates**, which count as not better on the rate criterion:\n" *
            "  - DLRA, sweep: 8 rows, no rate\n  - POD, sweep: 8 rows, no rate\n" *
            "  - PSD, sweep: 8 rows, no rate\n  - untied, sweep: 24 rows, no rate\n" *
            "  - weight-tied, sweep: 24 rows, no rate\n",
            page)
        @test occursin(
            "The sweep rows of the autoencoders hold no rate, so bound A cannot hold in the sweep.",
            page)
        # no factor of the page is infinite, so no note on a zero median
        @test !occursin("×∞", page)
        @test !occursin("A factor of ∞", page)
    end
    # sweep rows of the autoencoders that hold a rate: bound A can hold in the sweep, and only
    # the sweep rows of the baselines are empty
    rated = Dict((a, "sweep", α, n, s) => Dict("rate_status" => "ok", "rate" => "-0.15",
                     "rate_full_order" => "-0.15")
    for a in ("untied", "weight-tied") for α in SWEEP_ALPHAS for n in DIMS for s in SEEDS)
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; over = rated), out)
        page = read(out, String)
        @test !occursin("bound A cannot hold in the sweep", page)
        @test occursin("  - PSD, sweep: 8 rows, no rate\n\n", page)
        @test !occursin("untied, sweep:", page)
    end
    # a rate in the sweep rows of a baseline alone leaves bound A out of reach in the sweep
    rated = Dict(("POD", "sweep", α, n, "") =>
                     Dict("rate_status" => "ok", "rate" => "-0.15",
                         "rate_full_order" => "-0.15") for α in SWEEP_ALPHAS for n in DIMS)
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; over = rated), out)
        @test occursin("bound A cannot hold in the sweep", read(out, String))
    end
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; over), out)
        page = read(out, String)
        for n in (4, 8), s in SEEDS

            @test occursin(
                "  - weight-tied, data, two_stream, α = 0.05, n = $n, seed $s\n",
                page)
        end
        @test !occursin("two_stream, α = 0.05, n = 16, seed 42\n", page)
        @test occursin("The cause of the weight-tied two-stream rows (I3)", page)
        # the zero energy drift of the stationary rows gives a factor ∞, and its table says
        # where it comes from; the L¹ drift of 1.4e-16 gives a finite factor and no note
        section(title) = first(split(last(split(page, "### $title\n")), "### "))
        note = "- A factor of ∞ comes from a median of zero. The *Stationary rows* of the " *
               "*Findings* list the rows whose reduced state does not move.\n"
        @test occursin("yes (×∞)", section("two_stream: energy drift"))
        @test occursin(note, section("two_stream: energy drift"))
        @test !occursin("×∞", section("two_stream: L¹ drift"))
        @test !occursin(note, section("two_stream: L¹ drift"))
        @test occursin(
            "The range of the error is wider than 1 % in: two_stream, data, α = 0.05, n = 32.\n",
            page)
    end
    # stationary rows of another network carry no cause
    over = Dict(("untied", "nonlinear_landau", 4, s) => Dict("energy_drift" => "0.0",
                    "l1_drift" => "0.0", "l2_drift" => "0.0", "entropy_drift" => "0.0")
    for s in SEEDS)
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; over), out)
        page = read(out, String)
        @test occursin("  - untied, data, nonlinear_landau, α = 0.5, n = 4, seed 42\n", page)
        @test !occursin("The cause of", page)
    end
    # a zero median energy drift in rows that are not stationary: the note names no finding
    over = Dict(("weight-tied", "two_stream", n, s) => Dict("energy_drift" => "0.0")
    for n in (4, 8) for s in SEEDS)
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; over), out)
        page = read(out, String)
        @test occursin("yes (×∞)", page)
        @test occursin("- A factor of ∞ comes from a median of zero.\n", page)
        @test !occursin("Stationary rows", page)
    end
    # every baseline without a rate: the factor ∞ of the rate comes from an infinite best
    # baseline, not from a median of zero, so no note
    over = Dict((m, "linear_landau", n, "") => Dict("rate_status" => "fit failed")
    for m in VlasovReductionBenchmark.VERDICT_BASELINES for n in DIMS)
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; over), out)
        page = read(out, String)
        @test occursin(
            "| 4 | ∞ | ∞ | ∞ | 0.5 [0.5, 0.5] | 0.5 [0.5, 0.5] | POD ∞ | yes (×∞) |",
            page)
        @test !occursin("A factor of ∞", page)
    end
end

@testset "The particle frontends" begin
    # a frontend below the sorted POD at one n only does not fail; failed rows do not enter
    frontends = vcat(
        # at n = 8 a tie with the sorted POD, which is not below it
        [("slot", n, s, "ok", n == 16 ? 0.1 : n == 8 ? 0.2 : 0.3) for n in DIMS
         for s in SEEDS],
        [("slot", 4, "45", "failed at step 2", 0.01)],
        [("pod", n, "none", "ok", 0.2) for n in DIMS],
        [("pod_labelled", n, "none", "ok", 0.4) for n in DIMS])
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; frontends), out)
        page = read(out, String)
        @test occursin("No frontend fails at M = 256", page)
        @test occursin("| slot | 4 | 256 | 0.3 [0.3, 0.3] | 0.2 | 0.4 | no |", page)
        @test occursin("| slot | 8 | 256 | 0.2 [0.2, 0.2] | 0.2 | 0.4 | no |", page)
        @test occursin("| slot | 16 | 256 | 0.1 [0.1, 0.1] | 0.2 | 0.4 | yes |", page)
        @test occursin("  - slot, n = 4, seed 45: failed at step 2\n", page)
    end
    # counts that differ at the highest threshold carry no note of saturation
    mktempdir() do root
        out = joinpath(root, "verdict.md")
        write_verdict(; results_dirs = fixture(root; pod_modes = (51, 80, 47, 81)), out)
        page = read(out, String)
        @test occursin("| labelled | 4096 | 0.9999 | 80 |", page)
        @test !occursin("the count saturates", page)
    end
end

@testset "The script" begin
    @test parse_options(["--results-dir", "a", "b", "--out", "v.md"]) ==
          (results_dirs = ["a", "b"], out = "v.md")
    @test parse_options(["--out", "v.md", "--results-dir", "a", "--results-dir", "b"]) ==
          (results_dirs = ["a", "b"], out = "v.md")
    usage = r"^usage: julia --project scripts/verdict\.jl --results-dir <r>\.\.\. --out <file>$"
    for args in (["--results-dir", "a"], ["--out", "v.md"], [
        "--results-dir", "--out", "v.md"],
        ["--results-dir", "a", "--out"], ["--results-dir", "a", "--out", "v.md", "--x"],
        ["--results-dir", "a", "--results-dir", "--out", "v.md"])
        e = thrown(() -> parse_options(args))
        @test e isa ArgumentError
        @test occursin(usage, e.msg)
    end
    mktempdir() do root
        dirs = fixture(root)
        out = joinpath(root, "results", "verdict.md")
        printed = joinpath(root, "printed.txt")
        open(
            io -> redirect_stdout(() -> main(["--results-dir", dirs..., "--out", out]), io),
            printed, "w")
        @test chomp(read(printed, String)) == last(readlines(out))
    end
end
