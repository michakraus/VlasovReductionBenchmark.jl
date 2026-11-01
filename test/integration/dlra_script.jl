using TOML
using VlasovReductionBenchmark
using VlasovReductionBenchmark: background, fitted_rate, max_drift, read_run,
                                relative_l2_errors, reduced_rate
using Test

include("../helpers/thrown.jl")
include("../../scripts/dlra.jl")

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/dlra\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
    for args in (["--runs-dir", "a"], ["--runs-dir", "a", "--results-dir", "b", "--x"],
        ["--runs-dir", "a", "--results-dir"])
        e = thrown(() -> parse_options(args))
        @test e isa ArgumentError
        @test occursin(usage, e.msg)
    end
end

# The header and the rows of a CSV file without quoted fields, as vectors of strings.
function read_csv(path)
    lines = readlines(path)
    return split(lines[1], ','), [split(l, ',') for l in lines[2:end]]
end

# The cases and amplitudes of the rows: the three data runs and the two test amplitudes.
const RUNS = (("data", "linear_landau", "0.01"), ("data", "nonlinear_landau", "0.5"),
    ("data", "two_stream", "0.05"), ("sweep", "linear_landau", "0.015"),
    ("sweep", "linear_landau", "0.03"))

@testset "Smoke run of the script" begin
    mktempdir() do root
        runs, results = joinpath(root, "runs"), joinpath(root, "results")
        redirect_stdout(devnull) do
            run_full_order(; runs_dir = runs, results_dir = results, smoke = true)
        end
        output = joinpath(root, "output.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                main(["--runs-dir", runs, "--results-dir", results, "--smoke"])
            end
        end
        printed = read(output, String)
        # the checks run before the measurement
        for name in ("linear_landau", "nonlinear_landau", "two_stream")
            @test occursin("full-rank check $name on 32 × 64", printed)
        end
        @test occursin("rate check full-rank solver linear Landau γ", printed)
        @test occursin("rate check DLRA r = 16 linear Landau γ", printed)

        header, rows = read_csv(joinpath(results, "dlra.csv"))
        col(name) = findfirst(==(name), header)
        @test header == collect(VlasovReductionBenchmark.DLRA_COLUMNS)
        @test all(length(r) == length(header) for r in rows)
        @test sort([(r[col("kind")], r[col("case")], r[col("alpha")], r[col("n")])
                    for r in rows]) ==
              sort([(k, c, a, n) for (k, c, a) in RUNS for n in ("4", "8")])
        @test all(r[col("method")] == "DLRA" for r in rows)
        # every DLRA run of the smoke mode completes (measured)
        @test all(r[col("status")] == "ok" for r in rows)
        @test occursin("$(length(rows)) rows, 0 failed", printed)

        for r in rows
            n = parse(Int, r[col("n")])
            @test parse(Int, r[col("state_size")]) == n * (32 + 64) + n^2
            errors = parse.(Float64, split(r[col("l2_errors")], ';'))
            # one error per snapshot, every 0.5 up to t = 40 or t = 60
            tend = r[col("case")] == "linear_landau" ? 40 : 60
            @test length(errors) == 2tend + 1
            @test maximum(errors) == parse(Float64, r[col("l2_error_max")])
            fluct = parse.(Float64, split(r[col("l2_error_fluct")], ';'))
            @test length(fluct) == length(errors)
            @test maximum(fluct) == parse(Float64, r[col("max_l2_error_fluct")])
            @test parse(Float64, r[col("offline_seconds")]) > 0
            @test parse(Float64, r[col("online_seconds_per_step")]) > 0
            @test parse(Float64, r[col("online_ratio")]) > 0
            @test parse(Float64, r[col("mass_drift")]) ≥ 0
        end

        # the online cost is measured in a fresh process
        cost = TOML.parsefile(joinpath(results, "dlra_cost.toml"))
        @test cost["pid"] != getpid()
        @test (cost["steps"], cost["warmup"], cost["dt"]) == (100, 10, 0.025)
        @test !isempty(cost["uptime"])
        full_seconds = TOML.parsefile(joinpath(results, "full_order_cost.toml"))["seconds_per_step"]
        @test cost["full_order_seconds_per_step"] == full_seconds
        @test length(cost["rows"]) == length(rows)
        for c in cost["rows"]
            r = only(filter(
                r -> (r[col("case")] * "_alpha" * r[col("alpha")] * ".h5",
                    r[col("n")]) == (c["file"], string(c["n"])),
                rows))
            seconds = parse(Float64, r[col("online_seconds_per_step")])
            @test seconds == c["seconds_per_step"]
            @test parse(Float64, r[col("online_ratio")]) ≈ seconds / full_seconds rtol=8eps()
        end

        # rows against DLRA runs of their own: rank 4 and rank 8 of the two-stream run, whose
        # largest error is not at the first snapshot, and the sweep at α = 0.03
        row(kind, case, alpha, n) = only(filter(
            r -> (r[col("kind")], r[col("case")], r[col("alpha")], r[col("n")]) ==
                 (kind, case, alpha, n), rows))
        value(r, name) = parse(Float64, r[col(name)])
        window = (0.0, 25.0)
        for (kind, file, n) in (("data", "two_stream_alpha0.05.h5", 4),
            ("data", "two_stream_alpha0.05.h5", 8),
            ("sweep", "linear_landau_alpha0.03.h5", 8))
            run = read_run(joinpath(runs, file))
            S = run.snapshots
            dlra = run_dlra(run.case, 32, 64, n)
            full = run_full_rank(run.case, 32, 64)
            r = row(kind, run.case.name, string(run.case.α), string(n))
            errors = relative_l2_errors(dlra.snapshots, S)
            fluct = relative_l2_errors(dlra.snapshots, S, background(S[:, :, 1]))
            @test argmax(errors) > 1
            @test value(r, "l2_error_max") ≈ maximum(errors) rtol=8eps()
            @test value(r, "max_l2_error_fluct") ≈ maximum(fluct) rtol=8eps()
            # the written series, snapshot by snapshot
            @test all(isapprox.(parse.(Float64, split(r[col("l2_errors")], ';')), errors;
                rtol = 8eps()))
            @test all(isapprox.(
                parse.(Float64, split(r[col("l2_error_fluct")], ';')), fluct;
                rtol = 8eps()))
            # the rate of the full-order run, which differs from the DLRA rate on these rows
            @test value(r, "rate_full_order") ≈
                  fitted_rate(run.case.name, run.times, run.E1, window) rtol=8eps()
            @test value(r, "full_rank_l2_error_max") ≈
                  maximum(relative_l2_errors(full.snapshots, S)) rtol=8eps()
            @test value(r, "max_l2_error_fluct_full_rank") ≈
                  maximum(relative_l2_errors(dlra.snapshots, full.snapshots,
                background(full.snapshots[:, :, 1]))) rtol=8eps()
            for (name, series) in (("energy_drift", dlra.energy), ("l1_drift", dlra.l1),
                ("l2_drift", dlra.l2), ("entropy_drift", dlra.entropy),
                ("mass_drift", dlra.mass))
                @test value(r, name) ≈ max_drift(series) rtol=8eps()
            end
            @test parse(Int, r[col("nonpositive_nodes")]) == maximum(dlra.nonpositive)
            rate, rate_status = reduced_rate(run.case.name, dlra.times, dlra.E1, window)
            @test r[col("rate_status")] == rate_status
            rate === nothing || @test value(r, "rate") ≈ rate rtol=8eps()
        end
        # the rows of one run share the error of its full-rank solver; the runs differ in it
        full_errors = [row(k, c, a, "4")[col("full_rank_l2_error_max")]
                       for (k, c, a) in RUNS]
        @test all(row(k, c, a, "8")[col("full_rank_l2_error_max")] == e
        for ((k, c, a), e) in zip(RUNS, full_errors))
        @test allunique(full_errors)
        # nonlinear Landau damping has no fitted rate of the full-order run
        @test row("data", "nonlinear_landau", "0.5", "4")[col("rate_full_order")] == ""
        # each run measures the wall time of its own initial factorisation
        @test allunique(r[col("offline_seconds")] for r in rows)
        # the rows of the two ranks differ in their error
        @test value(row("data", "two_stream", "0.05", "4"), "l2_error_max") !=
              value(row("data", "two_stream", "0.05", "8"), "l2_error_max")

        # every DLRA run broken at step 2: each row records it, and the script goes on
        broken = joinpath(root, "broken")
        mkpath(broken)
        cp(joinpath(results, "full_order_cost.toml"), joinpath(broken, "full_order_cost.toml"))
        output = joinpath(root, "broken.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                run_dlra_benchmark(; runs_dir = runs, results_dir = broken, smoke = true,
                    perturb = (s, S) -> s == 2 && (S[1] = NaN))
            end
        end
        header, rows = read_csv(joinpath(broken, "dlra.csv"))
        @test length(rows) == 10
        for r in rows
            @test r[col("status")] == "failed at step 2"
            @test parse(Int, r[col("state_size")]) > 0
            @test r[col("full_rank_l2_error_max")] != ""
            for name in ("l2_error_max", "l2_errors", "l2_error_fluct", "max_l2_error_fluct",
                "rate", "energy_drift", "mass_drift", "max_l2_error_fluct_full_rank",
                "online_seconds_per_step", "online_ratio")
                @test r[col(name)] == ""
            end
        end
        @test occursin("10 rows, 10 failed", read(output, String))
        @test isempty(get(TOML.parsefile(joinpath(broken, "dlra_cost.toml")), "rows", []))

        # the grids of the cost file and of the run files are those of the mode
        e = thrown(() -> run_dlra_benchmark(; runs_dir = runs, results_dir = results))
        @test e isa ErrorException
        @test occursin(
            r"^the full-order cost is measured on 32 × 64, not on the 64 × 128 grid of this mode$",
            e.msg)
        wrong = joinpath(root, "wrong")
        mkpath(wrong)
        open(
            io -> TOML.print(io,
                Dict("nx" => 64, "nv" => 128, "seconds_per_step" => 1.0)),
            joinpath(wrong, "full_order_cost.toml"), "w")
        e = thrown(() -> redirect_stdout(devnull) do
            run_dlra_benchmark(; runs_dir = runs, results_dir = wrong)
        end)
        @test e isa ErrorException
        @test occursin(
            r"^linear_landau_alpha0\.01\.h5 holds a 32 × 64 run, not the 64 × 128 grid of this mode$",
            e.msg)
    end
end

@testset "Full run of the script" begin
    mktempdir() do root
        runs, results = joinpath(root, "runs"), joinpath(root, "results")
        redirect_stdout(devnull) do
            run_full_order(; runs_dir = runs, results_dir = results)
        end
        output = joinpath(root, "output.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                main(["--runs-dir", runs, "--results-dir", results])
            end
        end
        printed = read(output, String)
        @test occursin("rate check DLRA r = 16 linear Landau γ", printed)

        header, rows = read_csv(joinpath(results, "dlra.csv"))
        col(name) = findfirst(==(name), header)
        @test sort([(r[col("kind")], r[col("case")], r[col("alpha")], r[col("n")])
                    for r in rows]) ==
              sort([(k, c, a, n) for (k, c, a) in RUNS for n in ("4", "8", "16", "32")])
        for r in rows
            n = parse(Int, r[col("n")])
            @test parse(Int, r[col("state_size")]) == n * (64 + 128) + n^2
            @test r[col("status")] == "ok" ||
                  occursin(r"^failed at step \d+$", r[col("status")])
        end
        @test occursin("$(length(rows)) rows", printed)
    end
end
