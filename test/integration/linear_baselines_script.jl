using TOML
using VlasovReductionBenchmark
using VlasovReductionBenchmark: csv_field, reduced_rate
using Test

include("../helpers/thrown.jl")
include("../../scripts/linear_baselines.jl")

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/linear_baselines\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
    for args in (["--runs-dir", "a"], ["--runs-dir", "a", "--results-dir", "b", "--x"],
        ["--runs-dir", "a", "--results-dir"])
        e = thrown(() -> parse_options(args))
        @test e isa ArgumentError
        @test occursin(usage, e.msg)
    end
end

# The fields of one CSV line, where a quoted field may hold commas and doubled quotes.
function csv_fields(line)
    fields, field, quoted, i = String[], IOBuffer(), false, firstindex(line)
    while i ≤ lastindex(line)
        c = line[i]
        if quoted && c == '"'
            next = nextind(line, i)
            if next ≤ lastindex(line) && line[next] == '"'
                write(field, '"')
                i = next
            else
                quoted = false
            end
        elseif c == '"'
            quoted = true
        elseif c == ',' && !quoted
            push!(fields, String(take!(field)))
        else
            write(field, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(field)))
    return fields
end

# The header and the rows of a CSV file, as vectors of strings.
function read_csv(path)
    lines = readlines(path)
    return csv_fields(lines[1]), [csv_fields(l) for l in lines[2:end]]
end

@testset "CSV fields and reduced rates" begin
    @test csv_field("ok") == "ok"
    @test csv_field("a, b") == "\"a, b\""
    @test csv_field("say \"x\"") == "\"say \"\"x\"\"\""
    @test csv_fields(join(csv_field.(["a, b", "c", "say \"x\"", ""]), ',')) ==
          ["a, b", "c", "say \"x\"", ""]

    t = collect(0:0.05:40)
    E1 = exp.(-0.15 .* t) .* abs.(cos.(1.4 .* t))
    rate, status = reduced_rate("linear_landau", t, E1, (0.0, 40.0))
    @test rate ≈ -0.15 rtol=1e-2
    @test status == "ok"
    @test reduced_rate("nonlinear_landau", t, E1, (0.0, 40.0)) == (nothing, "none")
    # a damped E₁ without oscillation has no maxima to fit
    @test reduced_rate("linear_landau", t, exp.(-0.15 .* t), (0.0, 25.0)) == (nothing,
        "fit failed: the Landau fit found 0 maxima of log|E₁| in [0.0, 25.0]; it needs at least four")
    # an error that is not the fit's own propagates
    @test_throws DomainError reduced_rate("linear_landau", t, -E1, (0.0, 40.0))
end

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

        header, rows = read_csv(joinpath(results, "linear_baselines.csv"))
        col(name) = findfirst(==(name), header)
        @test header[1:5] == ["method", "kind", "case", "alpha", "n"]
        for name in ("status", "l2_error_max", "l2_errors", "l2_error_fluct",
            "max_l2_error_fluct", "rate", "rate_full_order",
            "energy_drift", "l1_drift", "l2_drift", "entropy_drift", "nonpositive_nodes",
            "projection_error", "offline_seconds", "online_seconds_per_step", "online_ratio")
            @test name in header
        end
        @test all(length(r) == length(header) for r in rows)

        data = filter(r -> r[col("kind")] == "data", rows)
        sweep = filter(r -> r[col("kind")] == "sweep", rows)
        # POD and PSD, three cases at their I1 amplitude, n ∈ {4, 8}
        @test sort([(r[1], r[3], r[4], r[5]) for r in data]) ==
              sort([(m, c, a, n) for m in ("POD", "PSD")
                    for (c, a) in (("linear_landau", "0.01"), ("nonlinear_landau", "0.5"),
            ("two_stream", "0.05"))
                    for n in ("4", "8")])
        # the sweep: each method and n at each test amplitude
        @test sort([(r[1], r[4], r[5]) for r in sweep]) ==
              sort([(m, a, n) for m in ("POD", "PSD") for a in ("0.015", "0.03")
                    for n in ("4", "8")])

        nfailed = 0
        for r in vcat(data, sweep)
            status = r[col("status")]
            @test status == "ok" || occursin(r"^failed at step \d+$", status)
            if status == "ok"
                @test parse(Float64, r[col("l2_error_max")]) ≥ 0
                # the fluctuation is a fraction of f, so the error relative to it is larger
                @test parse(Float64, r[col("max_l2_error_fluct")]) >
                      parse(Float64, r[col("l2_error_max")])
                @test parse(Float64, r[col("offline_seconds")]) > 0
            else
                nfailed += 1
                @test r[col("l2_error_max")] == r[col("max_l2_error_fluct")] == ""
            end
        end
        for r in data
            r[col("status")] == "ok" || continue
            errors = parse.(Float64, split(r[col("l2_errors")], ';'))
            @test maximum(errors) == parse(Float64, r[col("l2_error_max")])
            fluct = parse.(Float64, split(r[col("l2_error_fluct")], ';'))
            @test length(fluct) == length(errors)
            @test maximum(fluct) == parse(Float64, r[col("max_l2_error_fluct")])
            @test parse(Float64, r[col("online_seconds_per_step")]) > 0
            @test parse(Float64, r[col("online_ratio")]) > 0
            # a rate for the two linear cases only, where the fit finds its maxima
            case, rate_status = r[col("case")], r[col("rate_status")]
            if case == "nonlinear_landau"
                @test rate_status == "none"
                @test r[col("rate")] == r[col("rate_full_order")] == ""
            elseif case == "two_stream"
                @test rate_status == "ok"
            else
                @test rate_status == "ok" ||
                      startswith(rate_status, "fit failed: the Landau fit")
                @test parse(Float64, r[col("rate_full_order")]) < 0
            end
            @test (r[col("rate")] == "") == (rate_status != "ok")
        end
        @test occursin("$(length(rows)) rows, $nfailed failed", printed)

        # the online cost is measured in a fresh process
        cost = TOML.parsefile(joinpath(results, "linear_baselines_cost.toml"))
        @test cost["pid"] != getpid()
        @test cost["steps"] == 100
        @test !isempty(cost["uptime"])
        @test cost["full_order_seconds_per_step"] ==
              TOML.parsefile(joinpath(results, "full_order_cost.toml"))["seconds_per_step"]

        # every reduced run broken at step 2: each row records it, and the script goes on
        broken = joinpath(root, "broken")
        mkpath(broken)
        cp(joinpath(results, "full_order_cost.toml"), joinpath(broken, "full_order_cost.toml"))
        output = joinpath(root, "broken.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                run_linear_baselines(; runs_dir = runs, results_dir = broken, smoke = true,
                    perturb = (s, z) -> s == 2 && (z[1] = NaN))
            end
        end
        header, rows = read_csv(joinpath(broken, "linear_baselines.csv"))
        @test length(rows) == 20
        for r in rows
            @test r[col("status")] == "failed at step 2"
            @test r[col("offline_seconds")] != ""
            for name in ("l2_error_max", "l2_errors", "l2_error_fluct", "max_l2_error_fluct",
                "rate", "energy_drift",
                "online_seconds_per_step", "online_ratio")
                @test r[col(name)] == ""
            end
        end
        @test occursin("20 rows, 20 failed", read(output, String))
        @test isempty(get(TOML.parsefile(joinpath(broken, "linear_baselines_cost.toml")),
            "rows", []))

        # the grids of the cost file and of the run files are those of the mode
        e = thrown(() -> run_linear_baselines(; runs_dir = runs, results_dir = results))
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
            run_linear_baselines(; runs_dir = runs, results_dir = wrong)
        end)
        @test e isa ErrorException
        @test occursin(
            r"^linear_landau_alpha0\.01\.h5 holds a 32 × 64 run, not the 64 × 128 grid of this mode$",
            e.msg)
    end
end
