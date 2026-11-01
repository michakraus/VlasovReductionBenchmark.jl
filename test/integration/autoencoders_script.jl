using TOML
using VlasovReductionBenchmark
using VlasovReductionBenchmark: read_run, reduced_rate, train_untied, train_weight_tied
using Test

include("../helpers/thrown.jl")
include("../../scripts/autoencoders.jl")

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/autoencoders\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
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

        header, rows = read_csv(joinpath(results, "autoencoders.csv"))
        col(name) = findfirst(==(name), header)
        @test header[1:6] == ["method", "kind", "case", "alpha", "n", "seed"]
        for name in ("status", "l2_error_max", "l2_errors", "l2_error_fluct",
            "max_l2_error_fluct", "rate", "rate_status", "rate_full_order", "energy_drift",
            "l1_drift", "l2_drift", "entropy_drift", "nonpositive_nodes", "projection_error",
            "offline_seconds", "online_seconds_per_step", "online_ratio", "loss_stage_I",
            "loss_final")
            @test name in header
        end
        @test all(length(r) == length(header) for r in rows)

        data = filter(r -> r[col("kind")] == "data", rows)
        sweep = filter(r -> r[col("kind")] == "sweep", rows)
        networks = ("untied", "weight-tied")
        # each network, the three cases at the amplitude of their full-order run, n ∈ {4, 8}
        @test sort([(r[1], r[3], r[4], r[5], r[6]) for r in data]) ==
              sort([(m, c, a, n, "42") for m in networks
                    for (c, a) in (("linear_landau", "0.01"), ("nonlinear_landau", "0.5"),
            ("two_stream", "0.05"))
                    for n in ("4", "8")])
        # the sweep: each network and n at each test amplitude
        @test sort([(r[1], r[4], r[5], r[6]) for r in sweep]) ==
              sort([(m, a, n, "42") for m in networks for a in ("0.015", "0.03")
                    for n in ("4", "8")])

        # every training and every reduced run of the smoke mode completes (measured)
        @test all(r -> r[col("status")] == "ok", rows)
        @test occursin("$(length(rows)) rows, 0 failed", printed)
        for r in rows
            # the training moved: its last loss is below its loss after Stage I
            @test parse(Float64, r[col("loss_final")]) <
                  parse(Float64, r[col("loss_stage_I")])
            @test parse(Float64, r[col("offline_seconds")]) > 0
            @test parse(Float64, r[col("max_l2_error_fluct")]) > 0
        end
        for r in data
            errors = parse.(Float64, split(r[col("l2_errors")], ';'))
            @test maximum(errors) == parse(Float64, r[col("l2_error_max")])
            fluct = parse.(Float64, split(r[col("l2_error_fluct")], ';'))
            @test maximum(fluct) == parse(Float64, r[col("max_l2_error_fluct")])
            @test parse(Float64, r[col("online_seconds_per_step")]) > 0
            @test parse(Float64, r[col("state_size")]) == parse(Float64, r[col("n")])
        end
        # each training is its own: the losses differ by network, case and n
        @test allunique([r[col("loss_stage_I")] for r in data])

        # the online cost is measured in a fresh process
        cost = TOML.parsefile(joinpath(results, "autoencoders_cost.toml"))
        @test cost["pid"] != getpid()
        @test cost["steps"] == 100
        @test !isempty(cost["uptime"])
        full_seconds = TOML.parsefile(joinpath(results, "full_order_cost.toml"))["seconds_per_step"]
        @test cost["full_order_seconds_per_step"] == full_seconds
        @test length(cost["rows"]) == length(data)
        key(r) = (r[col("case")] * "_alpha" * r[col("alpha")] * ".h5", r[col("method")],
            r[col("n")], r[col("seed")])
        for c in cost["rows"]
            r = only(filter(
                r -> key(r) == (c["file"], c["network"], string(c["n"]), string(c["seed"])),
                data))
            seconds = parse(Float64, r[col("online_seconds_per_step")])
            @test seconds == c["seconds_per_step"]
            @test parse(Float64, r[col("online_ratio")]) ≈ seconds / full_seconds rtol=8eps()
        end

        # the canonical flow of the energy against the Arakawa field, at the first snapshot
        landau = read_run(joinpath(runs, "linear_landau_alpha0.01.h5"))
        recorded = TOML.parsefile(joinpath(results, "autoencoders_hamiltonian.toml"))
        expected = canonical_flow_comparison(landau.case, landau.snapshots[:, :, 1])
        @test recorded["cosine"] == expected.cosine
        @test recorded["norm_ratio"] == expected.norm_ratio
        @test (recorded["nx"], recorded["nv"], recorded["t"]) == (32, 64, 0.0)

        # rows against trainings and reduced runs of their own, by network and n
        row(kind, method, alpha, n) = only(filter(
            r -> (r[col("kind")], r[col("method")], r[col("alpha")], r[col("n")]) ==
                 (kind, method, alpha, n), rows))
        value(r, name) = parse(Float64, r[col(name)])
        two_stream = read_run(joinpath(runs, "two_stream_alpha0.05.h5"))
        for (method, build, run, alpha, n) in (
            ("weight-tied", train_weight_tied, landau, "0.01", 8),
            ("untied", train_untied, two_stream, "0.05", 4))
            S = run.snapshots
            f0 = background(S[:, :, 1])
            trained = build(S, f0, n; seed = 42, steps = 100, label = method)
            r = row("data", method, alpha, string(n))
            @test value(r, "loss_stage_I") ≈ trained.training.first rtol=1e-8
            @test value(r, "loss_final") ≈ trained.training.last rtol=1e-8
            @test value(r, "projection_error") ≈
                  VlasovReductionBenchmark.relative_projection_error(trained.reduction, S) rtol=1e-8
            reduced = run_reduced(trained.reduction, run.case, 32, 64, S[:, :, 1])
            errors = relative_l2_errors(reduced.snapshots, S)
            # the maximum of this row's error is not at its first snapshot
            @test argmax(errors) > 1
            @test value(r, "l2_error_max") ≈ maximum(errors) rtol=1e-6
            @test value(r, "energy_drift")≈max_drift(reduced.energy) rtol=1e-6 atol=1e-12
        end
        # a sweep row: the untied network at n = 4 on the three training runs, at 0.03
        train = [read_run(joinpath(runs, "linear_landau_alpha$α.h5")).snapshots
                 for α in (0.005, 0.01, 0.02)]
        f0 = background(train[1][:, :, 1])
        trained = train_untied(cat(train...; dims = 3), f0, 4; seed = 42, steps = 100,
            label = "sweep")
        test = read_run(joinpath(runs, "linear_landau_alpha0.03.h5"))
        reduced = run_reduced(trained.reduction, test.case, 32, 64, test.snapshots[:, :, 1])
        errors = relative_l2_errors(reduced.snapshots, test.snapshots)
        @test argmax(errors) > 1
        r = row("sweep", "untied", "0.03", "4")
        @test value(r, "l2_error_max") ≈ maximum(errors) rtol=1e-6
        @test value(r, "loss_final") ≈ trained.training.last rtol=1e-8

        # every reduced run broken at step 2: each row records it, and the script goes on
        broken = joinpath(root, "broken")
        mkpath(broken)
        cp(joinpath(results, "full_order_cost.toml"), joinpath(broken, "full_order_cost.toml"))
        output = joinpath(root, "broken.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                run_autoencoders(; runs_dir = runs, results_dir = broken, smoke = true,
                    perturb = (s, z) -> s == 2 && (z[1] = NaN))
            end
        end
        header, rows = read_csv(joinpath(broken, "autoencoders.csv"))
        @test length(rows) == 20
        for r in rows
            @test r[col("status")] == "failed at step 2"
            @test r[col("loss_final")] != ""
            for name in ("l2_error_max", "l2_errors", "max_l2_error_fluct", "energy_drift",
                "online_seconds_per_step", "online_ratio")
                @test r[col(name)] == ""
            end
        end
        @test occursin("20 rows, 20 failed", read(output, String))
        @test isempty(get(
            TOML.parsefile(joinpath(broken, "autoencoders_cost.toml")), "rows",
            []))

        # a training whose loss does not move stops the script with its labelled message
        e = thrown(() -> redirect_stdout(devnull) do
            run_autoencoders(; runs_dir = runs, results_dir = broken, smoke = true,
                step_size = nextfloat(0.0))
        end)
        @test e isa ErrorException
        @test occursin(
            r"^untied, linear_landau_alpha0\.01\.h5, n = 4, seed 42: the training loss went from (\S+) to \1, not below$",
            e.msg)

        # a training whose loss becomes non-finite is a row with its status, and the script
        # goes on
        output = joinpath(root, "overflow.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                run_autoencoders(; runs_dir = runs, results_dir = broken, smoke = true,
                    step_size = floatmax(Float64) / 2)
            end
        end
        header, rows = read_csv(joinpath(broken, "autoencoders.csv"))
        @test length(rows) == 20
        for r in rows
            @test occursin(r"^failed at step [1-9]\d*$", r[col("status")])
            @test !isfinite(value(r, "loss_final"))
            # a failed training integrates nothing and measures no projection error
            @test r[col("l2_error_max")] == r[col("online_seconds_per_step")] ==
                  r[col("projection_error")] == ""
        end
        @test occursin("20 rows, 20 failed", read(output, String))

        # the grids of the cost file and of the run files are those of the mode
        e = thrown(() -> run_autoencoders(; runs_dir = runs, results_dir = results))
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
            run_autoencoders(; runs_dir = runs, results_dir = wrong)
        end)
        @test e isa ErrorException
        @test occursin(
            r"^linear_landau_alpha0\.01\.h5 holds a 32 × 64 run, not the 64 × 128 grid of this mode$",
            e.msg)
    end
end
