using HDF5: attrs, h5open
using LinearAlgebra: norm, svd, svdvals
using Printf: @sprintf
using VlasovReductionBenchmark
using Test

include("../helpers/thrown.jl")
include("../../scripts/reconstruction.jl")

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/reconstruction\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
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
        output = joinpath(root, "output.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                main(["--runs-dir", runs, "--results-dir", results, "--smoke"])
            end
        end
        printed = read(output, String)

        # the long runs, to t = 40 on the smoke grid
        for file in ("two_stream_long_alpha0.05.h5", "two_stream_beams_long_alpha0.05.h5",
            "nonlinear_landau_long_alpha0.5.h5")
            h5open(joinpath(runs, file)) do f
                @test (attrs(f)["tend"], attrs(f)["nx"], attrs(f)["nv"]) == (40.0, 32, 64)
            end
        end

        header, rows = read_csv(joinpath(results, "reconstruction.csv"))
        @test header == ["dataset", "split", "method", "n", "seed", "state_size", "status",
            "error", "max_snapshot_error", "snapshot_errors", "train_snapshots",
            "test_snapshots", "offline_seconds", "loss_first", "loss_last"]
        col(name) = findfirst(==(name), header)
        value(r, name) = parse(Float64, r[col(name)])
        @test all(length(r) == length(header) for r in rows)

        # five datasets, two splits, five methods, n ∈ {4, 8}, one seed of each network
        datasets = ("advection_half", "advection_full", "two_stream_long",
            "two_stream_beams_long", "nonlinear_landau_long")
        methods = ("POD" => "", "PSD" => "", "untied" => "42", "weight-tied" => "42",
            "rank-r" => "")
        @test length(rows) == 100
        @test sort([(r[1], r[2], r[3], r[4], r[5]) for r in rows]) ==
              sort([(d, s, m, n, seed) for d in datasets
                    for s in ("in-sample", "extrapolation") for (m, seed) in methods
                    for n in ("4", "8")])
        @test all(r -> r[col("status")] == "ok", rows)
        @test occursin("100 rows, 0 failed", printed)

        snapshots = Dict("advection_half" => 67, "advection_full" => 133,
            "two_stream_long" => 81, "two_stream_beams_long" => 81,
            "nonlinear_landau_long" => 81)
        for r in rows
            errors = parse.(Float64, split(r[col("snapshot_errors")], ';'))
            @test length(errors) == parse(Int, r[col("test_snapshots")])
            @test maximum(errors) == value(r, "max_snapshot_error")
            m = snapshots[r[col("dataset")]]
            train, test = parse(Int, r[col("train_snapshots")]), length(errors)
            if r[col("split")] == "in-sample"
                @test test == m
                @test train == (r[col("method")] == "rank-r" ? 0 : m)
            else
                @test test == m - (m + 1) ÷ 2
                @test train == (r[col("method")] == "rank-r" ? 0 : (m + 1) ÷ 2)
            end
            n = parse(Int, r[col("n")])
            @test parse(Int, r[col("state_size")]) ==
                  (r[col("method")] == "rank-r" ? n * (32 + 64) + n^2 : n)
            if r[col("method")] in ("untied", "weight-tied")
                @test value(r, "loss_last") < value(r, "loss_first")
            end
            # the rank-r reference builds nothing
            @test r[col("method")] == "rank-r" ? r[col("offline_seconds")] == "" :
                  value(r, "offline_seconds") > 0
        end

        # one row by value: POD of n = 4 in extrapolation on the two-stream run, from the SVD of
        # its 41 training fluctuations around the background of the first
        S = h5open(f -> read(f, "snapshots"), joinpath(runs, "two_stream_long_alpha0.05.h5"))
        f0 = vec(sum(S[:, :, 1]; dims = 1)) ./ 32
        G = reshape(S .- f0', 32 * 64, 81)
        U = svd(G[:, 1:41]).U[:, 1:4]
        test = G[:, 42:81]
        errors = [norm(g - U * (U' * g)) / norm(g) for g in eachcol(test)]
        @test argmax(errors) > 1
        r = only(filter(
            r -> (r[1], r[2], r[3], r[4]) ==
                 ("two_stream_long", "extrapolation", "POD", "4"),
            rows))
        @test value(r, "max_snapshot_error") ≈ maximum(errors) rtol=1e-10
        @test parse.(Float64, split(r[col("snapshot_errors")], ';')) ≈ errors rtol=1e-10
        @test value(r, "error") ≈ norm(test - U * (U' * test)) / norm(test) rtol=1e-10

        # the singular values and the mode counts
        header_sv, sv = read_csv(joinpath(results, "singular_values.csv"))
        @test header_sv ==
              ["dataset", "snapshots", "modes_99", "modes_9999", "singular_values"]
        @test [r[1] for r in sv] == collect(datasets)
        σ = svdvals(G)
        r = only(filter(r -> r[1] == "two_stream_long", sv))
        @test parse.(Float64, split(r[5], ';'))≈σ ./ σ[1] rtol=1e-10 atol=1e-14
        energy = cumsum(σ .^ 2) ./ sum(σ .^ 2)
        @test parse(Int, r[3]) == findfirst(≥(0.99), energy)
        @test parse(Int, r[4]) == findfirst(≥(0.9999), energy)
        @test parse(Int, r[2]) == 81

        # every number of the page, outside the column names in backticks, is a value of a CSV
        # file at the four significant digits of the page
        page = read(joinpath(results, "reconstruction.md"), String)
        values = Set{String}()
        for (_, rs) in (read_csv(joinpath(results, "reconstruction.csv")),
                read_csv(joinpath(results, "singular_values.csv"))), r in rs, field in r,
            x in split(field, ';')
            v = tryparse(Float64, x)
            v === nothing || push!(values, x, @sprintf("%.4g", v))
        end
        numbers = [m.match
                   for m in eachmatch(r"\d+(\.\d+)?(e[-+]?\d+)?",
            replace(page, r"`[^`]*`" => ""))]
        @test !isempty(numbers)
        @test all(in(values), numbers)
        for d in datasets, s in ("in-sample", "extrapolation")

            @test occursin("## $d, $s\n", page)
        end
        @test occursin("No row failed.", page)
    end
end
