using HDF5
using TOML
using Test

include("../../scripts/full_order.jl")

# The script's full mode, with every run, rate check and invariant assertion on 64 × 128.
@testset "Full run of the script" begin
    mktempdir() do root
        runs, results = joinpath(root, "runs"), joinpath(root, "results")
        main(["--runs-dir", runs, "--results-dir", results])

        @test length(readdir(runs)) == 7
        for file in readdir(runs)
            h5open(joinpath(runs, file)) do f
                @test (attrs(f)["nx"], attrs(f)["nv"]) == (64, 128)
                ts = read(f, "snapshot_times")
                @test size(read(f, "snapshots")) == (64, 128, length(ts))
            end
        end

        cost = TOML.parsefile(joinpath(results, "full_order_cost.toml"))
        @test (cost["nx"], cost["nv"], cost["steps"]) == (64, 128, 100)
        @test cost["seconds_per_step"] > 0
    end
end
