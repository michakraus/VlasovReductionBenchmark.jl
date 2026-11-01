using HDF5
using TOML
using Test
using VlasovReductionBenchmark
using VlasovReductionBenchmark: train_subsample

include("../../scripts/particles.jl")

const L = 4π

# The script's smoke mode, with every assertion: 512 particles over [0, 10], and 100 training
# steps of each frontend on 64 of them at n = 4 and 8.
@testset "Smoke run of the particle script" begin
    mktempdir() do root
        runs, results = joinpath(root, "runs"), joinpath(root, "results")
        output = joinpath(root, "output.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                main(["--runs-dir", runs, "--results-dir", results, "--smoke"])
            end
        end
        printed = read(output, String)

        @test readdir(runs) == ["particles_nonlinear_landau.h5"]
        positions, velocities, momentum = h5open(joinpath(runs, readdir(runs)[1])) do f
            @test attrs(f)["particles"] == 512
            @test attrs(f)["tend"] == 10.0
            @test read(f, "snapshot_times") ≈ 0:0.5:10 atol=1e-12
            @test length(read(f, "E1")) == 201
            @test maximum(abs, read(f, "energy_drift")) ≤ 1e-3
            read(f, "positions"), read(f, "velocities"), read(f, "momentum_drift")
        end
        @test size(positions) == (512, 21)
        @test all(xi -> 0 ≤ xi < L, positions)
        # the largest momentum drift, printed with `%.4e`
        printed_drift = match(r"largest momentum drift (\S+),", printed)
        @test printed_drift !== nothing
        @test parse(Float64, printed_drift.captures[1]) ≈ maximum(abs, momentum) rtol=1e-3

        pod = readlines(joinpath(results, "particle_pod_modes.csv"))
        @test pod[1] == "snapshots,particles,threshold,modes"
        Xs, Vs = sorted_snapshots(positions, velocities)
        @test pod[2:end] == [
            "labelled,512,0.99,$(pod_modes(vcat(positions, velocities), 0.99))",
            "labelled,512,0.9999,$(pod_modes(vcat(positions, velocities), 0.9999))",
            "sorted,512,0.99,$(pod_modes(vcat(Xs, Vs), 0.99))",
            "sorted,512,0.9999,$(pod_modes(vcat(Xs, Vs), 0.9999))"]

        lines = readlines(joinpath(results, "particle_frontends.csv"))
        @test lines[1] ==
              "method,n,seed,particles,steps,status,relative_error,seconds_per_step"
        rows = split.(lines[2:end], ',')
        @test [(r[1], r[2], r[3]) for r in rows] ==
              [("sort", "4", "42"), ("sort", "8", "42"), ("slot", "4", "42"),
            ("slot", "8", "42"), ("pairwise", "4", "42"), ("pairwise", "8", "42"),
            ("pod", "4", "none"), ("pod", "8", "none")]
        # every frontend, :pairwise included, trains on the 64 particles of the subsample
        @test all(r -> r[4] == "64", rows)
        @test all(r -> r[5] == "100", rows[1:6])
        @test all(r -> r[6] == "ok", rows)
        @test all(r -> 0 < parse(Float64, r[7]) < 2, rows)
        Z = train_subsample(positions, velocities, 64)
        Zs = vcat(sorted_snapshots(Z[1:64, :], Z[65:end, :])...)
        @test parse(Float64, rows[7][7]) ≈ pod_error(Zs, 4) rtol=1e-12
        @test parse(Float64, rows[8][7]) ≈ pod_error(Zs, 8) rtol=1e-12
        @test occursin("0 of 6 training runs failed", printed)

        cost = TOML.parsefile(joinpath(results, "particle_cost.toml"))
        @test (cost["particles"], cost["steps"], cost["dt"]) == (512, 100, 0.05)
        @test cost["seconds_per_step"] > 0
        @test !isempty(cost["uptime"])
        @test occursin("HDF5", cost["note"])
        # measured in a fresh process, not in this one
        @test cost["pid"] != getpid()
    end
end
