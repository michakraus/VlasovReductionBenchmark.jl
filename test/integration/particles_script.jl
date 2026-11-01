using HDF5
using TOML
using Test

include("../../scripts/particles.jl")

# The script's full mode: the PIC run of 4096 particles over [0, 40], its energy assertion, the
# POD mode counts, and the training of the three frontends on 256 particles, 2000 steps, three
# seeds and four latent dimensions.
@testset "Full run of the particle script" begin
    mktempdir() do root
        runs, results = joinpath(root, "runs"), joinpath(root, "results")
        main(["--runs-dir", runs, "--results-dir", results])

        @test readdir(runs) == ["particles_nonlinear_landau.h5"]
        h5open(joinpath(runs, "particles_nonlinear_landau.h5")) do f
            @test attrs(f)["particles"] == 4096
            @test size(read(f, "positions")) == (4096, 81)
            @test length(read(f, "E1")) == 801
            @test maximum(abs, read(f, "energy_drift")) ≤ 1e-3
        end

        pod = readlines(joinpath(results, "particle_pod_modes.csv"))
        @test length(pod) == 5
        @test all(line -> split(line, ',')[2] == "4096", pod[2:end])

        rows = split.(readlines(joinpath(results, "particle_frontends.csv"))[2:end], ',')
        @test length(rows) == 3 * 4 * 3 + 4
        @test all(r -> r[4] == "256", rows)
        @test count(r -> r[1] == "pairwise", rows) == 12
        @test all(r -> r[5] == "2000", filter(r -> r[1] != "pod", rows))

        cost = TOML.parsefile(joinpath(results, "particle_cost.toml"))
        @test (cost["particles"], cost["steps"]) == (4096, 100)
        @test cost["seconds_per_step"] > 0
    end
end
