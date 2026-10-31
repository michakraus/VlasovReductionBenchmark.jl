using HDF5
using LinearAlgebra: dot
using PoissonSolvers: FFTWBasis, Potential, solve
using TOML
using VlasovReductionBenchmark
using Test

include("helpers/thrown.jl")

# A short linear Landau run, 20 steps of 0.05, for the tests of the run's assertions.
short_case() = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 1.0)

@testset "Initial conditions and models" begin
    case = linear_landau()
    @test (case.name, case.α, case.k, case.tend) == ("linear_landau", 0.01, 0.5, 40.0)
    @test (nonlinear_landau().α, nonlinear_landau().tend) == (0.5, 60.0)
    @test (two_stream().name, two_stream().α, two_stream().tend) ==
          ("two_stream", 0.05, 60.0)

    x, v = grid(case, 32, 64)
    @test length(x) == 32 && length(v) == 64
    @test x[1] == 0
    @test step(x) ≈ 4π / 32 rtol=1e-14
    @test (first(v), last(v)) == (-6.0, 6.0)

    for c in (linear_landau(0.02), nonlinear_landau(), two_stream())
        f = initial_condition(c, x, v)
        @test size(f) == (32, 64)
        # the closed forms of the decisions, at every node
        g = c.name == "two_stream" ? v .^ 2 .* exp.(-v .^ 2 ./ 2) ./ sqrt(2π) :
            exp.(-v .^ 2 ./ 2) ./ sqrt(2π)
        @test f ≈ (1 .+ c.α .* cos.(0.5 .* x)) .* g' rtol=1e-14
    end

    model = vlasov_model(two_stream(), 32, 64)
    @test size(model.distribution.values) == (32, 64)
    @test model.distribution.values == initial_condition(two_stream(), x, v)
end

@testset "Rate fits" begin
    t = collect(0:0.05:40)
    γ, ω = -0.15, 1.4
    y = log.(abs.(exp.(γ .* t) .* cos.(ω .* t)))
    fit = landau_rate_fit(t, y, (0.0, 40.0))
    @test fit.γ ≈ γ rtol=1e-2
    @test fit.ω ≈ ω rtol=1e-2
    @test fit.npeaks ≥ 4

    e = thrown(() -> landau_rate_fit(t, -t, (0.0, 40.0)))
    @test e isa ErrorException
    @test occursin(
        r"^the Landau fit found 0 maxima of log\|E₁\| in \[0\.0, 40\.0\]; it needs at least four$",
        e.msg)

    # |cos 1.4t| peaks at t = 2.24, 4.49, 6.73, 8.98, …: four maxima in [0, 9.5] are enough,
    # three in [0, 7.5] are not
    @test landau_rate_fit(t, y, (0.0, 9.5)).npeaks == 4
    e = thrown(() -> landau_rate_fit(t, y, (0.0, 7.5)))
    @test e isa ErrorException
    @test occursin(
        r"^the Landau fit found 3 maxima of log\|E₁\| in \[0\.0, 7\.5\]; it needs at least four$",
        e.msg)
    # a plateau of two equal samples is one maximum
    tp = collect(0.1:0.1:2.4)
    @test landau_rate_fit(tp, repeat([0.0, 1.0, 1.0, 0.0], 6), (0.0, 2.4)).npeaks == 6

    @test growth_rate_fit(t, 0.26 .* t .- 3, (5.0, 25.0)) ≈ 0.26 rtol=1e-12
    # samples outside the window do not enter the fit
    yb = ifelse.(5 .≤ t .≤ 25, 0.26 .* t, 0.0)
    @test growth_rate_fit(t, yb, (5.0, 25.0)) ≈ 0.26 rtol=1e-12

    @test check_rate("γ", -0.1550, -0.1534, 0.02) === nothing
    e = thrown(() -> check_rate("γ", -0.1700, -0.1500, 0.02))
    @test e isa ErrorException
    @test occursin(
        r"^rate check γ: fitted -0\.17000, oracle -0\.15000, relative error 0\.1333, above 0\.02$",
        e.msg)
end

@testset "Diagnostics" begin
    # f = (1 + cos kx) with k = 1/2 on 4 x-nodes of [0, 4π), times 1 at v = 0 and 2 at v = 1,
    # with G = I: ρ = φ = 3 hv (1 + cos kx)
    k = 0.5
    x, v = range(0, 4π; length = 5)[1:4], [0.0, 1.0]
    hx, hv = step(x), 1.0
    f = (1 .+ cos.(k .* x)) .* [1.0 2.0]
    d = VlasovReductionBenchmark.diagnostics(
        vec(f), [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],
        k, x, v, hx, hv)
    # f = [2 4; 1 2; 0 0; 1 2] and ρ = [6, 3, 0, 3], with hx = π
    @test d.mass ≈ 12hx rtol=1e-14
    @test d.field_energy ≈ 27hx rtol=1e-14
    @test d.energy ≈ 4hx + 27hx rtol=1e-14
    # Σ φ_i e^{-ikx_i} = 6 - 3i + 0 + 3i = 6, so |E₁| = k (2/4) 6 = 1.5
    @test d.E1 ≈ 1.5 rtol=1e-14
    @test d.l2 ≈ sqrt(30hx) rtol=1e-14
    # f vanishes at x = π, so two nodes are skipped by the entropy and counted
    @test d.nonpositive == 2
    @test d.entropy ≈ -14hx * log(2) rtol=1e-14
end

@testset "Settings of the two modes" begin
    # the decided grids, fit windows and tolerances; this file runs only the smoke mode
    @test VlasovReductionBenchmark.FULL_MODE ==
          (grid = (64, 128), landau_window = (0.0, 40.0), landau_rtol = 0.02,
        two_stream_rtol = 0.05)
    @test VlasovReductionBenchmark.SMOKE_MODE ==
          (grid = (32, 64), landau_window = (0.0, 25.0), landau_rtol = 0.10,
        two_stream_rtol = 0.10)
    @test VlasovReductionBenchmark.TRAIN_AMPLITUDES == (0.005, 0.01, 0.02)
    @test VlasovReductionBenchmark.TEST_AMPLITUDES == (0.015, 0.03)
    @test (VlasovReductionBenchmark.TIMESTEP, VlasovReductionBenchmark.SNAPSHOT_INTERVAL) ==
          (0.05, 0.5)
    @test (VlasovReductionBenchmark.MASS_TOLERANCE,
        VlasovReductionBenchmark.ENERGY_TOLERANCE) ==
          (1e-12, 1e-4)
end

@testset "Invariant assertions of a run" begin
    # an unperturbed short run keeps both invariants
    run = run_case(short_case(), 32, 64)
    @test length(run.times) == 21
    @test run.times[end] ≈ 1.0 rtol=1e-14
    @test maximum(abs, run.mass_drift) ≤ 1e-12
    @test maximum(abs, run.energy_drift) ≤ 1e-4

    # one node raised by 1e-6 after the first step moves the mass by about 6e-9 relative
    e = thrown(() -> run_case(short_case(), 32, 64;
        perturb = (s, q) -> s == 1 && (q[1] += 1e-6)))
    @test e isa ErrorException
    @test occursin(
        r"^linear_landau, α = 0\.01: the mass drifted by \S+ relative at step 1, above 1\.0e-12$",
        e.msg)

    # mass moved from the node at v = -0.095 to the node at v = 5.81 keeps the mass and raises
    # the energy by about 2e-3 relative
    e = thrown(() -> run_case(short_case(), 32, 64;
        perturb = (s, q) -> s == 1 && (q[32 * 31 + 1] -= 1e-2; q[32 * 62 + 1] += 1e-2)))
    @test e isa ErrorException
    @test occursin(
        r"^linear_landau, α = 0\.01: the energy drifted by \S+ relative at step 1, above 0\.0001$",
        e.msg)

    # a non-finite value stops the run, and no file is written for it
    mktempdir() do d
        e = thrown(() -> run_case(short_case(), 32, 64; runs_dir = d,
            perturb = (s, q) -> s == 2 && (q[5] = NaN)))
        @test e isa ErrorException
        @test occursin(
            r"^linear_landau, α = 0\.01: the state holds a non-finite value after step 2$",
            e.msg)
        @test isempty(readdir(d))
    end
end

@testset "Run file" begin
    mktempdir() do d
        path = joinpath(d, "linear_landau_alpha0.01.h5")
        # a file already there is overwritten
        write(path, "not a run")
        run = run_case(short_case(), 32, 64; runs_dir = d, label = "train")
        @test readdir(d) == ["linear_landau_alpha0.01.h5"]
        h5open(path) do f
            @test size(read(f, "snapshots")) == (32, 64, 3)
            @test read(f, "snapshot_times") ≈ [0.0, 0.5, 1.0] atol=1e-12
            @test read(f, "snapshots")[:, :, end] == reshape(run.state, 32, 64)
            @test length(read(f, "E1")) == length(read(f, "field_energy")) == 21
            @test read(f, "times") ≈ 0:0.05:1 atol=1e-12
            @test length(read(f, "l2_drift")) == length(read(f, "entropy_drift")) == 21
            @test read(f, "nonpositive_nodes") isa Vector{Int}
            a = attrs(f)
            @test (a["case"], a["alpha"], a["k"], a["label"]) ==
                  ("linear_landau", 0.01, 0.5, "train")
            @test (a["nx"], a["nv"], a["dt"]) == (32, 64, 0.05)
        end
    end
end

include("../scripts/full_order.jl")

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/full_order\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
    e = thrown(() -> parse_options(["--runs-dir", "a"]))
    @test e isa ArgumentError
    @test occursin(usage, e.msg)
    e = thrown(() -> parse_options(["--runs-dir", "a", "--results-dir", "b", "--x"]))
    @test e isa ArgumentError
    @test occursin(usage, e.msg)
    # an option without its value
    e = thrown(() -> parse_options(["--runs-dir", "a", "--results-dir"]))
    @test e isa ArgumentError
    @test occursin(usage, e.msg)
end

@testset "Smoke run of the script" begin
    mktempdir() do root
        runs, results = joinpath(root, "runs"), joinpath(root, "results")
        output = joinpath(root, "output.txt")
        # the measuring process starts with 3 threads from the environment; this process keeps
        # the count it started with, which is not 3 where the suite runs here
        withenv("JULIA_NUM_THREADS" => "3") do
            open(output, "w") do io
                redirect_stdout(io) do
                    main(["--runs-dir", runs, "--results-dir", results, "--smoke"])
                end
            end
        end
        printed = read(output, String)

        sweep = ("linear_landau_alpha0.005.h5" => "train",
            "linear_landau_alpha0.01.h5" => "train",
            "linear_landau_alpha0.02.h5" => "train",
            "linear_landau_alpha0.015.h5" => "test",
            "linear_landau_alpha0.03.h5" => "test",
            "nonlinear_landau_alpha0.5.h5" => "none",
            "two_stream_alpha0.05.h5" => "none")
        @test sort(readdir(runs)) == sort(collect(first.(sweep)))
        for (file, label) in sweep
            h5open(joinpath(runs, file)) do f
                @test attrs(f)["label"] == label
                @test (attrs(f)["nx"], attrs(f)["nv"]) == (32, 64)
                ts = read(f, "snapshot_times")
                # one snapshot every 0.5 time units, from t = 0
                @test ts ≈ 0.5 .* (0:(length(ts) - 1)) atol=1e-12
                @test ts[end] ≈ attrs(f)["tend"] atol=1e-12
                @test size(read(f, "snapshots")) == (32, 64, length(ts))
                @test eltype(read(f, "snapshots")) == Float64

                # The series of the file at the snapshot steps, recomputed from the snapshots
                # on the grid of the decisions, with φ from PoissonSolvers' own solve.
                k, nx, nv = attrs(f)["k"], 32, 64
                L = 2π / k
                x, hx, hv = (0:(nx - 1)) .* (L / nx), L / nx, 12 / (nv - 1)
                solver = Potential(FFTWBasis((0.0, L), nx)).solver
                S, steps = read(f, "snapshots"), round.(Int, ts ./ 0.05) .+ 1
                l2(F) = sqrt(hx * hv * sum(abs2, F))
                entropy(F) = -hx * hv * sum(z * log(z) for z in F if z > 0; init = 0.0)
                ρ = [hv .* vec(sum(S[:, :, n]; dims = 2)) for n in eachindex(ts)]
                φ = [solve(solver, ρn) for ρn in ρ]
                # The two Poisson solves differ by up to 4e-7 relative in the field energy and
                # 2e-12 in |E₁| (measured), where the field is weakest.
                @test read(f, "field_energy")[steps] ≈
                      [hx / 2 * dot(ρn, φn) for (ρn, φn) in zip(ρ, φ)] rtol=1e-5
                @test read(f, "E1")[steps] ≈
                      [k * (2 / nx) * abs(sum(φn .* exp.(-im * k .* x))) for φn in φ] rtol=1e-9
                @test read(f, "l2_drift")[steps] ≈
                      [(l2(S[:, :, n]) - l2(S[:, :, 1])) / l2(S[:, :, 1])
                       for n in eachindex(ts)] atol=1e-13
                @test read(f, "entropy_drift")[steps] ≈
                      [(entropy(S[:, :, n]) - entropy(S[:, :, 1])) /
                       abs(entropy(S[:, :, 1]))
                       for n in eachindex(ts)] atol=1e-13
                @test read(f, "nonpositive_nodes")[steps] ==
                      [count(≤(0), S[:, :, n]) for n in eachindex(ts)]
            end
        end

        # The rate checks fit the run at α = 0.01 over the smoke window [0, 25], and a two-stream
        # run at α = 1e-3 over [5, 25]; the script prints each fit to five decimals.
        fitted(name) = parse(Float64,
            match(Regex("rate check $name: fitted (\\S+),"), printed).captures[1])
        landau = h5open(joinpath(runs, "linear_landau_alpha0.01.h5")) do f
            landau_rate_fit(read(f, "times"), log.(read(f, "E1")), (0.0, 25.0))
        end
        @test abs(fitted("linear Landau γ") - landau.γ) ≤ 1e-5
        @test abs(fitted("linear Landau ω") - landau.ω) ≤ 1e-5
        check = run_case(VlasovCase("two_stream", two_stream_profile, 1e-3, 0.5, 25.0), 32, 64)
        @test abs(fitted("two-stream growth rate") -
                  growth_rate_fit(check.times, log.(check.E1), (5.0, 25.0))) ≤ 1e-5

        cost = TOML.parsefile(joinpath(results, "full_order_cost.toml"))
        @test cost["steps"] == 100
        @test (cost["nx"], cost["nv"]) == (32, 64)
        @test cost["seconds_per_step"] > 0
        @test !isempty(cost["uptime"])
        # measured in a fresh process, not in this one
        @test cost["pid"] != getpid()
        @test cost["threads"] == 3
    end
end
