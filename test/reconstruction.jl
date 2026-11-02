using LinearAlgebra: Diagonal, norm, qr
using QuadGK: quadgk
using Random: Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: mode_count, rank_error, reconstruction_errors,
                                reconstruction_mode, reconstruction_row,
                                snapshot_singular_values,
                                write_reconstruction_page
using Test

include("helpers/thrown.jl")

const L = 4π

# The Maxwellian times the bump of width 1 at `x0`, with its two periodic images: the closed
# form at t = 0, for x - x0 in [-L/2, L/2).
bump(x, v, x0) = exp(-v^2 / 2) / sqrt(2π) * sum(exp(-(x - x0 - m * L)^2 / 2) for m in -1:1)

@testset "Advection snapshots" begin
    x, v = grid(linear_landau(), 32, 64)
    x0 = L / 2

    # at t = 0 the array is the closed form
    S0 = advection_snapshots(32, 64, [0.0]; σ = 1.0, x0)
    @test size(S0) == (32, 64, 1)
    expected = [bump(x[i], v[j], x0) for i in 1:32, j in 1:64]
    @test maximum(abs, S0[:, :, 1] - expected) ≤ 1e-15 * maximum(abs, expected)
    # the defaults are the bump of width 1 at the middle of the interval
    @test advection_snapshots(32, 64, [0.0, 3.0]) ==
          advection_snapshots(32, 64, [0.0, 3.0]; σ = 1.0, x0 = L / 2)

    # the pointwise function solves ∂f/∂t + v ∂f/∂x = 0 at five points off the grid
    f(x, v, t) = advection_profile(x, v, t; σ = 1.0, x0, L)
    rng = Xoshiro(42)
    h, fmax = 1e-4, 1 / sqrt(2π)
    for _ in 1:5
        xp, vp, tp = L * rand(rng), -2 + 4 * rand(rng), 40 * rand(rng)
        ∂t = (f(xp, vp, tp + h) - f(xp, vp, tp - h)) / 2h
        ∂x = (f(xp + h, vp, tp) - f(xp - h, vp, tp)) / 2h
        @test abs(∂t + vp * ∂x) ≤ 1e-6 * fmax
    end

    # the mass of every snapshot over a recurrence time of the smoke grid is that of the first
    times = advection_times(64, 0.5; fraction = 1.0)
    S = advection_snapshots(32, 64, times; σ = 1.0, x0)
    @test size(S) == (32, 64, 133)
    hx, hv = step(x), step(v)
    mass = [hx * hv * sum(view(S, :, :, s)) for s in axes(S, 3)]
    @test maximum(abs, mass .- mass[1]) ≤ 1e-12 * mass[1]
end

@testset "Long runs" begin
    M(u) = exp(-2u^2) * 2 / sqrt(2π)
    @test quadgk(two_stream_beams_profile, -6, 6; rtol = 1e-14)[1] ≈ 1 atol=1e-12
    for v in (0.3, 1.5, 2.7, 5.9)
        @test two_stream_beams_profile(-v) == two_stream_beams_profile(v)
    end
    @test two_stream_beams_profile(1.5) ≈ (M(0) + M(3)) / 2 rtol=1e-15

    for tend in (100.0, 40.0)
        cases = long_run_cases(tend)
        @test [c.name for c in cases] ==
              ["two_stream_long", "two_stream_beams_long", "nonlinear_landau_long"]
        @test [c.profile for c in cases] ==
              [two_stream_profile, two_stream_beams_profile, maxwellian]
        @test [(c.α, c.k, c.tend) for c in cases] ==
              [(0.05, 0.5, tend), (0.05, 0.5, tend), (0.5, 0.5, tend)]
    end
end

@testset "Advection times" begin
    TR = 2π / (0.5 * 12 / 127)
    half = advection_times(128, 0.5; fraction = 1 / 2)
    full = advection_times(128, 0.5; fraction = 1.0)
    @test length(half) == 134
    @test length(full) == 267
    @test last(full) ≈ TR rtol=1e-12
    @test last(half) ≈ TR / 2 rtol=1e-12
    @test first(half) == first(full) == 0.0
    @test maximum(abs, diff(full) .- TR / 266) ≤ 1e-12 * TR
end

@testset "Time split" begin
    split = time_split(collect(range(0.0, 40.0; length = 81)), 40.0)
    @test split.train == 1:41
    @test split.test == 42:81
    # the middle snapshot of an odd count trains
    full = advection_times(128, 0.5; fraction = 1.0)
    split = time_split(full, last(full))
    @test split.train == 1:134
    @test split.test == 135:267
end

@testset "Rank-r error" begin
    σ = [3.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    U = Matrix(qr(randn(Xoshiro(1), 8, 6)).Q)
    V = Matrix(qr(randn(Xoshiro(2), 6, 6)).Q)
    g = U * Diagonal(σ) * V'
    for r in 1:5
        @test abs(rank_error(g, r) - sqrt(sum(abs2, σ[(r + 1):end]) / sum(abs2, σ))) ≤ 1e-14
    end

    # the aggregate over two snapshots of singular values (3, 0.5) and (2, 1), at rank 1; the
    # larger error is at the second snapshot
    f0 = [1.0, 2.0, 3.0, 4.0]
    W = Matrix(qr(randn(Xoshiro(5), 4, 2)).Q)
    S = cat(f0' .+ U[:, 3:4] * Diagonal([3.0, 0.5]) * W',
        f0' .+ U[:, 1:2] * Diagonal([2.0, 1.0]) * W'; dims = 3)
    errors = reconstruction_errors("fixture", 1, S, f0)
    @test errors.error ≈ sqrt((0.25 + 1) / (9.25 + 5)) rtol=1e-14
    @test errors.snapshot_errors ≈ [sqrt(0.25 / 9.25), sqrt(1 / 5)] rtol=1e-14
    @test errors.max_snapshot_error == errors.snapshot_errors[2]
end

# Fourier modes in x of zero mean, times a profile in v, on the 8 × 4 fixture grid.
function fourier_fixture(amplitudes)
    x = range(0, 2π; length = 9)[1:8]
    modes = [cos.(x), sin.(x), cos.(2x), sin.(2x), cos.(3x), sin.(3x)]
    w = [1.0, 2.0, -1.0, 0.5]
    f0 = [3.0, 4.0, 5.0, 6.0]
    S = zeros(8, 4, length(amplitudes))
    for s in eachindex(amplitudes)
        S[:, :, s] .= f0' .+ amplitudes[s] .* modes[s] .* w'
    end
    return S
end

@testset "POD extrapolation error" begin
    # the three training snapshots (t ≤ 2.5) and the three test snapshots hold orthogonal modes,
    # on a background of 3–6; the test modes are larger, so a basis of all six snapshots holds them
    S = fourier_fixture([1.0, 0.9, 0.8, 5.0, 4.0, 3.0])
    times = collect(0.0:1.0:5.0)
    row = reconstruction_row("fixture", S, times, "extrapolation", "POD", 2)
    @test row["status"] == "ok"
    @test abs(parse(Float64, row["error"]) - 1) ≤ 1e-12
    @test abs(parse(Float64, row["max_snapshot_error"]) - 1) ≤ 1e-12
    @test all(e -> abs(parse(Float64, e) - 1) ≤ 1e-12, split(row["snapshot_errors"], ';'))
    @test (row["train_snapshots"], row["test_snapshots"]) == ("3", "3")
    @test (row["seed"], row["state_size"], row["loss_first"]) == ("", "2", "")

    # in sample, the basis of n = 2 holds the two largest modes
    row = reconstruction_row("fixture", S, times, "in-sample", "POD", 2)
    @test parse(Float64, row["error"]) ≈
          sqrt((9 + 1 + 0.81 + 0.64) / (25 + 16 + 9 + 1 + 0.81 + 0.64)) rtol=1e-12
    @test (row["train_snapshots"], row["test_snapshots"]) == ("6", "6")

    # the rank-r reference: one mode per snapshot, so rank 1 reproduces each
    row = reconstruction_row("fixture", S, times, "extrapolation", "rank-r", 1)
    @test parse(Float64, row["error"]) ≤ 1e-14
    @test row["state_size"] == string(1 * (8 + 4) + 1)
    @test (row["train_snapshots"], row["test_snapshots"], row["offline_seconds"]) ==
          ("0", "3", "")
end

@testset "Unknown split and method" begin
    S = fourier_fixture([1.0, 0.9, 0.8])
    e = thrown(() -> reconstruction_row("fixture", S, [0.0, 1.0, 2.0], "all", "POD", 1))
    @test e isa ArgumentError
    @test occursin(r"^the split is \"in-sample\" or \"extrapolation\", got \"all\"$", e.msg)
    e = thrown(() -> reconstruction_row(
        "fixture", S, [0.0, 1.0, 2.0], "in-sample", "DMD", 1))
    @test e isa ArgumentError
    @test occursin(r"^no reconstruction method \"DMD\"$", e.msg)
end

@testset "Snapshot without fluctuation" begin
    S = fourier_fixture([1.0, 0.0, 0.8])
    f0 = [3.0, 4.0, 5.0, 6.0]
    r = pod_reduction(S, f0, 1)
    e = thrown(() -> reconstruction_errors("fixture", r, S))
    @test e isa ErrorException
    @test occursin(
        r"^fixture: test snapshot 2 has no fluctuation: \S+ ≤ 1\.0e-14 times \S+$",
        e.msg)
    e = thrown(() -> reconstruction_errors("fixture", 1, S, f0))
    @test e isa ErrorException
    @test occursin(r"^fixture: test snapshot 2 has no fluctuation", e.msg)
    # a fluctuation of about 2e-15 of the state is below the floor too
    S = fourier_fixture([1.0, 1e-14, 0.8])
    e = thrown(() -> reconstruction_errors("fixture", 1, S, f0))
    @test e isa ErrorException
    @test occursin(r"^fixture: test snapshot 2 has no fluctuation", e.msg)
end

# A reduction whose aggregate error is not that of its per-snapshot errors.
struct InconsistentReduction
    f0::Vector{Float64}
end
function VlasovReductionBenchmark.relative_projection_error(::InconsistentReduction, S)
    size(S, 3) == 1 ? 0.5 : 0.25
end

@testset "Per-snapshot errors that miss the aggregate" begin
    S = fourier_fixture([1.0, 0.9, 0.8])
    e = thrown(() -> reconstruction_errors(
        "fixture", InconsistentReduction([
            3.0, 4.0, 5.0, 6.0]), S))
    @test e isa ErrorException
    @test occursin(
        r"^fixture: the per-snapshot errors give the aggregate 0\.5, not the 0\.25 of relative_projection_error$",
        e.msg)
    # an exact reconstruction: errors of order 1e-16 pass the absolute tolerance
    row = reconstruction_row(
        "fixture", fourier_fixture([1.0, 0.9, 0.8, 5.0, 4.0, 3.0]), 0:5, "in-sample", "POD", 6)
    @test parse(Float64, row["error"]) < 1e-14
end

@testset "Singular values and mode counts" begin
    σ = [10.0, 4.0, 1.0, 0.3, 0.1, 0.01]
    # six orthonormal fluctuations of zero mean in x, on a background
    A = randn(Xoshiro(3), 8, 4, 6)
    A .-= sum(A; dims = 1) ./ 8
    U = Matrix(qr(reshape(A, 32, 6)).Q)
    V = Matrix(qr(randn(Xoshiro(4), 6, 6)).Q)
    f0 = [3.0, 4.0, 5.0, 6.0]
    S = reshape(U * Diagonal(σ) * V', 8, 4, 6) .+ f0'
    s = snapshot_singular_values(S)
    @test length(s) == 6
    @test maximum(abs, s - σ ./ 10) ≤ 1e-14
    # Σσ² reaches 99 % at 2 modes (0.9906) and 99.99 % at 4 (0.99991)
    @test mode_count(s, 0.99) == 2
    @test mode_count(s, 0.9999) == 4
    @test mode_count(σ, 0.99) == 2
end

@testset "Reconstruction mode" begin
    @test reconstruction_mode(false) ==
          (grid = (64, 128), ns = (4, 8, 16, 32), steps = 2000,
        seeds = (untied = (42, 43, 44), weight_tied = (42,)), long_tend = 100.0,
        advection_fractions = (advection_half = 0.5, advection_full = 1.0))
    @test reconstruction_mode(true) ==
          (grid = (32, 64), ns = (4, 8), steps = 100,
        seeds = (untied = (42,), weight_tied = (42,)), long_tend = 40.0,
        advection_fractions = (advection_half = 0.5, advection_full = 1.0))
    full = reconstruction_mode(false)
    @test VlasovReductionBenchmark.method_seeds(full, "untied") == (42, 43, 44)
    @test VlasovReductionBenchmark.method_seeds(full, "weight-tied") == (42,)
    for method in ("POD", "PSD", "rank-r")
        @test VlasovReductionBenchmark.method_seeds(full, method) == (nothing,)
    end
end

@testset "A failed training" begin
    times = advection_times(64, 0.5; fraction = 1.0)
    S = advection_snapshots(32, 64, times; σ = 1.0, x0 = L / 2)
    row = reconstruction_row("advection_full", S, times, "extrapolation", "untied", 4;
        seed = 42, steps = 2, step_size = nextfloat(0.0))
    @test occursin(r"^failed: the loss did not fall \((\S+) → \1\)$", row["status"])
    @test row["error"] == row["max_snapshot_error"] == row["snapshot_errors"] == ""
    @test row["loss_first"] == row["loss_last"] != ""
    @test (row["seed"], row["train_snapshots"], row["test_snapshots"]) == ("42", "67", "66")
end

@testset "Page" begin
    mktempdir() do dir
        header = "dataset,split,method,n,seed,state_size,status,error,max_snapshot_error," *
                 "snapshot_errors,train_snapshots,test_snapshots,offline_seconds,loss_first," *
                 "loss_last"
        lines = [header,
            "d1,in-sample,POD,4,,4,ok,0.25,0.5,0.1;0.5,3,3,0.01,,",
            "d1,in-sample,untied,4,42,4,ok,0.375,0.5,0.1;0.5,3,3,1.5,0.75,0.5",
            "d1,in-sample,untied,4,43,4,ok,0.125,0.5,0.1;0.5,3,3,1.5,0.75,0.5",
            "d1,in-sample,untied,4,44,4,\"failed at step 7\",,,,3,3,1.5,0.75,Inf",
            "d1,in-sample,weight-tied,4,42,4,\"failed: the loss did not fall\",,,,3,3,2.5,0.75,0.75",
            "d1,in-sample,rank-r,4,,80,ok,0.0625,0.5,0.1;0.5,0,3,,,",
            "d1,in-sample,POD,8,,8,ok,0.5,0.5,0.1;0.5,3,3,0.01,,",
            "d2,in-sample,POD,4,,4,ok,0.75,0.75,0.75,3,3,0.01,,"]
        write(joinpath(dir, "reconstruction.csv"), join(lines, '\n') * '\n')
        write(joinpath(dir, "singular_values.csv"),
            "dataset,snapshots,modes_99,modes_9999,singular_values\nd1,3,2,3,1.0;0.5;0.25\n")
        page = read(write_reconstruction_page(dir), String)
        d1, d2 = split(page, "## d2, in-sample")
        # the median of the two completed seeds, and their range; the failed seed is listed
        @test occursin("| untied | 0.25 [0.125, 0.375] | — |", d1)
        @test occursin("| POD | 0.25 | 0.5 |", d1)
        @test occursin("| rank-r | 0.0625 | — |", d1)
        @test occursin("| rank-r state size | 80 | — |", d1)
        @test occursin("| POD | 0.75 |", d2)
        @test !occursin("rank-r", d2)
        # the mode counts of a dataset stand in its own section, and only there
        @test occursin("Mode counts: `modes_99` = 2, `modes_9999` = 3 of 3 snapshots.", d1)
        @test !occursin("Mode counts", d2)
        @test occursin("2 of 8 rows failed:", page)
        @test occursin("- d1, in-sample, untied, n = 4, seed 44: failed at step 7", page)
        @test !occursin("No row failed.", page)
        # a method with no completed row has no value, and a split with no row no table
        @test occursin("| weight-tied | — | — |", d1)
        @test !occursin("extrapolation", page)
    end
end
