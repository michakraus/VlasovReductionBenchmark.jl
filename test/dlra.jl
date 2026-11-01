using LinearAlgebra: I, norm, svdvals
using Random: Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: DLRA_STRIDE, DLRA_TIMESTEP, centred_derivative,
                                dlra_discretisation, dlra_mode, factored_state,
                                spectral_derivative, weighted_orthonormal
using Test

include("helpers/thrown.jl")

const CASES = (linear_landau(), nonlinear_landau(), two_stream())

@testset "Derivatives" begin
    # the spectral derivative is exact on each Fourier mode below the Nyquist mode
    nx, k = 32, 0.5
    x = 2π / k .* (0:(nx - 1)) ./ nx
    D = spectral_derivative(nx, k)
    for m in (1, 3, 15)
        @test D * sin.(m * k .* x) ≈ m * k .* cos.(m * k .* x) atol=1e-12
        @test D * cos.(m * k .* x) ≈ -m * k .* sin.(m * k .* x) atol=1e-12
    end
    @test norm(D * cos.(16k .* x)) < 1e-12
    @test norm(D * ones(nx)) < 1e-13

    # the centred difference, periodic over the nodes: (sin θ(j+1) - sin θ(j-1)) / 2h
    nv, hv = 64, 12 / 63
    θ = 2π * 3 / nv
    @test centred_derivative(nv, hv) * sin.(θ .* (1:nv)) ≈
          sin(θ) / hv .* cos.(θ .* (1:nv)) atol=1e-12
    Dv = centred_derivative(nv, hv)
    @test Dv[1, nv] == -1 / (2hv)
    @test Dv[nv, 1] == 1 / (2hv)
end

@testset "RK4" begin
    # one step of ẏ = λy multiplies by the RK4 polynomial of z = λh
    λ, h = [-0.3 1.1; -1.1 -0.3], 0.2
    Z = λ * h
    @test VlasovReductionBenchmark.rk4(y -> λ * y, [1.0, 2.0], h) ≈
          (I + Z + Z^2 / 2 + Z^3 / 6 + Z^4 / 24) * [1.0, 2.0] rtol=1e-14
end

@testset "Constants and modes" begin
    @test DLRA_TIMESTEP == 0.025
    # snapshots every 0.5, as in the full-order runs
    @test DLRA_STRIDE * DLRA_TIMESTEP == 0.5
    full, smoke = dlra_mode(false), dlra_mode(true)
    @test (full.grid, full.ranks, full.landau_window, full.landau_rtol) ==
          ((64, 128), (4, 8, 16, 32), (0.0, 40.0), 0.02)
    @test (smoke.grid, smoke.ranks, smoke.landau_window, smoke.landau_rtol) ==
          ((32, 64), (4, 8), (0.0, 25.0), 0.10)
    # the state size r (Nx + Nv) + r², in closed form
    @test dlra_state_size(4, 64, 128) == 784
    @test dlra_state_size(16, 32, 64) == 1792
end

@testset "Initial factorisation" begin
    d = dlra_discretisation(linear_landau(), 32, 64)
    @test d.f0 ≈ maxwellian.(d.v) rtol=1e-14
    for r in (1, 4, 8)
        X, S, V = initial_factors(d, r)
        @test size(X) == (32, r)
        @test size(S) == (r, r)
        @test size(V) == (64, r)
        @test norm(d.hx .* (X' * X) - I) < 1e-13
        @test norm(d.hv .* (V' * V) - I) < 1e-13
        @test norm(factored_state(d, X, S, V) - d.f_init) < 1e-14 * norm(d.f_init)
        # the fluctuation α cos(kx) g(v) has rank one: one singular value, the others zero
        σ = svdvals(S)
        @test σ[1] ≈
              0.01 * sqrt(d.hx * sum(abs2, cos.(0.5 .* d.x)) *
                   d.hv * sum(abs2, maxwellian.(d.v))) rtol=1e-12
        @test all(<(1e-15), σ[2:end])
    end
    # the completed columns come from Xoshiro(42): the same on every call, and, in every case,
    # whose fluctuation has rank one, the weighted orthonormal completion of the first draws of
    # Xoshiro(42), X's before V's
    @test initial_factors(d, 8) == initial_factors(d, 8)
    for case in (linear_landau(0.005), CASES...), (nx, nv) in ((32, 64), (64, 128))

        d = dlra_discretisation(case, nx, nv)
        X, _, V = initial_factors(d, 4)
        rng = Xoshiro(42)
        for (B, h) in ((X, d.hx), (V, d.hv))
            R = randn(rng, size(B, 1), 3)
            R .-= B[:, 1:1] * (h .* (B[:, 1:1]' * R))
            Q = weighted_orthonormal(R, h)
            @test abs.(h .* (Q' * B[:, 2:4])) ≈ I(3) atol=1e-12
        end
    end
end

@testset "A step keeps the bases orthonormal in the weighted inner product" begin
    for case in CASES
        d = dlra_discretisation(case, 32, 64)
        X, S, V = initial_factors(d, 8)
        f = factored_state(d, X, S, V)
        for _ in 1:3
            X, S, V = bug_step(d, X, S, V, DLRA_TIMESTEP)
        end
        @test size(S) == (8, 8)
        @test norm(factored_state(d, X, S, V) - f) > 1e-6 * norm(f)
        @test norm(d.hx .* (X' * X) - I) < 1e-12
        @test norm(d.hv .* (V' * V) - I) < 1e-12
    end
end

@testset "Check (a): full rank in x agrees with the full-rank solver" begin
    for case in CASES
        @test full_rank_agreement(case, 32, 64, 32, 20) ≤ 1e-10
    end
    # the step 0.025 is stable on the full grid
    for case in (linear_landau(), two_stream())
        err = full_rank_agreement(case, 64, 128, 64, 40)
        @test isfinite(err)
        @test err ≤ 1e-10
    end
    # on 128 × 128 it is not (|v| k_max dt = 4.7 > 2√2): over 40 steps the difference is NaN,
    # the check of 20 steps throws, the full-rank solver throws, and a step returns its
    # non-finite S untruncated
    @test isnan(full_rank_agreement(linear_landau(), 128, 128, 128, 40))
    e = thrown(() -> redirect_stdout(devnull) do
        VlasovReductionBenchmark.check_full_rank_agreement(linear_landau(), 128, 128)
    end)
    @test e isa ErrorException
    @test occursin(
        r"^full-rank check linear_landau on 128 × 128: the DLRA run at r = 128 differs from the full-rank solver by \S+ relative over 20 steps, above 1\.0e-10$",
        e.msg)
    short = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 1.0)
    e = thrown(() -> run_full_rank(short, 128, 128))
    @test e isa ErrorException
    @test occursin(
        r"^the full-rank solver of linear_landau, α = 0\.01 holds a non-finite value after step \d+$",
        e.msg)
    d = dlra_discretisation(linear_landau(), 32, 64)
    X, S, V = initial_factors(d, 4)
    X, S, V = bug_step(d, X, fill(NaN, 4, 4), V, DLRA_TIMESTEP)
    @test size(S) == (8, 8)
    @test !any(isfinite, S)
end

@testset "Checks (b) and (c): the damping rate of linear Landau damping on 64 × 128" begin
    oracle = imag(landau_root(0.5))
    full = run_full_rank(linear_landau(), 64, 128)
    @test length(full.times) == 1601
    @test size(full.snapshots, 3) == 81
    γ = landau_rate_fit(full.times, log.(full.E1), (0.0, 40.0)).γ
    @test abs(γ - oracle) ≤ 0.02 * abs(oracle)

    run = run_dlra(linear_landau(), 64, 128, 16)
    @test run.status == "ok"
    γ = landau_rate_fit(run.times, log.(run.E1), (0.0, 40.0)).γ
    @test abs(γ - oracle) ≤ 0.02 * abs(oracle)

    # the checks of the smoke mode pass; a DLRA run that stops fails them
    @test redirect_stdout(devnull) do
        VlasovReductionBenchmark.check_dlra_rates(dlra_mode(true))
    end === nothing
    e = thrown(() -> redirect_stdout(devnull) do
        VlasovReductionBenchmark.check_dlra_rates(dlra_mode(true);
            perturb = (s, S) -> s == 5 && (S[1] = NaN))
    end)
    @test e isa ErrorException
    @test occursin(
        r"^the DLRA run of rank 16 of linear Landau damping on 32 × 64 stopped: failed at step 5$",
        e.msg)
end

@testset "A DLRA run" begin
    case = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 2.0)
    run = run_dlra(case, 32, 64, 4)
    @test run.status == "ok"
    @test run.failed_step === nothing
    @test run.times ≈ 0:0.025:2.0
    @test length(run.mass) == length(run.E1) == length(run.nonpositive) == 81
    # the snapshots every 0.5, the first the initial condition
    @test size(run.snapshots) == (32, 64, 5)
    d = dlra_discretisation(case, 32, 64)
    @test run.snapshots[:, :, 1] ≈ d.f_init rtol=1e-14
    @test run.snapshots[:, :, end] ≈ factored_state(d, run.factors...) rtol=1e-14
    @test run.offline_seconds > 0
    # the second snapshot is the state at t = 0.5, the last of a run to t = 0.5
    half = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 0.5)
    @test run.snapshots[:, :, 2] == run_dlra(half, 32, 64, 4).snapshots[:, :, 2]
    @test run.snapshots[:, :, 2] != run.snapshots[:, :, 1]
    full = run_full_rank(case, 32, 64)
    @test full.snapshots[:, :, 1] == d.f_init
    @test full.snapshots[:, :, 2] == run_full_rank(half, 32, 64).snapshots[:, :, 2]
    @test full.snapshots[:, :, 2] != full.snapshots[:, :, 1]
    @test length(full.E1) == 81

    # a non-finite S ends the run with its step
    broken = run_dlra(case, 32, 64, 4; perturb = (s, S) -> s == 3 && (S[1] = NaN))
    @test broken.status == "failed at step 3"
    @test broken.failed_step == 3
    @test length(broken.E1) == 3

    # the rank is at least one and at most the smaller grid dimension
    for r in (0, 33)
        e = thrown(() -> run_dlra(case, 32, 64, r))
        @test e isa ArgumentError
        @test occursin(Regex("^the rank $r is not in 1:32 for the 32 × 64 grid\$"), e.msg)
    end
end
