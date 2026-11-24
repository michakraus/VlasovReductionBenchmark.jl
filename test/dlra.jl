using LinearAlgebra: norm
using VlasovReductionBenchmark
# VlasovMethods is no test dependency; the benchmark's own import of it is
using VlasovReductionBenchmark.VlasovMethods: VlasovMethods, AugmentedBUG, GridDistribution,
                                              LowRankDistribution, lowrank_step!
using VlasovReductionBenchmark: DLRA_STRIDE, DLRA_TIMESTEP, SNAPSHOT_INTERVAL, TIMESTEP,
                                background, dlra_mode, full_order_stepper,
                                relative_l2_errors, spectral_derivative, vlasov_model
using Test

include("helpers/thrown.jl")

const CASES = (linear_landau(), nonlinear_landau(), two_stream())

# The `nx × nv` node values of the low-rank state `Y`.
node_values(Y) = GridDistribution(Y).values

@testset "Spectral derivative" begin
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
end

@testset "Constants and modes" begin
    # the DLRA runs step as the full-order runs do, and store their snapshots every 0.5
    @test DLRA_TIMESTEP == TIMESTEP == 0.05
    @test DLRA_STRIDE == 10
    @test DLRA_STRIDE * DLRA_TIMESTEP == SNAPSHOT_INTERVAL
    full, smoke = dlra_mode(false), dlra_mode(true)
    @test (full.grid, full.ranks, full.landau_window, full.landau_rtol) ==
          ((64, 128), (4, 8, 16, 32), (0.0, 40.0), 0.02)
    @test (smoke.grid, smoke.ranks, smoke.landau_window, smoke.landau_rtol) ==
          ((32, 64), (4, 8), (0.0, 25.0), 0.10)
    # the state size r (Nx + Nv) + r², in closed form
    @test dlra_state_size(4, 64, 128) == 784
    @test dlra_state_size(16, 32, 64) == 1792
end

@testset "A DLRA step is one step of VlasovMethods' augmented BUG" begin
    # one step of the run against one step of AugmentedBUG() on the factorisation of rank 4
    # with no fixed modes
    case = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 0.05)
    run = run_dlra(case, 32, 64, 4)
    @test run.status == "ok"
    model = vlasov_model(case, 32, 64)
    Y = LowRankDistribution(model.distribution, 4)
    @test Y.nfixed == 0
    f0 = node_values(Y)
    field = VlasovMethods.LowRankFields1D1V(Y, model.work)
    lowrank_step!(Y, AugmentedBUG(), field, 0.0, TIMESTEP)
    @test (run.factors.X, run.factors.S, run.factors.V) == (Y.X, Y.S, Y.V)
    @test run.factors.nfixed == Y.nfixed == 0
    @test norm(node_values(run.factors) - f0) > 1e-6 * norm(f0)
    @test run.snapshots[:, :, 1] ≈ f0 rtol=1e-14

    # the timed steps of the online cost are three such steps
    timed = LowRankDistribution(model.distribution, 4)
    @test VlasovReductionBenchmark.timed_bug_steps(timed, field, 3) > 0
    for _ in 1:2
        lowrank_step!(Y, AugmentedBUG(), field, 0.0, TIMESTEP)
    end
    @test (timed.X, timed.S, timed.V) == (Y.X, Y.S, Y.V)
end

@testset "Full rank agrees with the full-order run" begin
    for case in CASES
        err = full_rank_agreement(case, 32, 64, 32, 20)
        @test isfinite(err)
        @test err ≤ 1e-10
        # the reference is the integrator of run_case: its state after 10 steps is the second
        # snapshot of the full-order run, bit for bit
        step! = full_order_stepper(case, 32, 64, 10)
        local f
        for s in 1:10
            f = step!(s)
        end
        @test f == run_case(case, 32, 64).snapshots[:, :, 2]
    end
    for case in (linear_landau(), two_stream())
        err = full_rank_agreement(case, 64, 128, 64, 40)
        @test isfinite(err)
        @test err ≤ 1e-10
    end
    # a DLRA run made non-finite at step 5: the agreement is NaN, and the check throws
    perturb = (s, S) -> s == 5 && (S[1] = NaN)
    @test isnan(full_rank_agreement(linear_landau(), 32, 64, 32, 20; perturb))
    e = thrown(() -> redirect_stdout(devnull) do
        VlasovReductionBenchmark.check_full_rank_agreement(linear_landau(), 32, 64; perturb)
    end)
    @test e isa ErrorException
    @test occursin(
        r"^full-rank check linear_landau on 32 × 64: the DLRA run at r = 32 differs from the full-rank solver by NaN relative over 20 steps, above 1\.0e-10$",
        e.msg)
    # the check passes and returns the agreement where nothing breaks the run
    err = redirect_stdout(devnull) do
        VlasovReductionBenchmark.check_full_rank_agreement(linear_landau(), 32, 64)
    end
    @test err == full_rank_agreement(linear_landau(), 32, 64, 32, 20)
end

@testset "The damping rate of linear Landau damping" begin
    oracle = imag(landau_root(0.5))
    run = run_dlra(linear_landau(), 64, 128, 16)
    @test run.status == "ok"
    γ = landau_rate_fit(run.times, log.(run.E1), (0.0, 40.0)).γ
    @test abs(γ - oracle) ≤ 0.02 * abs(oracle)

    # the checks of the smoke mode pass, the full-order run first
    mktempdir() do dir
        path = joinpath(dir, "rates.txt")
        result = open(path, "w") do io
            redirect_stdout(io) do
                VlasovReductionBenchmark.check_dlra_rates(dlra_mode(true))
            end
        end
        @test result === nothing
        printed = read(path, String)
        first_check = findfirst("rate check full-order run linear Landau γ", printed)
        second_check = findfirst("rate check DLRA r = 16 linear Landau γ", printed)
        @test first_check !== nothing
        @test second_check !== nothing
        @test first(first_check) < first(second_check)
    end
    # a DLRA run that stops fails them
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
    @test run.times == [s * TIMESTEP for s in 0:40]
    @test run.times ≈ 0:0.05:2.0
    for series in (run.mass, run.energy, run.E1, run.l1, run.l2, run.entropy,
        run.nonpositive)
        @test length(series) == 41
    end
    @test eltype(run.E1) == eltype(run.l2) == Float64
    # the snapshots every 0.5, the first the initial condition, the last the final state
    @test size(run.snapshots) == (32, 64, 5)
    @test eltype(run.snapshots) == Float64
    f_init = initial_condition(case, grid(case, 32, 64)...)
    @test run.snapshots[:, :, 1] ≈ f_init rtol=1e-14
    @test run.snapshots[:, :, end] ≈ node_values(run.factors) rtol=1e-14
    @test eltype(run.factors.S) == Float64
    @test size(run.factors.S) == (4, 4)
    @test run.factors.nfixed == 0
    @test run.offline_seconds > 0
    # the second snapshot is the state at t = 0.5, the last of a run to t = 0.5
    half = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 0.5)
    @test run.snapshots[:, :, 2] == run_dlra(half, 32, 64, 4).snapshots[:, :, 2]
    @test run.snapshots[:, :, 2] != run.snapshots[:, :, 1]

    # a non-finite S ends the run with its step
    broken = run_dlra(case, 32, 64, 4; perturb = (s, S) -> s == 3 && (S[1] = NaN))
    @test broken.status == "failed at step 3"
    @test broken.failed_step == 3
    @test length(broken.E1) == length(broken.mass) == 3

    # the rank is at least one and at most the smaller grid dimension
    for r in (0, 33)
        e = thrown(() -> run_dlra(case, 32, 64, r))
        @test e isa ArgumentError
        @test occursin(Regex("^the rank $r is not in 1:32 for the 32 × 64 grid\$"), e.msg)
    end
end

@testset "A DLRA row against the full-order run" begin
    # a short nonlinear Landau run, which has no fitted rate, so no fit window matters
    case = VlasovCase("nonlinear_landau", maxwellian, 0.5, 0.5, 5.5)
    run = run_case(case, 32, 64)
    row = VlasovReductionBenchmark.dlra_data_row(run, "data", 4, dlra_mode(true))
    dlra = run_dlra(case, 32, 64, 4)
    value(name) = parse(Float64, row[name])

    # the error of the rank against the full-order run, relative to its fluctuation
    against_run = relative_l2_errors(dlra.snapshots, run.snapshots,
        background(run.snapshots[:, :, 1]))
    @test value("max_l2_error_fluct") ≈ maximum(against_run) rtol=8eps()
    @test value("l2_error_max") ≈
          maximum(relative_l2_errors(dlra.snapshots, run.snapshots)) rtol=8eps()
    # the reference of the two full-rank columns is the full-order run itself
    @test row["full_rank_l2_error_max"] == "0.0"
    @test row["max_l2_error_fluct_full_rank"] == row["max_l2_error_fluct"]
    @test value("max_l2_error_fluct") > 0

    # a run that stops leaves the column of the DLRA error empty
    broken = VlasovReductionBenchmark.dlra_data_row(run, "data", 4, dlra_mode(true);
        perturb = (s, S) -> s == 2 && (S[1] = NaN))
    @test broken["status"] == "failed at step 2"
    @test broken["full_rank_l2_error_max"] == "0.0"
    @test broken["max_l2_error_fluct_full_rank"] == ""
    @test broken["max_l2_error_fluct"] == ""
end
