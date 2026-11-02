using ForwardDiff: ForwardDiff
using GeometricMachineLearning: Batch, CPU, DataLoader, GSympNet, NeuralNetwork, train!
using GeometricOptimizers: Adam, TrainingOptimizer
using LinearAlgebra: I
using Random: Random, Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: invariants
using Test

include("helpers/thrown.jl")

const NX, NV = 32, 64

# The 40 snapshots of a two-stream run on the smoke grid, t ∈ [0, 19.5].
const TWO_STREAM = run_case(VlasovCase("two_stream", two_stream_profile, 0.05, 0.5, 19.5),
    NX, NV)

# Two harmonic oscillators, q_j = cos(ω_j t) and p_j = -sin(ω_j t), ω = (1, 1.5), 40 columns at
# Δt = 0.5, as the latent vector [q; p].
const ω = [1.0, 1.5]
const OSCILLATOR = vcat(cos.(ω * (0.5 * (0:39))'), -sin.(ω * (0.5 * (0:39))'))
const M_TRAIN = 20

J(n) = [zeros(n, n) I(n); -I(n) zeros(n, n)]

# Every parameter of a network, flattened into one vector.
leaves(x::AbstractArray{<:Number}) = vec(collect(x))
leaves(x::Union{NamedTuple, Tuple}) = reduce(vcat, [leaves(y) for y in values(x)])
leaves(x) = leaves(getfield(x, :params))
weights(nn) = leaves(nn.params)

# A network stub whose call `k` returns column `k + 1` of `Z`, whatever its argument: a rollout
# from `Z[:, 1]` then gives `Z` itself.
struct Replay
    Z::Matrix{Float64}
    calls::Base.RefValue{Int}
end
Replay(Z) = Replay(Z, Ref(0))
(r::Replay)(z) = r.Z[:, (r.calls[] += 1) + 1]

@testset "split" begin
    S = TWO_STREAM.snapshots
    @test size(S, 3) == 40
    f0 = background(S[:, :, 1])
    r = psd_reduction(S, f0, 8)
    Z = latent_snapshots(r, S)
    @test size(Z) == (8, 40)
    W = parity_snapshots(S, f0)
    N = NX * NV ÷ 2
    Φ = psd_basis(W, 8)
    qp = qp_split(Z)
    # q is Φᵀ times the odd parity block, p is Φᵀ times the even block
    @test isapprox(qp.q, Φ' * W[1:N, :]; rtol = 1e-12)
    @test isapprox(qp.p, Φ' * W[(N + 1):(2N), :]; rtol = 1e-12)
    @test qp_split([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]) ==
          (q = [1.0 2.0; 3.0 4.0], p = [5.0 6.0; 7.0 8.0])

    # the columns after `m_train` are not read
    Zn = copy(OSCILLATOR)
    Zn[:, (M_TRAIN + 1):end] .= NaN
    trained = train_latent_sympnet(Zn, M_TRAIN; epochs = 5, seed = 42)
    @test length(trained.history) == 5
    @test all(isfinite, trained.history)
    @test all(isfinite, weights(trained.network))
end

@testset "seed" begin
    Random.seed!(1)
    a = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 20, seed = 42)
    Random.seed!(2)
    b = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 20, seed = 42)
    c = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 20, seed = 43)
    @test weights(a.network) == weights(b.network)
    @test a.history == b.history
    @test maximum(abs, weights(a.network) - weights(c.network)) > 1e-3
    @test a.network.architecture isa GSympNet
    @test a.network.architecture.dim == 4
    @test a.network.architecture.upscaling_dimension == 8
    @test a.network.architecture.n_layers == 4
    @test eltype(weights(a.network)) == Float64

    # with minibatches of 4 of the 19 pairs the shuffle orders the steps; the seed fixes it too
    Random.seed!(1)
    a4 = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 20, seed = 42, batch_size = 4)
    Random.seed!(2)
    b4 = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 20, seed = 42, batch_size = 4)
    @test weights(a4.network) == weights(b4.network)
    @test a4.history == b4.history
    @test a4.history != a.history

    # the protocol built from GeometricMachineLearning directly: parameters from Xoshiro(seed),
    # Adam at the package's step size, and the shuffle of the global generator seeded with seed
    nn = NeuralNetwork(GSympNet(4; upscaling_dimension = 8, n_layers = 4), CPU(), Float64;
        rng = Xoshiro(42))
    optimizer = TrainingOptimizer(nn.params; algorithm = Adam(),
        linesearch = VlasovReductionBenchmark.TRAINING_STEP_SIZE)
    loader = DataLoader((q = OSCILLATOR[1:2, 1:M_TRAIN], p = OSCILLATOR[3:4, 1:M_TRAIN]);
        suppress_info = true)
    Random.seed!(42)
    history = train!(nn, optimizer, loader, Batch(4), 20; show_progress = false)
    @test history == a4.history
    @test weights(nn) == weights(a4.network)
    two = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 1, seed = 42, n_layers = 2)
    @test two.network.architecture.n_layers == 2

    # the global generator is left as it was
    Random.seed!(7)
    x = rand()
    Random.seed!(7)
    train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 1, seed = 42)
    @test rand() == x
end

@testset "symplecticity" begin
    nn = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 20, seed = 42).network
    for s in (1, 10, 30)
        M = ForwardDiff.jacobian(nn, OSCILLATOR[:, s])
        @test maximum(abs, M' * J(2) * M - J(2)) ≤ 1e-10
        @test maximum(abs, M - I) > 1e-3
    end
end

@testset "loss" begin
    history = train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 200, seed = 42).history
    @test length(history) == 200
    @test history[end] < history[1] / 2
end

@testset "rollout" begin
    nn = NeuralNetwork(GSympNet(4; upscaling_dimension = 8, n_layers = 4), CPU(), Float64;
        rng = Xoshiro(1))
    for x in nn.params, y in values(x)

        fill!(y, 0.0)
    end
    z0 = [0.3, -0.2, 0.5, 0.1]
    Z = latent_rollout(nn, z0, 7)
    @test size(Z) == (4, 7)
    @test all(Z[:, s] == z0 for s in 1:7)

    # column s + 1 is the step of column s
    Z2 = latent_rollout(z -> 2z, z0, 5)
    @test size(Z2) == (4, 5)
    @test all(Z2[:, s] == 2^(s - 1) * z0 for s in 1:5)
end

@testset "run" begin
    S = TWO_STREAM.snapshots
    m = size(S, 3)
    f0 = background(S[:, :, 1])
    r = pod_reduction(S, f0, m)
    Z = latent_snapshots(r, S)
    case = VlasovCase("two_stream", two_stream_profile, 0.05, 0.3, 19.5)

    # one changed snapshot after m_train
    Zc = copy(Z)
    Zc[:, M_TRAIN + 1] .*= 1.5
    run = latent_run(r, case, S, Replay(Zc), M_TRAIN)
    @test run.status == "ok"
    @test run.diverged_at === nothing
    @test run.l2_error_in_sample < 1e-10
    changed = reshape(decode(r, Zc[:, M_TRAIN + 1]), NX, NV, 1)
    e = only(relative_l2_errors(changed, S[:, :, (M_TRAIN + 1):(M_TRAIN + 1)], f0))
    @test e > 0.1
    @test isapprox(run.l2_error_extrapolation, e / sqrt(m - M_TRAIN); rtol = 1e-8)

    # the drifts of the invariants on the grid of the case, k = 0.3
    run = latent_run(r, case, S, Replay(Z), M_TRAIN)
    @test run.status == "ok"
    @test size(run.snapshots) == size(S)
    @test isapprox(run.snapshots, S; rtol = 1e-12)
    model = vlasov_model(case, NX, NV)
    x, v = model.distribution.x, model.distribution.v
    series = [invariants(run.snapshots[:, :, s], model.work, 0.3, x, v, step(x), step(v))
              for s in 1:m]
    for (name, drift) in ((:energy, :energy_drift), (:l1, :l1_drift), (:l2, :l2_drift),
        (:mass, :mass_drift))
        @test isapprox(
            getfield(run, drift), max_drift([getfield(d, name) for d in series]);
            rtol = 1e-12)
    end
    @test run.energy_drift > 1e-6
end

@testset "diverged" begin
    S = TWO_STREAM.snapshots
    f0 = background(S[:, :, 1])
    r = pod_reduction(S, f0, 8)
    Z = latent_snapshots(r, S)
    Z[:, 5] .= Inf
    run = latent_run(r, linear_landau(), S, Replay(Z), M_TRAIN)
    @test run.status == "diverged"
    @test run.diverged_at == 5
    @test isnan(run.l2_error_in_sample)
    @test isnan(run.l2_error_extrapolation)
    @test all(isnan, (run.energy_drift, run.l1_drift, run.l2_drift, run.mass_drift))
    # the columns before the first non-finite one are decoded, the rest are NaN
    @test all(run.snapshots[:, :, s] ≈ decode(r, Z[:, s]) for s in 1:4)
    @test all(isnan, run.snapshots[:, :, 5:end])

    # a NaN in one entry diverges as an Inf does
    Z = latent_snapshots(r, S)
    Z[2, 30] = NaN
    run = latent_run(r, linear_landau(), S, Replay(Z), M_TRAIN)
    @test (run.status, run.diverged_at) == ("diverged", 30)
end

@testset "edges" begin
    S = TWO_STREAM.snapshots
    m = size(S, 3)
    f0 = background(S[:, :, 1])
    r = pod_reduction(S, f0, 8)
    Z = latent_snapshots(r, S)
    case = linear_landau()

    @test_throws ArgumentError qp_split(zeros(3, 10))
    @test_throws ArgumentError train_latent_sympnet(zeros(3, 10), 5; epochs = 1, seed = 42)
    @test_throws ArgumentError train_latent_sympnet(OSCILLATOR, 1; epochs = 1, seed = 42)
    @test_throws ArgumentError train_latent_sympnet(OSCILLATOR, 41; epochs = 1, seed = 42)
    @test_throws ArgumentError train_latent_sympnet(OSCILLATOR, M_TRAIN; epochs = 0, seed = 42)
    Zn = copy(OSCILLATOR)
    Zn[2, M_TRAIN] = Inf
    @test_throws ArgumentError train_latent_sympnet(Zn, M_TRAIN; epochs = 1, seed = 42)
    @test_throws ArgumentError latent_run(r, case, S, Replay(Z), 1)
    @test_throws ArgumentError latent_run(r, case, S, Replay(Z), m + 1)

    # the bounds themselves are accepted
    @test length(train_latent_sympnet(OSCILLATOR, 2; epochs = 1, seed = 42).history) == 1
    @test length(train_latent_sympnet(OSCILLATOR, 40; epochs = 1, seed = 42).history) == 1
    @test latent_run(r, case, S, Replay(Z), 2).status == "ok"
    whole = latent_run(r, case, S, Replay(Z), m)
    @test whole.status == "ok"
    @test isfinite(whole.l2_error_in_sample)
    @test isnan(whole.l2_error_extrapolation)
end
