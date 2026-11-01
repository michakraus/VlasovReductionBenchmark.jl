using LinearAlgebra: I, norm, pinv
using Random: Xoshiro
using GeometricMachineLearning: GeometricMachineLearning, BilinearStiefelLayer, CPU, Chain,
                                NeuralNetwork, PSDArch, StiefelManifold, params
using GeometricOptimizers: Adam, CompositeMethod, FactoredStiefelAdam, TrainingOptimizer,
                           optimization_step!, solve!
using VlasovReductionBenchmark
using VlasovReductionBenchmark: NETWORKS, VlasovCase, fluctuation, maxwellian,
                                psd_coordinates, relative_projection_error, train_network!,
                                weight_tied_network
using Test
using Zygote: Zygote

include("../helpers/thrown.jl")

const GML = GeometricMachineLearning
const NX, NV = 32, 64

# 41 snapshots of a free-streaming Landau perturbation on the smoke grid: their fluctuations
# have an even and an odd part in v, and enough of them for the 64 PSD modes of the untied network.
const CASE = linear_landau()
const X, V = grid(CASE, NX, NV)
const S = stack(
    t -> maxwellian.(V') .* (1 .+ 0.05 .* cos.(0.5 .* X .- t .* V')), range(0, 10; length = 41))
const F0 = background(S[:, :, 1])

# The `StiefelManifold` leaves of a parameter tree.
stiefel_leaves(Y::StiefelManifold) = [Y]
function stiefel_leaves(t::Union{NamedTuple, Tuple})
    reduce(vcat, map(stiefel_leaves, values(t)); init = StiefelManifold[])
end
stiefel_leaves(t::GML.NetworkParameters) = stiefel_leaves(params(t))
stiefel_leaves(::Any) = StiefelManifold[]
orthonormality_defect(Y) = norm(transpose(parent(Y)) * parent(Y) - I)

# The full-order field of `CASE` on the smoke grid, as a function of the state.
const FIELD, FIELD_PARAMS = full_order_field(CASE, NX, NV)
function arakawa(f::AbstractVector)
    ḟ = similar(f)
    FIELD(ḟ, 0.0, copy(f), FIELD_PARAMS)
    return ḟ
end

# Both networks after 20 training steps, at n = 4.
const TRAINED = Dict(name => build(S, F0, 4; seed = 42, steps = 20, label = name)
for (name, build) in NETWORKS)

@testset "The weight-tied network" begin
    G = fluctuation(S, F0)
    nn = weight_tied_network(G, 4; seed = 42)
    ps = params(nn)
    # the layer's bases are those of its own `solve!` on the fluctuations
    frontend = BilinearStiefelLayer(NX, NV, 2, 2)
    reference = params(NeuralNetwork(Chain(frontend), CPU(), Float64; rng = Xoshiro(1))).L1
    solve!(frontend, reference, G)
    @test parent(ps.frontends.grid.Φx) == parent(reference.Φx)
    @test parent(ps.frontends.grid.Φv) == parent(reference.Φv)
    # the core's last layer is the PSD lift of `PSDArch(8, 4)` `solve!` on the layer's coordinates
    coordinates = frontend(G, reference)
    psd = NeuralNetwork(PSDArch(8, 4), CPU(), Float64; rng = Xoshiro(1))
    solve!(psd, vcat(coordinates.q, coordinates.p))
    Φ = parent(params(psd)[1].weight)
    @test parent(ps.core[end].weight) == Φ
    # so the decode–encode of the core is the projection of that lift, whatever the layers between
    for s in (1, 20, 41)
        g = G[:, :, s]
        y = frontend(g, reference)
        ŷ = frontend(GML.decode(nn, :grid, GML.encode(nn, :grid, g)), reference)
        @test vcat(ŷ.q, ŷ.p) ≈ vcat(Φ * (Φ' * y.q), Φ * (Φ' * y.p)) rtol=1e-12
    end
end

@testset "Encode and decode" begin
    G = fluctuation(S, F0)
    # the weight-tied network: the round trip is the network's own reconstruction
    trained = TRAINED["weight-tied"]
    r, nn = trained.reduction, trained.network
    for s in (1, 20, 41)
        own = GML.decode(nn, :grid, GML.encode(nn, :grid, G[:, :, s])) .+ F0'
        @test decode(r, encode(r, S[:, :, s])) ≈ own rtol=1e-12
    end
    @test encode(r, S[:, :, 7]) ≈ GML.encode(nn, :grid, G[:, :, 7]) rtol=1e-12
    # the encoder–decoder identity on the latent side
    for seed in 1:3
        z = randn(Xoshiro(seed), 4)
        @test encode(r, decode(r, z)) ≈ z rtol=1e-12
    end
    @test relative_projection_error(r, S) ≈
          sqrt(sum(
        s -> sum(abs2,
            GML.decode(nn, :grid, GML.encode(nn, :grid, G[:, :, s])) - G[:, :, s]),
        axes(G, 3)) / sum(abs2, G)) rtol=1e-12

    # the untied network: its reconstruction of the 128 PSD coordinates, lifted to the grid
    trained = TRAINED["untied"]
    r, nn = trained.reduction, trained.network
    W = parity_snapshots(S, F0)
    Φ = psd_basis(W, 128)
    @test size(Φ, 2) == 64
    N = size(Φ, 1)
    for s in (1, 20, 41)
        y = vcat(Φ' * W[1:N, s], Φ' * W[(N + 1):end, s])
        ŷ = nn(y)
        own = parity_join(vcat(Φ * ŷ[1:64], Φ * ŷ[65:end]), F0)
        @test decode(r, encode(r, S[:, :, s])) ≈ own rtol=1e-12
        @test psd_coordinates(Φ, W[:, s]) == y
    end
    @test psd_coordinates(Φ, W) == vcat(Φ' * W[1:N, :], Φ' * W[(N + 1):end, :])
end

@testset "Training" begin
    # the Stiefel leaves stay orthonormal under the steps of `FactoredStiefelAdam`
    for (name, trained) in TRAINED
        @test trained.training.status == "ok"
        @test trained.training.last < trained.training.first
        leaves = stiefel_leaves(params(trained.network))
        @test !isempty(leaves)
        for Y in leaves
            @test orthonormality_defect(Y) ≤ 50 * eps()
        end
    end
    @test length(stiefel_leaves(params(TRAINED["weight-tied"].network))) == 3

    # the steps are those of `FactoredStiefelAdam` on the Stiefel leaves: plain `Adam` there
    # keeps them orthonormal as well, so only the steps themselves tell the two apart
    G = fluctuation(S, F0)
    loss = GML.CharbonnierRelativeLoss()
    function stepped(algorithm)
        nn = weight_tied_network(G, 4; seed = 42)
        ps = params(nn)
        optimizer = TrainingOptimizer(ps; algorithm, linesearch = 1e-3)
        for _ in 1:3
            optimization_step!(ps, optimizer,
                Zygote.gradient(p -> loss(nn.model, p, :grid, G), ps)[1])
        end
        return stiefel_leaves(ps)
    end
    nn = weight_tied_network(G, 4; seed = 42)
    try
        train_network!(p -> loss(nn.model, p, :grid, G), params(nn), 3; label = "three")
    catch e
        # three steps need not lower the loss; the leaves have taken them either way
        e isa ErrorException || rethrow()
    end
    trained = stiefel_leaves(params(nn))
    factored = stepped(CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam()))
    plain = stepped(CompositeMethod(; manifold = Adam(), array = Adam()))
    @test all(parent(a) ≈ parent(b) for (a, b) in zip(trained, factored))
    @test !all(parent(a) ≈ parent(b) for (a, b) in zip(trained, plain))

    # a training that does not move: the step size that GeometricOptimizers takes closest to zero
    for (name, build) in NETWORKS
        e = thrown(() -> build(S, F0, 4; seed = 42, steps = 3, step_size = nextfloat(0.0),
            label = "$name, still"))
        @test e isa ErrorException
        @test occursin(
            Regex("^$name, still: the training loss went from (\\S+) to \\1, not below\$"),
            e.msg)
    end

    # a loss that becomes non-finite ends the training with the step after which it is
    nn = TRAINED["untied"].network
    Y = randn(Xoshiro(5), 128, 3)
    function counting_loss(calls, nan_from)
        return ps -> begin
            k = Zygote.@ignore calls[] += 1
            loss = sum(abs2, nn.model(Y, ps) .- Y)
            k ≥ nan_from ? loss * NaN : loss
        end
    end
    for (steps, nan_from, status) in ((10, 4, "failed at step 3"), (
        3, 4, "failed at step 3"),
        (10, 1, "failed at step 0"))
        ps = deepcopy(params(nn))
        result = train_network!(counting_loss(Ref(0), nan_from), ps, steps; label = "nan")
        @test result.status == status
        @test !isfinite(result.last)
        # the first loss is the loss before the first step
        nan_from > 1 && @test result.first == sum(abs2, nn.model(Y, params(nn)) .- Y)
    end

    # the reconstruction term of the untied loss is GML's `CharbonnierRelativeLoss`: of the
    # weight-tied network, whose latent term vanishes, as encode ∘ decode is the identity
    nn = TRAINED["weight-tied"].network
    G = fluctuation(S, F0)
    Ĝ = stack(s -> GML.decode(nn, :grid, GML.encode(nn, :grid, G[:, :, s])), axes(G, 3))
    @test VlasovReductionBenchmark.charbonnier_reconstruction(Ĝ, G, 1e-3) ≈
          GML.CharbonnierRelativeLoss()(nn.model, params(nn), :grid, G) rtol=1e-10
end

@testset "Manifold Galerkin field of the networks" begin
    ż = zeros(4)
    for (name, trained) in TRAINED
        r = trained.reduction
        z = encode(r, S[:, :, 13]) .+ 1e-3 .* randn(Xoshiro(3), 4)
        # (ii) the Jacobian against a central finite difference of the decoder
        J = decoder_jacobian(r, z)
        @test size(J) == (NX * NV, 4)
        h = 1e-6
        Jfd = reduce(hcat,
            [(vec(decode(r, z .+ h .* (1:4 .== i))) -
              vec(decode(r, z .- h .* (1:4 .== i)))) ./
             2h for i in 1:4])
        @test norm(J - Jfd) ≤ sqrt(eps()) * norm(J)
        # the field is the least-squares solve against that Jacobian
        ManifoldGalerkinField(r, FIELD, FIELD_PARAMS)(ż, 0.0, z, nothing)
        @test ż ≈ pinv(J) * arakawa(vec(decode(r, z))) rtol=1e-10

        # (iii) the decoded initial state is the snapshot to the network's reconstruction error
        f = S[:, :, 1]
        short = VlasovCase(CASE.name, CASE.profile, CASE.α, CASE.k, 1.0)
        reduced = run_reduced(r, short, NX, NV, f)
        @test reduced.status == "ok"
        @test reduced.states[1] == encode(r, f)
        @test norm(reduced.snapshots[:, :, 1] - f) ≈ norm(decode(r, encode(r, f)) - f) rtol=1e-12
        @test length(reduced.times) == 21
        # the run moves, and each state is kept as it was: a snapshot every 10 steps
        @test reduced.states[end] != reduced.states[1]
        for k in 1:3
            @test reduced.snapshots[:, :, k] == decode(r, reduced.states[10k - 9])
        end
        # a step that leaves a non-finite state ends the run
        broken = run_reduced(
            r, short, NX, NV, f; perturb = (s, z) -> s == 2 && (z[1] = NaN))
        @test (broken.status, broken.failed_step) == ("failed at step 2", 2)
    end
end
