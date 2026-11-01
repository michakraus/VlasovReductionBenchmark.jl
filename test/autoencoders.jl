using ForwardDiff: ForwardDiff
using LinearAlgebra: norm, pinv, qr
using Random: Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: AUTOENCODER_COLUMNS, FULL_MODE, GridCoordinates,
                                LinearReduction, PARITY_MODES, RankDeficientDecoder,
                                SMOKE_MODE, VlasovCase, autoencoder_mode, check_rank,
                                diagnostics, galerkin_field, write_autoencoder_rows
using Test

include("helpers/thrown.jl")

const NX, NV = 32, 64
const CASE = linear_landau()
const X, V = grid(CASE, NX, NV)
const F_INIT = initial_condition(CASE, X, V)
const F0 = background(F_INIT)

# The full-order field of `CASE` on the smoke grid, as a function of the state.
const FIELD, FIELD_PARAMS = full_order_field(CASE, NX, NV)
function arakawa(f::AbstractVector)
    ḟ = similar(f)
    FIELD(ḟ, 0.0, copy(f), FIELD_PARAMS)
    return ḟ
end
with_background(g::AbstractVector) = g .+ repeat(F0; inner = NX)

@testset "Modes" begin
    @test autoencoder_mode(true) ==
          (grid_mode = SMOKE_MODE, ns = (4, 8), steps = 100, seeds = (42,))
    @test autoencoder_mode(false) ==
          (grid_mode = FULL_MODE, ns = (4, 8, 16, 32), steps = 2000, seeds = (42, 43, 44))
    @test PARITY_MODES == Dict(4 => (2, 2), 8 => (2, 4), 16 => (4, 4), 32 => (4, 8))
end

@testset "Manifold Galerkin field of a linear decoder" begin
    rng = Xoshiro(11)
    z = 1e-2 .* randn(rng, 4)
    ż = zeros(4)
    # (i) an orthonormal linear decoder gives the Galerkin field of the same basis
    B = Matrix(qr(randn(rng, NX * NV, 4)).Q)
    linear = AutoencoderReduction(g -> B' * vec(g), identity, B, F0, NX, NV)
    ManifoldGalerkinField(linear, FIELD, FIELD_PARAMS)(ż, 0.0, z, nothing)
    expected = zeros(4)
    galerkin_field(LinearReduction(GridCoordinates(), B, F0, NX, NV), FIELD, FIELD_PARAMS)(
        expected, 0.0, z, nothing)
    @test ż ≈ expected rtol=1e-12
    @test ż ≈ B' * arakawa(with_background(B * z)) rtol=1e-12
    # a linear decoder whose columns are not orthonormal gives the least-squares solution
    A = B * [1.0 0.3 0.0 0.2; 0.0 2.0 0.5 0.0; 0.0 0.0 0.5 0.1; 0.0 0.0 0.0 3.0]
    skewed = AutoencoderReduction(g -> pinv(A) * vec(g), identity, A, F0, NX, NV)
    ManifoldGalerkinField(skewed, FIELD, FIELD_PARAMS)(ż, 0.0, z, nothing)
    @test ż ≈ pinv(A) * arakawa(with_background(A * z)) rtol=1e-12
    @test !isapprox(ż, A' * arakawa(with_background(A * z)); rtol = 1e-2)
    # a lift that does not fit the grid
    e = thrown(() -> AutoencoderReduction(
        identity, identity, A[1:(end - 1), :], F0, NX, NV))
    @test e isa DimensionMismatch
    @test occursin(r"^a lift of 2047 rows does not fit a 32 × 64 grid$", e.msg)
    # the Jacobian of a linear decoder is its matrix
    @test decoder_jacobian(skewed, z) ≈ A rtol=1e-14
    @test decode(skewed, z) ≈ reshape(with_background(A * z), NX, NV) rtol=1e-14
end

@testset "Rank of the decoder" begin
    for M in (zeros(3, 2), [1.0 0.0; 0.0 1e-13; 0.0 0.0], [1.0 1.0; 1.0 1.0; 0.0 0.0])
        e = thrown(() -> check_rank(M))
        @test e isa RankDeficientDecoder
        @test occursin(
            r"^the decoder's Jacobian has the relative singular value \S+, below 1\.0e-12$",
            sprint(showerror, e))
    end
    @test check_rank([1.0 0.0; 0.0 1e-12; 0.0 0.0]) === nothing
    @test check_rank([1.0 0.0; 0.0 5e-12; 0.0 0.0]) === nothing
    @test check_rank([1.0 0.0; 0.0 1e-11; 0.0 0.0]) === nothing
    @test thrown(() -> check_rank([2.0 0.0; 0.0 1.9e-12; 0.0 0.0])) isa RankDeficientDecoder
    # through dual numbers, as the Newton solve of a step evaluates the field
    @test thrown(() -> check_rank(ForwardDiff.Dual.([1.0 0.0; 0.0 0.0], 1.0))) isa
          RankDeficientDecoder
    @test check_rank(ForwardDiff.Dual.([1.0 0.0; 0.0 1.0; 0.0 0.0], 1.0)) === nothing
    @test check_rank(ForwardDiff.Dual.(ForwardDiff.Dual.([1.0 0.0; 0.0 1.0], 1.0), 2.0)) ===
          nothing

    # a decoder of rank 1 ends the run at the initial state
    B = Matrix(qr(randn(Xoshiro(2), NX * NV, 2)).Q)
    encoder = g -> B' * vec(g)
    flat = AutoencoderReduction(encoder, z -> [z[1] + z[2], z[1] + z[2]], B, F0, NX, NV)
    short = VlasovCase(CASE.name, CASE.profile, CASE.α, CASE.k, 1.0)
    reduced = run_reduced(flat, short, NX, NV, F_INIT)
    @test (reduced.status, reduced.failed_step) ==
          ("failed at step 0: rank-deficient decoder", 0)
    @test isempty(reduced.states)
    @test isempty(reduced.energy)
    # a decoder that loses its rank after 100 evaluations at a state, inside a step: the first
    # solution step evaluates the field 77 times (measured)
    calls = Ref(0)
    function losing(z)
        eltype(z) === Float64 && (calls[] += 1)
        return calls[] > 100 ? [z[1] + z[2], z[1] + z[2]] : copy(z)
    end
    reduced = run_reduced(
        AutoencoderReduction(encoder, losing, B, F0, NX, NV), short, NX, NV,
        F_INIT)
    @test occursin(r"^failed at step [1-9]\d*: rank-deficient decoder$", reduced.status)
    @test reduced.failed_step ≥ 1
    @test length(reduced.states) == reduced.failed_step
end

@testset "The canonical flow of the pulled-back energy" begin
    model = vlasov_model(CASE, NX, NV)
    hx, hv = step(X), step(V)
    # the energy of `diagnostics` in parity coordinates, differentiated by ForwardDiff
    energy(w) = diagnostics(vec(parity_join(w, F0)), model.work, CASE.k, X, V, hx, hv).energy
    w = parity_split(F_INIT, F0)
    ∇H = ForwardDiff.gradient(energy, w)
    N = length(w) ÷ 2
    flow = vcat(∇H[(N + 1):end], -∇H[1:N])
    field = parity_split(reshape(arakawa(vec(F_INIT)), NX, NV), zeros(NV))
    comparison = canonical_flow_comparison(CASE, F_INIT)
    @test comparison.cosine ≈ flow' * field / (norm(flow) * norm(field)) atol=1e-14
    @test comparison.norm_ratio ≈ norm(flow) / norm(field) rtol=1e-10
    # the record for the bracket: the two are orthogonal, and the flow is much larger
    @test abs(comparison.cosine) < 1e-12
    @test comparison.norm_ratio > 100
end

@testset "The CSV file" begin
    @test AUTOENCODER_COLUMNS[1:6] == ("method", "kind", "case", "alpha", "n", "seed")
    @test AUTOENCODER_COLUMNS[(end - 1):end] == ("loss_stage_I", "loss_final")
    mktempdir() do dir
        path = write_autoencoder_rows(joinpath(dir, "a.csv"),
            [Dict("method" => "untied", "status" => "failed, at step 3", "seed" => "42")])
        lines = readlines(path)
        @test lines[1] == join(AUTOENCODER_COLUMNS, ',')
        fields = split(lines[2], ',')
        @test fields[1] == "untied"
        @test fields[6] == "42"
        @test occursin("\"failed, at step 3\"", lines[2])
    end
end
