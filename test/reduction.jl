using LinearAlgebra: I, norm, opnorm, qr, svd
using Random: Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: GridCoordinates, LinearReduction, ParityCoordinates
using Test

include("helpers/thrown.jl")

# The snapshots of a short linear Landau run on the smoke grid, t ∈ [0, 5], 11 snapshots.
const SHORT_RUN = run_case(VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 5.0), 32, 64)

J(n) = [zeros(n, n) I(n); -I(n) zeros(n, n)]

# The bytes that one call of the vector field `g` allocates, after a first call.
function allocations(g::G, ż::A, z::A) where {G, A}
    (g(ż, 0.0, z, nothing); @allocated g(ż, 0.0, z, nothing))
end

# Grid coordinates whose map from the fluctuation throws an `ArgumentError` once `broken[]` is
# set: a step of a reduced run that raises an error which is not the solver's.
struct FailingCoordinates <: VlasovReductionBenchmark.Coordinates
    broken::Base.RefValue{Bool}
end
function VlasovReductionBenchmark.to_coordinates!(w, c::FailingCoordinates, g, nx, nv)
    c.broken[] && throw(ArgumentError("the coordinates of this probe fail"))
    return copyto!(w, g)
end
function VlasovReductionBenchmark.from_coordinates!(g, ::FailingCoordinates, w, nx, nv)
    copyto!(g, w)
end

@testset "POD basis" begin
    S = SHORT_RUN.snapshots
    f0 = background(S[:, :, 1])
    r = pod_reduction(S, f0, 4)
    V = r.basis
    @test size(V) == (32 * 64, 4)
    @test V' * V ≈ I(4) atol=1e-13
    # the leading left singular vectors of the fluctuation snapshots
    U = svd(reshape(S .- f0', 32 * 64, :)).U[:, 1:4]
    @test opnorm(U - V * (V' * U)) < 1e-10

    # a basis of the run's own first snapshot alone has one column, its fluctuation, and projects
    # that fluctuation to round-off; a basis built with f0 not subtracted misses it by O(1)
    f = S[:, :, 1]
    g = vec(f .- sum(f; dims = 1) ./ 32)
    r1 = pod_reduction(S[:, :, 1:1], f0, 1)
    V1 = r1.basis
    @test size(V1) == (32 * 64, 1)
    @test norm(g - V1 * (V1' * g)) / norm(g) < 1e-12
    @test norm(vec(decode(r1, encode(r1, f)) - f)) / norm(g) < 1e-12
    u = vec(f) / norm(f)
    @test norm(g - u * (u' * g)) / norm(g) > 0.5
    @test size(decode(r, encode(r, f))) == (32, 64)
    @test length(encode(r, f)) == 4
end

@testset "PSD basis" begin
    # a parity snapshot [Q; P] with distinct, decaying singular values of [Q P]
    N, m = 1024, 12
    W = randn(Xoshiro(11), 2N, m) .* (1 ./ (1:m))'
    Q, P = W[1:N, :], W[(N + 1):end, :]
    for n in (4, 8)
        Φ = psd_basis(W, n)
        @test size(Φ) == (N, n ÷ 2)
        A = [Φ zero(Φ); zero(Φ) Φ]
        @test norm(A' * J(N) * A - J(n ÷ 2)) < 1e-12
        U = svd(hcat(Q, P)).U[:, 1:(n ÷ 2)]
        # the sine of the largest principal angle between the two spans
        @test opnorm(U - Φ * (Φ' * U)) < 1e-10
    end

    S = SHORT_RUN.snapshots
    f0 = background(S[:, :, 1])
    r = psd_reduction(S, f0, 4)
    @test r.coordinates isa ParityCoordinates
    @test size(r.basis) == (32 * 64, 4)
    @test norm(r.basis' * J(32 * 32) * r.basis - J(2)) < 1e-12
    f = S[:, :, 1]
    @test norm(decode(r, encode(r, f)) - f) / norm(f) < 2 * 0.01
end

@testset "Snapshot projection errors" begin
    S = SHORT_RUN.snapshots
    f0 = background(S[:, :, 1])
    W = parity_snapshots(S, f0)
    @test size(W) == (32 * 64, 11)
    for n in (2, 4, 8)
        pod, psd = pod_reduction(S, f0, n), psd_reduction(S, f0, n)
        # in parity coordinates, relative to the whole parity snapshot
        A = psd.basis
        @test relative_projection_error(psd, S) ≈ norm(W - A * (A' * W)) / norm(W) rtol=1e-12
        G = reshape(S .- f0', 32 * 64, :)
        V = pod.basis
        # the parity map scales by 1/√2 and keeps the ratio
        @test relative_projection_error(pod, S) ≈ norm(G - V * (V' * G)) / norm(G) rtol=1e-12
        @test relative_projection_error(pod, S) ≤ relative_projection_error(psd, S) + 1e-12
    end
    # a full basis reproduces every snapshot
    @test relative_projection_error(pod_reduction(S, f0, 11), S) < 1e-12
end

@testset "POD optimality check" begin
    @test check_pod_optimality("linear_landau", [4, 8], [0.2, 0.1], [0.3, 0.1]) === nothing
    # an error that does not fall with n does not rise either
    @test check_pod_optimality("linear_landau", [4, 8], [0.1, 0.1], [0.1, 0.1]) === nothing
    # PSD may tie POD to 1e-12
    @test check_pod_optimality("linear_landau", [4], [0.1 + 5e-13], [0.1]) === nothing

    e = thrown(() -> check_pod_optimality("two_stream", [4, 8, 16], [0.2, 0.1, 0.15],
        [0.3, 0.2, 0.2]))
    @test e isa ErrorException
    @test occursin(
        r"^two_stream: the POD projection error rises from 0\.1 at n = 8 to 0\.15 at n = 16$",
        e.msg)
    e = thrown(() -> check_pod_optimality("two_stream", [4, 8], [0.2, 0.1], [0.3, 0.05]))
    @test e isa ErrorException
    @test occursin(
        r"^two_stream: the POD projection error 0\.1 at n = 8 exceeds the PSD error 0\.05 by more than 1\.0e-12$",
        e.msg)

    # the errors that the script writes, checked as they are computed
    S = SHORT_RUN.snapshots
    f0 = background(S[:, :, 1])
    errors = checked_projection_errors("linear_landau", S, f0, [2, 4])
    @test errors.pod ≈
          [relative_projection_error(pod_reduction(S, f0, n), S) for n in (2, 4)]
    @test errors.psd ≈
          [relative_projection_error(psd_reduction(S, f0, n), S) for n in (2, 4)]
    # with the reductions they measure, and the time of each build
    @test [r.basis for r in errors.reductions.pod] ≈
          [pod_reduction(S, f0, n).basis for n in (2, 4)]
    @test [r.basis for r in errors.reductions.psd] ≈
          [psd_reduction(S, f0, n).basis for n in (2, 4)]
    @test all(>(0), vcat(errors.seconds...))
    # a POD builder whose basis at n = 4 is the trailing modes, so the error rises with n
    trailing(S, f0, n) = n == 4 ?
                         LinearReduction(GridCoordinates(),
        pod_reduction(S, f0, 11).basis[:, 8:11], f0, 32, 64) : pod_reduction(S, f0, n)
    e = thrown(() -> checked_projection_errors(
        "linear_landau", S, f0, [2, 4]; pod = trailing))
    @test e isa ErrorException
    @test occursin(
        r"^linear_landau: the POD projection error rises from \S+ at n = 2 to \S+ at n = 4$",
        e.msg)
end

@testset "Linear reduction" begin
    # a grid basis and a parity basis of the same span decode to the same state
    nx, nv = 4, 6
    f0 = randn(Xoshiro(21), nv)
    B = Matrix(qr(randn(Xoshiro(22), nx * nv, nx * nv)).Q)
    r = LinearReduction(GridCoordinates(), B, f0, nx, nv)
    f = randn(Xoshiro(23), nx, nv)
    @test decode(r, encode(r, f)) ≈ f rtol=1e-13
    @test encode(r, f) ≈ B' * vec(f .- f0') rtol=1e-13
    rp = LinearReduction(ParityCoordinates(), B, f0, nx, nv)
    @test encode(rp, f) ≈ B' * parity_split(f, f0) rtol=1e-13
    @test decode(rp, encode(rp, f)) ≈ f rtol=1e-13

    e = thrown(() -> LinearReduction(GridCoordinates(), B[1:23, :], f0, nx, nv))
    @test e isa DimensionMismatch
    @test occursin(r"^a basis of 23 rows does not fit a 4 × 6 grid$", e.msg)
    e = thrown(() -> LinearReduction(GridCoordinates(), B, f0[1:5], nx, nv))
    @test e isa DimensionMismatch
    @test occursin(r"^the background has 5 values for 6 v nodes$", e.msg)
end

@testset "Galerkin field" begin
    # with a full orthogonal basis the reduced field is the full field in the new coordinates
    case = linear_landau()
    nx, nv = 8, 16
    model = vlasov_model(case, nx, nv)
    f = model.distribution.values
    f0 = background(f)
    F, params = full_order_field(case, nx, nv)
    ḟ = similar(vec(f))
    F(ḟ, 0.0, vec(copy(f)), params)
    for coords in (GridCoordinates(), ParityCoordinates())
        B = Matrix(qr(randn(Xoshiro(31), nx * nv, nx * nv)).Q)
        r = LinearReduction(coords, B, f0, nx, nv)
        field = galerkin_field(r, F, params)
        z = encode(r, f)
        ż = similar(z)
        field(ż, 0.0, z, nothing)
        expected = coords isa GridCoordinates ? B' * ḟ :
                   B' * parity_split(reshape(ḟ, nx, nv), zeros(nv))
        @test ż ≈ expected rtol=1e-12
        # and a truncated basis projects with Bᵀ
        r4 = LinearReduction(coords, B[:, 1:4], f0, nx, nv)
        z4 = encode(r4, f)
        ż4 = similar(z4)
        galerkin_field(r4, F, params)(ż4, 0.0, z4, nothing)
        g = similar(ḟ)
        F(g, 0.0, vec(decode(r4, z4)), params)
        e4 = coords isa GridCoordinates ? B[:, 1:4]' * g :
             B[:, 1:4]' * parity_split(reshape(g, nx, nv), zeros(nv))
        @test ż4 ≈ e4 rtol=1e-12

        # a Float64 state uses the field's buffers; the online cost of a reduced step depends
        # on it. Skipped where a run forces `--check-bounds=yes`.
        if Base.JLOptions().check_bounds != 1
            field4 = galerkin_field(r4, F, params)
            @test allocations(field4, ż4, z4) == 0
            @test allocations((ż, t, z, p) -> copy(z), ż4, z4) > 0
        end
    end
end

@testset "Reduced run" begin
    case = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 1.0)
    S = SHORT_RUN.snapshots
    f0 = background(S[:, :, 1])
    r = pod_reduction(S, f0, 8)
    run = run_reduced(r, case, 32, 64, S[:, :, 1])
    @test run.status == "ok"
    @test length(run.times) == 21
    @test size(run.snapshots) == (32, 64, 3)
    @test run.snapshots[:, :, 1] ≈ decode(r, encode(r, S[:, :, 1])) rtol=1e-14
    # the state moves
    @test run.snapshots[:, :, 2] != run.snapshots[:, :, 1]
    @test length(run.E1) == length(run.energy) == length(run.l1) == 21
    # the reduced run follows the full one over one time unit
    @test norm(run.snapshots[:, :, 3] - S[:, :, 3]) / norm(S[:, :, 3]) < 1e-3

    # a non-finite state stops the run at its step, and the run reports it
    stop = run_reduced(
        r, case, 32, 64, S[:, :, 1]; perturb = (s, z) -> s == 3 && (z[1] = NaN))
    @test stop.status == "failed at step 3"
    @test stop.failed_step == 3
    @test length(stop.states) == 3

    # a state scaled by 1e6 after step 1 is finite, and the Newton solve of step 2 gives up on it
    # (measured): the run stops on the solve, not on the state
    stop = run_reduced(
        r, case, 32, 64, S[:, :, 1]; perturb = (s, z) -> s == 1 && (z .*= 1e6))
    @test stop.status == "failed at step 2"
    @test all(isfinite, stop.states[end])

    # a state scaled by 1e200 after step 1 makes the Newton solve of step 2 raise a
    # `NonlinearSolverException` (measured): the run records the failed solve as its status
    for build in (pod_reduction, psd_reduction)
        stop = run_reduced(build(S, f0, 8), case, 32, 64, S[:, :, 1];
            perturb = (s, z) -> s == 1 && (z .*= 1e200))
        @test stop.status == "failed at step 2"
        @test stop.failed_step == 2
    end

    # any other exception of a step propagates
    broken = Ref(false)
    rb = LinearReduction(FailingCoordinates(broken), r.basis, f0, 32, 64)
    e = thrown(() -> run_reduced(rb, case, 32, 64, S[:, :, 1];
        perturb = (s, z) -> s == 1 && (broken[] = true)))
    @test e isa ArgumentError
    @test occursin(r"^the coordinates of this probe fail$", e.msg)
end
