using LinearAlgebra: Diagonal, I, cond, diag, norm, qr, svd
using QuadGK: quadgk
using VlasovReductionBenchmark
using VlasovReductionBenchmark: LinearReduction, ParityCoordinates, VDOMAIN
using Test

include("helpers/thrown.jl")

# The two-stream fixture: 40 snapshots, t = 0, 0.5, …, 19.5, of the two-stream instability on
# the smoke grid.
const CASE = VlasovCase("two_stream", two_stream_profile, 0.05, 0.5, 19.5)
const NX, NV = 32, 64
const TS = run_case(CASE, NX, NV).snapshots
const F0 = background(TS[:, :, 1])

J(h) = [zeros(h, h) I(h); -I(h) zeros(h, h)]

# The physicists' Hermite function h_n, orthonormal on the real line.
function hermite_function(n, v)
    H0, H1 = 1.0, 2v
    n == 0 && (H1 = H0)
    for m in 1:(n - 1)
        H0, H1 = H1, 2v * H1 - 2m * H0
    end
    return H1 * exp(-v^2 / 2) / sqrt(2.0^n * factorial(n) * sqrt(π))
end

# The latent pairing of the six Fourier–Hermite product modes `a(kx) h_b(v)`,
# `a ∈ (cos, sin)`, `b ∈ (1, 3, 5)`, on the `nx × nv` grid of `[0, 2π/k)` with the Maxwellian
# background, and its reference: `-(∫ a' c dx)(∫ f0' h_b sign h_e dv)` times the square of the
# scale of the functionals. A `q` column of the basis `[Φ 0; 0 Φ]` gives the functional
# `parity_join([φ; 0]) / (2 h_x h_v)`, the odd extension of `φ`; a `p` column the even one.
function fourier_hermite_errors(nx, nv; k = 1.0)
    case = VlasovCase("maxwellian", maxwellian, 0.0, k, 1.0)
    x, v = grid(case, nx, nv)
    vp = v[(nv ÷ 2 + 1):nv]
    hx, hv = step(x), step(v)
    modes = [(a, b) for b in (1, 3, 5) for a in (cos, sin)]
    M = reduce(hcat, [vec(a.(k .* x) .* hermite_function.(b, vp)') for (a, b) in modes])
    F = qr(M)
    Φ = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    r = LinearReduction(ParityCoordinates(), [Φ zero(Φ); zero(Φ) Φ], maxwellian.(v), nx, nv)
    Π = latent_pairing(r, k)

    # the discrete norm of a mode on the half grid is (nx/2) ∫_0^∞ h_b² dv / h_v = nx / (4 h_v)
    scale = 1 / (2 * sqrt(nx / (4hv)) * hx * hv)
    df0(w) = -w * maxwellian(w)
    xint(a, c) = a === c ? 0.0 : (a === cos ? -π : π)    # ∫_0^{2π/k} ∂ₓa(kx) c(kx) dx
    vint(b, e) = quadgk(
        w -> df0(w) * hermite_function(b, w) * sign(w) *
             hermite_function(e, w),
        VDOMAIN[1], 0.0, VDOMAIN[2];
        rtol = 1e-13)[1]
    reference = [-xint(a, c) * vint(b, e) * scale^2 for (a, b) in modes, (c, e) in modes]
    m = length(modes)
    return maximum(abs, Π[1:m, (m + 1):(2m)] - reference), maximum(abs, reference)
end

@testset "Fourier–Hermite pairing" begin
    e256, ref256 = fourier_hermite_errors(32, 256)
    e512, ref512 = fourier_hermite_errors(32, 512)
    @test e256 ≤ 2e-3 * ref256
    @test e512 / ref512 ≤ 0.3 * (e256 / ref256)
    # the scale of the functionals and of D_x on a period other than 2π
    e, ref = fourier_hermite_errors(32, 256; k = 0.5)
    @test e ≤ 2e-3 * ref
end

@testset "the derivative of the background" begin
    # both stencils are of second order, so they are exact on a quadratic, the end nodes included
    v = range(VDOMAIN[1], VDOMAIN[2]; length = 16)
    d = VlasovReductionBenchmark.background_derivative(v .^ 2 .+ v, step(v))
    @test d ≈ 2v .+ 1 rtol=1e-12
end

@testset "antisymmetry and parity blocks" begin
    r = psd_reduction(TS, F0, 8)
    Π = latent_pairing(r, CASE.k)
    @test size(Π) == (8, 8)
    @test norm(Π) > 0
    @test maximum(abs, Π + Π') ≤ 1e-12 * norm(Π)
    @test maximum(abs, Π[1:4, 1:4]) ≤ 1e-12 * norm(Π)
    @test maximum(abs, Π[5:8, 5:8]) ≤ 1e-12 * norm(Π)
end

@testset "zero x-average" begin
    r = psd_nonzero_reduction(TS, F0, 8)
    @test r isa LinearReduction{ParityCoordinates}
    Φ = r.basis[1:(NX * NV ÷ 2), 1:4]
    @test r.basis == [Φ zero(Φ); zero(Φ) Φ]
    @test Φ' * Φ ≈ I(4) atol=1e-12
    @test maximum(abs, sum(reshape(Φ, NX, NV ÷ 2, 4); dims = 1)) ≤ 1e-12
    # the leading left singular vectors of the parity snapshots without their x-average
    W = parity_snapshots(TS, F0)
    W3 = reshape(W, NX, NV, :)
    W0 = reshape(W3 .- sum(W3; dims = 1) ./ NX, NX * NV, :)
    N = NX * NV ÷ 2
    U = svd([W0[1:N, :] W0[(N + 1):end, :]]).U[:, 1:4]
    @test norm(U - Φ * (Φ' * U)) < 1e-10
end

@testset "no k = 0 content" begin
    v = grid(CASE, NX, NV)[2]
    S = reduce((A, B) -> cat(A, B; dims = 3),
        [repeat(((1 + 0.1s) .* two_stream_profile.(v))', NX) for s in 1:5])
    @test norm(parity_snapshots(S, background(S[:, :, 1]))) > 0
    e = thrown(() -> psd_nonzero_reduction(S, background(S[:, :, 1]), 4))
    @test e isa ArgumentError
    @test occursin("k = 0", sprint(showerror, e))
    e = thrown(() -> dual_psd_reduction(S, background(S[:, :, 1]), 4, CASE.k))
    @test e isa ArgumentError
    # snapshots equal to the background: no content at all
    S = repeat(reshape(F0, 1, NV, 1); outer = (NX, 1, 3))
    @test_throws ArgumentError psd_nonzero_reduction(S, F0, 4)
end

@testset "canonical latent pairing" begin
    for n in (4, 8)
        r = dual_psd_reduction(TS, F0, n, CASE.k)
        @test r isa DualReduction
        κ = pairing_condition(r)
        h = n ÷ 2
        @test κ ≈ cond(latent_pairing(r.r0, CASE.k)[1:h, (h + 1):n]) rtol=1e-12
        Π = latent_pairing(r, CASE.k)
        @test maximum(abs, Π - J(n ÷ 2)) ≤ 1e-10 * κ
    end
end

@testset "the same reconstruction as PSD without the x-average" begin
    for n in (4, 8)
        r = dual_psd_reduction(TS, F0, n, CASE.k)
        r0 = psd_nonzero_reduction(TS, F0, n)
        @test r.r0.basis == r0.basis
        for s in axes(TS, 3)
            f = TS[:, :, s]
            z, z0 = encode(r, f), encode(r0, f)
            @test length(z) == n
            @test z[(n ÷ 2 + 1):n] == z0[(n ÷ 2 + 1):n]
            @test norm(decode(r, z) - decode(r0, z0)) ≤ 1e-12 * norm(f .- F0')
        end
    end
end

@testset "a singular pairing" begin
    # The snapshots span the four even modes cos(kx) α_i(v) and sin(kx) β_j(v), so the odd–even
    # block of the pairing is π [0 -K; Kᵀ 0] up to a change of basis, with
    # K_ij = 2 Σ_{v_l > 0} f0'(v_l) α_i(v_l) β_j(v_l) (the common factors aside). β_2 depends on
    # t, det K is linear in t and zero at t*, and t = t* (1 + δ) sweeps the condition number
    # across the limit 1 / (100 eps).
    x, v = grid(CASE, NX, NV)
    hv = step(v)
    p = maxwellian.(v)
    d = [(-3p[1] + 4p[2] - p[3]) / (2hv);
         [(p[j + 1] - p[j - 1]) / (2hv) for j in 2:(NV - 1)];
         (3p[NV] - 4p[NV - 1] + p[NV - 2]) / (2hv)]
    vp = (NV ÷ 2 + 1):NV
    K(α, β) = sum(d[vp] .* α[vp] .* β[vp])
    α1, α2, β1, β2a, β2b = p, v .^ 2 .* p, p, v .^ 2 .* p, v .^ 4 .* p
    detK(t) = K(α1, β1) * K(α2, β2a .+ t .* β2b) - K(α1, β2a .+ t .* β2b) * K(α2, β1)
    tstar = -detK(0.0) / (detK(1.0) - detK(0.0))
    c, s_ = cos.(CASE.k .* x), sin.(CASE.k .* x)
    limit = 1 / (100eps())
    coefficients(s) = (0.01s, 0.003s^2, 0.002s^3, 0.001s^4)
    over, under = 0, 0
    for δ in [10.0 .^ (-8:-1:-16); 0.0]
        t = tstar * (1 + δ)
        modes = (c .* α1', c .* α2', s_ .* β1', s_ .* (β2a .+ t .* β2b)')
        S = reduce((A, B) -> cat(A, B; dims = 3),
            [p' .+ sum(coefficients(s) .* modes) for s in 0:6])
        f0 = background(S[:, :, 1])
        Π = latent_pairing(psd_nonzero_reduction(S, f0, 8), CASE.k)
        κ = cond(Π[1:4, 5:8])
        e = thrown(() -> dual_psd_reduction(S, f0, 8, CASE.k))
        if κ > limit
            over += 1
            @test e isa ArgumentError
            @test occursin("condition number $κ", sprint(showerror, e))
        else
            under += 1
            @test e === nothing
        end
    end
    @test over ≥ 2 && under ≥ 2

    # every mode holds cos(kx) and none sin(kx): the odd–even block is round-off, and at n = 4
    # it is 2 × 2 and antisymmetric, so its condition number is small
    cosine_modes = (c .* p', c .* (v .^ 2 .* p)', c .* (v .^ 4 .* p)', c .* (v .^ 6 .* p)')
    S = reduce((A, B) -> cat(A, B; dims = 3),
        [p' .+ sum(coefficients(s) .* cosine_modes) for s in 0:6])
    e = thrown(() -> dual_psd_reduction(S, background(S[:, :, 1]), 4, CASE.k))
    @test e isa ArgumentError
    msg = sprint(showerror, e)
    @test occursin("round-off", msg) && occursin("condition number", msg)
    # a sin(kx) part of relative size ε in the first mode gives a block of norm about
    # 2.5e-3 ε times the scale: ε = 1e-8 lies below the limit 1e-10, ε = 1e-6 above it
    for (ε, below) in ((1e-8, true), (1e-6, false))
        modes = ((c .+ ε .* s_) .* p', cosine_modes[2:4]...)
        S = reduce((A, B) -> cat(A, B; dims = 3),
            [p' .+ sum(coefficients(s) .* modes) for s in 0:6])
        e = thrown(() -> dual_psd_reduction(S, background(S[:, :, 1]), 4, CASE.k))
        @test below ? occursin("round-off", sprint(showerror, e)) : e === nothing
    end

    # the odd–even block of a PSD basis is antisymmetric, so for an odd n/2 it is singular
    Π = latent_pairing(psd_nonzero_reduction(TS, F0, 6), CASE.k)
    @test Π[1:3, 4:6] ≈ -transpose(Π[1:3, 4:6]) atol=1e-12 * norm(Π)
    @test_throws ArgumentError dual_psd_reduction(TS, F0, 6, CASE.k)
end

@testset "a background that is not even" begin
    v = grid(CASE, NX, NV)[2]
    f0 = maxwellian.(v .- 0.5)
    @test norm(f0 - reverse(f0)) > 1e-12 * norm(f0)
    @test_throws ArgumentError dual_psd_reduction(TS, f0, 4, CASE.k)
    B = psd_reduction(TS, F0, 4).basis
    r = LinearReduction(ParityCoordinates(), B, f0, NX, NV)
    @test_throws ArgumentError latent_pairing(r, CASE.k)
    # an asymmetry of 1e-14 ‖f0‖ passes the limit 1e-12 ‖f0‖, and one of 1e-10 ‖f0‖ does not
    g0 = F0 .* (1 .+ 1e-14 .* (1:NV) ./ NV)
    r = LinearReduction(ParityCoordinates(), B, g0, NX, NV)
    @test latent_pairing(r, CASE.k) isa Matrix{Float64}
    g0 = F0 .* (1 .+ 1e-10 .* (1:NV) ./ NV)
    r = LinearReduction(ParityCoordinates(), B, g0, NX, NV)
    @test_throws ArgumentError latent_pairing(r, CASE.k)
    # a zero background is even, and its pairing is zero
    r = LinearReduction(ParityCoordinates(), B, zeros(NV), NX, NV)
    @test latent_pairing(r, CASE.k) == zeros(4, 4)
end

@testset "the other edges" begin
    @test_throws ArgumentError dual_psd_reduction(TS, F0, 5, CASE.k)
    @test_throws ArgumentError psd_nonzero_reduction(TS, F0, 5)
    @test_throws ArgumentError dual_psd_reduction(TS, F0, 4, 0.0)
    @test_throws ArgumentError dual_psd_reduction(TS, F0, 4, -0.5)
    @test_throws ArgumentError latent_pairing(psd_reduction(TS, F0, 4), 0.0)
    for bad in (NaN, Inf, -Inf)
        S = copy(TS)
        S[3, 4, 2] = bad
        @test_throws ArgumentError dual_psd_reduction(S, F0, 4, CASE.k)
        e = thrown(() -> psd_nonzero_reduction(S, F0, 4))
        @test e isa ArgumentError
        @test occursin("non-finite", sprint(showerror, e))
    end
end
