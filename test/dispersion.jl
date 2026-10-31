using QuadGK: quadgk
using VlasovReductionBenchmark
using Test

include("helpers/thrown.jl")

@testset "Plasma dispersion function" begin
    # Z(0) = i√π, and on the real line Z(x) = -2 D(x) + i√π exp(-x²), with Dawson's integral
    # D(1) = 0.5380795069127684 (Abramowitz & Stegun, Table 7.5).
    @test plasma_dispersion(0.0) ≈ im * sqrt(π) rtol=1e-14
    @test plasma_dispersion(1.0) ≈ -2 * 0.5380795069127684 + im * sqrt(π) * exp(-1) rtol=1e-12

    # Z is analytic, with Z'(ζ) = -2 (1 + ζ Z(ζ)), on both sides of the real line.
    h = 1e-5
    for ζ in (0.7 + 0.4im, 1.0 - 0.1im, 1.2 - 0.6im)
        dZ = (plasma_dispersion(ζ + h) - plasma_dispersion(ζ - h)) / (2h)
        @test dZ ≈ -2 * (1 + ζ * plasma_dispersion(ζ)) rtol=1e-8
    end

    # Below the real line the Landau contour passes under the pole, so Z is the integral over
    # the real line plus the residue term 2i√π exp(-ζ²). The reflection conj(Z(conj ζ)) also
    # solves the identity above, and differs from Z by that term.
    for ζ in (1.0 - 0.1im, 1.2 - 0.6im)
        integral, _ = quadgk(s -> exp(-s^2) / (s - ζ), -Inf, Inf; rtol = 1e-13)
        @test plasma_dispersion(ζ) ≈ integral / sqrt(π) + 2im * sqrt(π) * exp(-ζ^2) rtol=1e-10
    end
end

@testset "Dielectric functions against quadrature" begin
    k = 0.5
    # Above the real line the Landau contour is the real line, so each closed form equals the
    # integral of f₀' / (v - ω/k) over it.
    for ω in (0.3im, 0.2 + 0.25im, 1.4 + 0.5im)
        @test dielectric_maxwellian(ω, k) ≈
              dielectric_quadrature(v -> -v * maxwellian(v), ω, k) rtol=1e-10
        @test dielectric_two_stream(ω, k) ≈
              dielectric_quadrature(two_stream_derivative, ω, k) rtol=1e-10
    end

    # the derivative of the two-stream profile, against a central difference
    for v in (-2.3, 0.4, 1.7)
        δ = 1e-5
        @test two_stream_derivative(v) ≈
              (two_stream_profile(v + δ) - two_stream_profile(v - δ)) / (2δ) rtol=1e-8
    end

    e = thrown(() -> dielectric_quadrature(two_stream_derivative, 1.0 - 0.1im, k))
    @test e isa ArgumentError
    @test occursin(
        r"^the quadrature over the real line is the Landau contour only for Im ω > 0; got ω = 1\.0 - 0\.1im$",
        e.msg)
    e = thrown(() -> dielectric_quadrature(two_stream_derivative, 1.0 + 0.0im, k))
    @test e isa ArgumentError
    @test occursin(
        r"^the quadrature over the real line is the Landau contour only for Im ω > 0; got ω = 1\.0 \+ 0\.0im$",
        e.msg)
end

@testset "Maxwellian root at k = 0.5" begin
    ω = landau_root(0.5)
    # the published root, 1.4156 - 0.1533i, to 1e-4 in each part
    @test abs(real(ω) - 1.4156) ≤ 1e-4
    @test abs(imag(ω) - (-0.1533)) ≤ 1e-4
    @test abs(dielectric_maxwellian(ω, 0.5)) ≤ 1e-12
end

@testset "Two-stream root at k = 0.5, two ways" begin
    ωZ = two_stream_root(0.5; method = :zfunction)
    ωQ = two_stream_root(0.5; method = :quadrature)
    @test imag(ωZ) > 0
    @test abs(ωZ - ωQ) ≤ 1e-6
    @test abs(dielectric_two_stream(ωZ, 0.5)) ≤ 1e-12
    @test abs(dielectric_quadrature(two_stream_derivative, ωQ, 0.5)) ≤ 1e-10

    e = thrown(() -> two_stream_root(0.5; method = :series))
    @test e isa ArgumentError
    @test occursin(r"^method is :zfunction or :quadrature; got :series$", e.msg)
end

@testset "Newton iteration" begin
    # a root of a polynomial, from a guess beside it
    @test dispersion_root(ω -> ω^2 + 1, 0.1 + 0.8im) ≈ im rtol=1e-12
    # from this guess, a stop at a step of 1e-4 in place of `tol` misses i by 1.4e-10
    @test abs(dispersion_root(ω -> ω^2 + 1, 0.3 + 1.5im) - im) ≤ 1e-14

    # a function without a root stops after the iteration cap
    e = thrown(() -> dispersion_root(exp, 1.0 + 0.0im; maxiter = 5))
    @test e isa ErrorException
    @test occursin(
        r"^Newton's iteration did not converge in 5 steps from ω = 1\.0 \+ 0\.0im$",
        e.msg)
end
