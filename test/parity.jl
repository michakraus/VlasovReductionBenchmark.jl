using Random: Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: mirror
using Test

include("helpers/thrown.jl")

@testset "Mirror node" begin
    # the v nodes of every grid are symmetric about 0, and node nv + 1 - j is the mirror of j
    for (nx, nv) in ((32, 64), (64, 128))
        _, v = grid(linear_landau(), nx, nv)
        @test [mirror(j, nv) for j in 1:nv] == nv:-1:1
        @test all(v[mirror(j, nv)] == -v[j] for j in 1:nv)
    end
end

@testset "Background" begin
    # the x-average of (1 + α cos kx) g(v) on a uniform periodic grid is g(v)
    case = linear_landau(0.3)
    x, v = grid(case, 32, 64)
    f = initial_condition(case, x, v)
    @test background(f) ≈ maxwellian.(v) rtol=1e-14
    @test background([1.0 2.0; 3.0 6.0]) == [2.0, 4.0]
end

@testset "Parity split" begin
    nx, nv = 6, 8
    h = nv ÷ 2
    f = randn(Xoshiro(1), nx, nv)
    f0 = randn(Xoshiro(2), nv)
    w = parity_split(f, f0)
    N = nx * h
    @test length(w) == 2N
    q, p = reshape(w[1:N], nx, h), reshape(w[(N + 1):end], nx, h)
    g = f .- f0'
    # the closed form: node h + l has v > 0, and its mirror is node h + 1 - l
    for l in 1:h, i in 1:nx

        @test q[i, l] == (g[i, h + l] - g[i, h + 1 - l]) / 2
        @test p[i, l] == (g[i, h + l] + g[i, h + 1 - l]) / 2
    end

    # the round trip on a random state
    for (nx, nv) in ((6, 8), (32, 64))
        f = randn(Xoshiro(3), nx, nv)
        f0 = randn(Xoshiro(4), nv)
        @test parity_join(parity_split(f, f0), f0) ≈ f rtol=1e-14
        @test size(parity_join(parity_split(f, f0), f0)) == (nx, nv)
    end

    # an even state has q = 0 exactly, an odd fluctuation has p = 0 exactly
    _, v = grid(linear_landau(), 32, 64)
    feven = randn(Xoshiro(5), 32) .* (v .^ 2)'
    w = parity_split(feven, zeros(64))
    @test all(iszero, w[1:(32 * 32)])
    @test any(!iszero, w[(32 * 32 + 1):end])
    f0 = exp.(-v .^ 2)
    fodd = f0' .+ randn(Xoshiro(6), 32) .* (v .^ 3)'
    # to the rounding of (f0 + g) - f0
    @test maximum(abs, parity_split(fodd, f0)[(32 * 32 + 1):end]) ≤
          8 * eps() * maximum(abs, fodd)

    # orthogonal up to the factor 1/√2
    f = randn(Xoshiro(7), nx, nv)
    @test sum(abs2, parity_split(f, zeros(nv))) ≈ sum(abs2, f) / 2 rtol=1e-14

    e = thrown(() -> parity_split(zeros(4, 5), zeros(5)))
    @test e isa ArgumentError
    @test occursin(r"^the parity split takes an even number of v nodes, got 5$", e.msg)
    e = thrown(() -> parity_split(zeros(4, 6), zeros(5)))
    @test e isa DimensionMismatch
    @test occursin(r"^the background has 5 values for 6 v nodes$", e.msg)
    e = thrown(() -> parity_join(zeros(25), zeros(6)))
    @test e isa DimensionMismatch
    @test occursin(r"^a parity vector of 25 values does not fit 6 v nodes$", e.msg)
    e = thrown(() -> parity_join(zeros(25), zeros(5)))
    @test e isa DimensionMismatch
    @test occursin(r"^a parity vector of 25 values does not fit 5 v nodes$", e.msg)
end
