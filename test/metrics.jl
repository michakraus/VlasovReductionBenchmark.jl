using VlasovReductionBenchmark
using VlasovReductionBenchmark: invariants
using Test

@testset "Relative L2 error" begin
    full = ones(2, 3, 2)
    reduced = copy(full)
    # snapshot 1 exact; snapshot 2 off by 1 at two of six nodes: sqrt(2 / 6)
    reduced[1, 1, 2] += 1
    reduced[2, 3, 2] -= 1
    e = relative_l2_errors(reduced, full)
    @test e[1] == 0
    @test e[2] ≈ sqrt(2 / 6) rtol=1e-15
    # 3 · full against full: an error of 2, not 4
    @test relative_l2_errors(3 .* full, full) ≈ [2.0, 2.0] rtol=1e-15
    # relative to the fluctuation around f0: f = f0 + 2 at every node, so ‖f - f0‖² = 6 · 4 and
    # the error of snapshot 2 is sqrt(2 / 24); relative to f it would be sqrt(2 / 100)
    f0 = [1.0, 2.0, 3.0]
    full = repeat(f0' .+ 2.0, 2, 1, 2)
    reduced = copy(full)
    reduced[1, 1, 2] += 1
    reduced[2, 3, 2] -= 1
    e = relative_l2_errors(reduced, full, f0)
    @test e[1] == 0
    @test e[2] ≈ sqrt(2 / 24) rtol=1e-15
    @test relative_l2_errors(reduced, full)[2] ≈ sqrt(2 / 100) rtol=1e-15
    e = try
        relative_l2_errors(ones(2, 3, 1), full)
    catch e
        e
    end
    @test e isa DimensionMismatch
    @test occursin(
        r"^the reduced snapshots \(2, 3, 1\) and the full ones \(2, 3, 2\) differ$", e.msg)
end

@testset "L1 norm and drifts" begin
    f = [1.0 -2.0; 0.0 3.0]
    @test l1_norm(f, 0.5, 0.25) == 0.125 * 6
    @test max_drift([2.0, 2.5, 1.0, 2.0]) == 0.5
    @test max_drift([-4.0, -3.0, -5.0]) == 0.25
    @test max_drift([1.0]) == 0
end

@testset "Invariants of a state" begin
    # the closed form of I1's diagnostics test, with the L1 norm beside it
    k = 0.5
    x, v = range(0, 4π; length = 5)[1:4], [0.0, 1.0]
    hx, hv = step(x), 1.0
    f = (1 .+ cos.(k .* x)) .* [1.0 2.0]
    G = [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    d = invariants(f, G, k, x, v, hx, hv)
    @test d.mass ≈ 12hx rtol=1e-14
    @test d.energy ≈ 31hx rtol=1e-14
    @test d.E1 ≈ 1.5 rtol=1e-14
    @test d.l1 ≈ 12hx rtol=1e-14
    @test d.l2 ≈ sqrt(30hx) rtol=1e-14
    @test d.entropy ≈ -14hx * log(2) rtol=1e-14
    @test d.nonpositive == 2
    # a negative node enters the L1 norm with its modulus, and not the mass
    # f = [-3 4; 1 2; 0 0; 1 2]
    f[1, 1] = -3.0
    d = invariants(f, G, k, x, v, hx, hv)
    @test d.l1 ≈ 13hx rtol=1e-14
    @test d.mass ≈ 7hx rtol=1e-14
    @test d.nonpositive == 3
end

@testset "Fitted rates" begin
    t = collect(0:0.05:40)
    γ, ω = -0.15, 1.4
    E1 = exp.(γ .* t) .* abs.(cos.(ω .* t))
    @test fitted_rate("linear_landau", t, E1, (0.0, 40.0)) ≈ γ rtol=1e-2
    # the window decides which maxima enter: after t = 25 the oscillation decays at 0.3
    E1w = ifelse.(t .≤ 25, E1, exp.(25γ .- 0.3 .* (t .- 25)) .* abs.(cos.(ω .* t)))
    @test fitted_rate("linear_landau", t, E1w, (0.0, 25.0)) ≈ γ rtol=1e-2
    @test fitted_rate("linear_landau", t, E1w, (0.0, 40.0)) < 1.1γ
    @test fitted_rate("two_stream", t, exp.(0.26 .* t), (0.0, 40.0)) ≈ 0.26 rtol=1e-12
    # the two-stream fit is over [5, 25] whatever the Landau window
    y = ifelse.(5 .≤ t .≤ 25, exp.(0.26 .* t), 1.0)
    @test fitted_rate("two_stream", t, y, (0.0, 40.0)) ≈ 0.26 rtol=1e-12
    @test fitted_rate("nonlinear_landau", t, E1, (0.0, 40.0)) === nothing
end

@testset "Trajectory metrics" begin
    # a reduced run equal to the full run but for one node, against the closed forms
    nx, nv = 4, 2
    full = ones(nx, nv, 3)
    reduced = copy(full)
    reduced[1, 1, 3] = 2.0
    series = (energy = [2.0, 2.2, 1.9], l1 = [1.0, 1.0, 1.5], l2 = [3.0, 3.0, 3.0],
        entropy = [-1.0, -1.1, -1.0], nonpositive = [0, 2, 1])
    # around f0 = 1/2 the fluctuation is 1/2 at each of the 8 nodes, ‖f - f0‖² = 2
    m = trajectory_metrics(reduced, full, series, [0.5, 0.5])
    @test m.l2_errors ≈ [0, 0, sqrt(1 / 8)] atol=1e-15
    @test m.l2_error_max ≈ sqrt(1 / 8) rtol=1e-15
    @test m.l2_errors_fluct ≈ [0, 0, sqrt(1 / 2)] atol=1e-15
    @test m.l2_error_fluct_max ≈ sqrt(1 / 2) rtol=1e-15
    # the largest error is the maximum over the snapshots, not the last one
    mid = copy(full)
    mid[1, 1, 2] = 2.0
    m = trajectory_metrics(mid, full, series, [0.5, 0.5])
    @test m.l2_error_max ≈ sqrt(1 / 8) rtol=1e-15
    @test m.l2_error_fluct_max ≈ sqrt(1 / 2) rtol=1e-15
    @test m.energy_drift ≈ 0.1 rtol=1e-12
    @test m.l1_drift ≈ 0.5 rtol=1e-12
    @test m.l2_drift == 0
    @test m.entropy_drift ≈ 0.1 rtol=1e-12
    @test m.nonpositive_max == 2
end
