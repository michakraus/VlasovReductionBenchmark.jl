"""
    VlasovCase(name, profile, α, k, tend)

A 1D1V Vlasov–Poisson test case: the initial condition
``f(x, v) = (1 + α \\cos kx)\\, g(v)`` with the velocity profile `profile` ``= g``, on the
periodic interval ``x ∈ [0, 2π/k)``, run over ``t ∈ [0, t_\\mathrm{end}]``. `name` labels its run
files. [`linear_landau`](@ref), [`nonlinear_landau`](@ref) and [`two_stream`](@ref) build the
three cases of the benchmark.
"""
struct VlasovCase{F}
    name::String
    profile::F
    α::Float64
    k::Float64
    tend::Float64
end

"""
    maxwellian(v)

The Maxwellian ``e^{-v^2/2} / \\sqrt{2π}`` of unit density and unit thermal velocity.
"""
maxwellian(v) = exp(-v^2 / 2) / sqrt(2π)

"""
    two_stream_profile(v)

The two-stream profile ``v^2 e^{-v^2/2} / \\sqrt{2π}`` of unit density.
"""
two_stream_profile(v) = v^2 * maxwellian(v)

"""
    linear_landau(α = 0.01)

Linear Landau damping: the Maxwellian with ``k = 0.5``, run to ``t = 40``.
"""
linear_landau(α = 0.01) = VlasovCase("linear_landau", maxwellian, α, 0.5, 40.0)

"""
    nonlinear_landau(α = 0.5)

Nonlinear Landau damping: the Maxwellian with ``k = 0.5``, run to ``t = 60``.
"""
nonlinear_landau(α = 0.5) = VlasovCase("nonlinear_landau", maxwellian, α, 0.5, 60.0)

"""
    two_stream(α = 0.05)

The two-stream instability: [`two_stream_profile`](@ref) with ``k = 0.5``, run to ``t = 60``.
"""
two_stream(α = 0.05) = VlasovCase("two_stream", two_stream_profile, α, 0.5, 60.0)

"""
    grid(case, nx, nv)

The nodes `(x, v)` of the `nx × nv` grid of `case`: `x` the `nx` periodic nodes of
``[0, 2π/k)``, from PoissonSolvers' `FFTWBasis`, and `v` the `nv` nodes of ``[-6, 6]`` with both
ends.
"""
function grid(case::VlasovCase, nx::Integer, nv::Integer)
    x = FFTWBasis((0.0, 2π / case.k), nx).xgrid[1:nx]
    v = range(VDOMAIN[1], VDOMAIN[2]; length = nv)

    return x, v
end

"""
    initial_condition(case, x, v)

The `length(x) × length(v)` matrix of the initial condition of `case` at the nodes `x` and `v`.
"""
function initial_condition(case::VlasovCase, x, v)
    return (1 .+ case.α .* cos.(case.k .* x)) .* case.profile.(v')
end

"""
    vlasov_model(case, nx, nv)

VlasovMethods' grid Vlasov–Poisson model of `case` on the `nx × nv` grid of [`grid`](@ref), with
the FFT Poisson solver on the same x-nodes.
"""
function vlasov_model(case::VlasovCase, nx::Integer, nv::Integer)
    x, v = grid(case, nx, nv)
    L = 2π / case.k
    dist = GridDistribution(initial_condition(case, x, v), (0.0, L), VDOMAIN)

    return VlasovPoisson(dist, Potential(FFTWBasis((0.0, L), nx)))
end
