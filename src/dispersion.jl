@doc raw"""
    plasma_dispersion(ζ)

The plasma dispersion function ``Z(ζ) = i\sqrt{π}\, w(ζ)``, with the Faddeeva function
``w(ζ) = e^{-ζ^2} \operatorname{erfc}(-iζ) = \operatorname{erfcx}(-iζ)``. It is the analytic
continuation of
```math
Z(ζ) = \frac{1}{\sqrt{π}} \int_{-∞}^{∞} \frac{e^{-s^2}}{s - ζ}\, ds , \qquad \operatorname{Im} ζ > 0 ,
```
into the whole plane, so below the real line it follows the Landau contour.
"""
plasma_dispersion(ζ) = im * sqrt(π) * erfcx(-im * ζ)

@doc raw"""
    dielectric_maxwellian(ω, k)

The dielectric function
``ε(k, ω) = 1 - k^{-2} \int f_0'(v) / (v - ω/k)\, dv`` of the [`maxwellian`](@ref) ``f_0``, on
the Landau contour:
```math
ε(k, ω) = 1 + \frac{1 + ζ Z(ζ)}{k^2} , \qquad ζ = \frac{ω}{\sqrt{2}\, k} .
```
"""
function dielectric_maxwellian(ω, k)
    ζ = ω / (sqrt(2) * k)

    return 1 + (1 + ζ * plasma_dispersion(ζ)) / k^2
end

@doc raw"""
    dielectric_two_stream(ω, k)

The dielectric function of [`two_stream_profile`](@ref) ``f_0 = v^2 g`` with ``g`` the
[`maxwellian`](@ref), on the Landau contour. With ``u = ω/k``, ``ζ = u/\sqrt{2}`` and
``f_0' = (2v - v^3)\, g``, the moments of ``g`` reduce the integral to
```math
\int \frac{f_0'(v)}{v - u}\, dv = 1 - u^2 + (2u - u^3) \frac{Z(ζ)}{\sqrt{2}} ,
\qquad ε(k, ω) = 1 - \frac{1}{k^2} \int \frac{f_0'(v)}{v - u}\, dv .
```
"""
function dielectric_two_stream(ω, k)
    u = ω / k
    integral = 1 - u^2 + (2u - u^3) * plasma_dispersion(u / sqrt(2)) / sqrt(2)

    return 1 - integral / k^2
end

"""
    two_stream_derivative(v)

The derivative ``(2v - v^3)\\, e^{-v^2/2} / \\sqrt{2π}`` of [`two_stream_profile`](@ref).
"""
two_stream_derivative(v) = (2v - v^3) * maxwellian(v)

@doc raw"""
    dielectric_quadrature(df0, ω, k)

The dielectric function ``ε(k, ω) = 1 - k^{-2} \int f_0'(v) / (v - ω/k)\, dv`` by adaptive
quadrature over the real line, with `df0` ``= f_0'``. The real line is the Landau contour only
above it, so `ω` must have ``\operatorname{Im} ω > 0``; otherwise it throws an `ArgumentError`.
"""
function dielectric_quadrature(df0, ω, k)
    imag(ω) > 0 || throw(ArgumentError(
        "the quadrature over the real line is the Landau contour only for Im ω > 0; got ω = $ω"))
    u = ω / k
    integral, _ = quadgk(v -> df0(v) / (v - u), -Inf, Inf; rtol = 1e-13)

    return 1 - integral / k^2
end

"""
    dispersion_root(ε, ω₀; tol = 1e-13, maxiter = 50)

The complex root of the analytic function `ε` by Newton's iteration from `ω₀`, with the
derivative by a central difference. It stops when a step is below `tol` relative to
``\\max(1, |ω|)``, and throws an error after `maxiter` steps.
"""
function dispersion_root(ε, ω₀; tol = 1e-13, maxiter = 50)
    ω = complex(float(ω₀))
    for _ in 1:maxiter
        h = 1e-6 * max(1, abs(ω))
        Δω = ε(ω) * 2h / (ε(ω + h) - ε(ω - h))
        ω -= Δω
        abs(Δω) ≤ tol * max(1, abs(ω)) && return ω
    end
    error("Newton's iteration did not converge in $maxiter steps from ω = $(complex(ω₀))")
end

"""
    landau_root(k)

The root ``ω(k)`` of [`dielectric_maxwellian`](@ref) by [`dispersion_root`](@ref) from
``\\sqrt{1 + 3k^2} - 0.1\\, i``, the Bohm–Gross frequency with a small damping. At ``k = 0.5`` it
is ``1.415662 - 0.153359\\, i``.
"""
landau_root(k) = dispersion_root(ω -> dielectric_maxwellian(ω, k), sqrt(1 + 3k^2) - 0.1im)

"""
    two_stream_root(k; method = :zfunction, guess = 0.25im)

The root ``ω(k)`` of the two-stream dielectric function from `guess`: through the plasma
dispersion function, [`dielectric_two_stream`](@ref), for `method = :zfunction`, and by
[`dielectric_quadrature`](@ref) of [`two_stream_derivative`](@ref) for `method = :quadrature`.
The quadrature holds for the growing root, ``\\operatorname{Im} ω > 0``.
"""
function two_stream_root(k; method::Symbol = :zfunction, guess = 0.25im)
    ε = if method === :zfunction
        ω -> dielectric_two_stream(ω, k)
    elseif method === :quadrature
        ω -> dielectric_quadrature(two_stream_derivative, ω, k)
    else
        throw(ArgumentError("method is :zfunction or :quadrature; got :$method"))
    end

    return dispersion_root(ε, guess)
end
