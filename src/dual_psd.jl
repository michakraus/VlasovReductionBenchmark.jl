"""
    psd_nonzero_reduction(S, f0, n)

PSD without the x-average: the [`LinearReduction`](@ref) in parity coordinates with the basis
``\\operatorname{diag}(Φ, Φ)`` of `n` columns, where ``Φ`` is the [`psd_basis`](@ref) of the
parity snapshots of the `nx × nv × m` snapshots `S` around `f0` after the x-average of each of
their `nv` parity rows is removed. Every column of ``Φ`` then has a zero x-average in each of
its `nv/2` velocity rows: the basis holds no ``k = 0`` content.

It throws an `ArgumentError` for an odd `n`, for a snapshot with a non-finite value, and when
the parity snapshots without their x-average have at most `1e-12` times the norm of those with
it, as for snapshots with no content beyond ``k = 0``.
"""
function psd_nonzero_reduction(S::AbstractArray{<:Any, 3}, f0::AbstractVector, n::Integer)
    nx, nv, _ = size(S)
    iseven(n) || throw(ArgumentError("PSD takes an even number of modes, got $n"))
    all(isfinite, S) || throw(ArgumentError("the snapshots hold a non-finite value"))
    W = parity_snapshots(S, f0)
    W3 = reshape(W, nx, nv, :)
    W0 = reshape(W3 .- sum(W3; dims = 1) ./ nx, nx * nv, :)
    norm(W0) > 1e-12 * norm(W) || throw(ArgumentError(
        "the snapshots hold only k = 0 content, which PSD without the x-average removes"))
    Φ = psd_basis(W0, n)
    return LinearReduction(ParityCoordinates(), [Φ zero(Φ); zero(Φ) Φ], f0, nx, nv)
end

"""
    background_derivative(f0, hv)

The derivative ``f_0'`` of the background `f0` on nodes of spacing `hv`: the second-order
central difference at the interior nodes, and the one-sided difference of second order at the
two end nodes.
"""
function background_derivative(f0::AbstractVector, hv::Real)
    g, nv = f0, length(f0)
    d = similar(g, Float64)
    d[1] = (-3 * g[1] + 4 * g[2] - g[3]) / (2hv)
    for j in 2:(nv - 1)
        d[j] = (g[j + 1] - g[j - 1]) / (2hv)
    end
    d[nv] = (3 * g[nv] - 4 * g[nv - 1] + g[nv - 2]) / (2hv)
    return d
end

@doc raw"""
    latent_pairing(r, k)

The `n × n` matrix ``Π_{ab} = Π(m_a, m_b)`` of the discrete Vlasov bracket, linearised at the
background `r.f0`, on the latent coordinate functionals of the parity reduction `r` on the
`nx × nv` grid of [`grid`](@ref) whose x domain is ``[0, 2π/k)``.

The functional ``m_a`` is the grid function with
``z_a = h_x h_v \sum_{ij} m_a(x_i, v_j)\, δf_{ij}`` for every fluctuation ``δf``, with
``h_x = 2π/(k n_x)`` and ``h_v = 12/(n_v - 1)``. Since
``z = B^{\mathsf T} C\, δf`` with ``C`` the parity split, ``m_a`` is
[`parity_join`](@ref) of column ``a`` of ``B``, divided by ``2 h_x h_v``. The pairing is
```math
Π(m, m') = -h_x h_v \sum_{ij} f_0'(v_j)\, (D_x m)_{ij}\, m'_{ij},
```
the bracket ``\int\!\!\int f_0 \{m, m'\}`` after an integration by parts, with ``D_x`` the
[`spectral_derivative`](@ref) and ``f_0'`` from [`background_derivative`](@ref). It is
antisymmetric, and for an even background its odd–odd and even–even blocks vanish.

It throws an `ArgumentError` for `k ≤ 0` and for a background `f0` that differs from
`reverse(f0)` by more than `1e-12 ‖f0‖`.
"""
function latent_pairing(r::LinearReduction{ParityCoordinates}, k::Real)
    k > 0 || throw(ArgumentError("the wave number must be positive, got $k"))
    norm(r.f0 - reverse(r.f0)) ≤ 1e-12 * norm(r.f0) ||
        throw(ArgumentError("the background is not even in v"))
    nx, nv = r.nx, r.nv
    hx, hv = 2π / (k * nx), (VDOMAIN[2] - VDOMAIN[1]) / (nv - 1)
    D = spectral_derivative(nx, k)
    df0 = background_derivative(r.f0, hv)
    # the functionals times h_x h_v, one per column, x fastest
    M = reduce(hcat, [parity_join!(similar(b), b, nx, nv) ./ 2 for b in eachcol(r.basis)])
    DM = reshape(D * reshape(M, nx, :), nx * nv, :)
    return -(DM' * (repeat(df0; inner = nx) .* M)) ./ (hx * hv)
end

"""
    DualReduction

Dual PSD: the reduction [`psd_nonzero_reduction`](@ref) `r0` with its odd (`q`) latent block
mapped by the invertible matrix `C`, so that ``z = [C z_{0,q}; z_{0,p}]`` with ``z_0`` the
latent state of `r0`. Build it with [`dual_psd_reduction`](@ref).
"""
struct DualReduction{RT <: LinearReduction{ParityCoordinates}, CT <: AbstractMatrix}
    r0::RT
    C::CT
end

"""
    dual_psd_reduction(S, f0, n, k)

The [`DualReduction`](@ref) of the reduction `r0 = `[`psd_nonzero_reduction`](@ref)`(S, f0, n)`
whose matrix ``C`` is the inverse of the odd–even block of [`latent_pairing`](@ref)`(r0, k)`:
the latent pairing of the result is canonical, ``[0\\; I; -I\\; 0]``, with `q` the odd block.

It throws an `ArgumentError` where `psd_nonzero_reduction` or `latent_pairing` does, and when
the 2-norm condition number of the odd–even block exceeds `1 / (100 eps())`.
"""
function dual_psd_reduction(S::AbstractArray{<:Any, 3}, f0::AbstractVector, n::Integer,
        k::Real)
    r0 = psd_nonzero_reduction(S, f0, n)
    h = n ÷ 2
    B = latent_pairing(r0, k)[1:h, (h + 1):n]
    κ = cond(B)
    κ ≤ 1 / (100 * eps(Float64)) || throw(ArgumentError(
        "the odd–even block of the latent pairing has the condition number $κ"))
    return DualReduction(r0, inv(B))
end

"""
    pairing_condition(r)

The 2-norm condition number of the odd–even block of the latent pairing of `r.r0` for the
[`DualReduction`](@ref) `r`, the inverse of its matrix `C`.
"""
pairing_condition(r::DualReduction) = cond(r.C)

"""
    encode(r::DualReduction, f)

The latent state ``[C z_q; z_p]`` of the `nx × nv` state `f`, with ``[z_q; z_p]`` its latent
state under `r.r0`.
"""
function encode(r::DualReduction, f::AbstractMatrix)
    z = encode(r.r0, f)
    h = length(z) ÷ 2
    return [r.C * z[1:h]; z[(h + 1):end]]
end

"""
    decode(r::DualReduction, z)

The `nx × nv` state of the latent state ``z = [z_q; z_p]``: the state of
``[C^{-1} z_q; z_p]`` under `r.r0`.
"""
function decode(r::DualReduction, z::AbstractVector)
    h = length(z) ÷ 2
    return decode(r.r0, [r.C \ z[1:h]; z[(h + 1):end]])
end

"""
    latent_pairing(r::DualReduction, k)

The latent pairing ``T Π_0 T^{\\mathsf T}`` of the [`DualReduction`](@ref) `r`, with ``Π_0``
the latent pairing of `r.r0` and ``T = \\operatorname{diag}(C, I)`` the map of its latent states.
"""
function latent_pairing(r::DualReduction, k::Real)
    Π0 = latent_pairing(r.r0, k)
    h = size(Π0, 1) ÷ 2
    T = [r.C zeros(h, h); zeros(h, h) I(h)]
    return T * Π0 * T'
end
