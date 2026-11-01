"""
    mirror(j, nv)

The velocity node ``v_{n_v + 1 - j} = -v_j`` that mirrors node `j` of the `nv` nodes of
[`grid`](@ref), which are symmetric about ``v = 0``.
"""
mirror(j::Integer, nv::Integer) = nv + 1 - j

"""
    background(f)

The spatial average ``f_0(v_j) = \\frac{1}{n_x} \\sum_i f_{ij}`` of the `nx × nv` state `f`, the
background around which every reduced model reduces the fluctuation.
"""
background(f::AbstractMatrix) = vec(sum(f; dims = 1)) ./ size(f, 1)

"""
    fluctuation(f, f0)

The fluctuation ``f - f_0`` of the `nx × nv` state `f`, or of every state of the
`nx × nv × m` array `f`, around the background `f0` of `nv` values.
"""
fluctuation(f::AbstractArray, f0::AbstractVector) = f .- reshape(f0, 1, :)

"""
    parity_split!(w, g, nx, nv)

Write the parity vector ``w = [q; p]`` of the fluctuation `g`, a vector of the ``n_x n_v`` nodal
values in x-fastest order: on the node ``(i, l)``, ``l = 1, …, n_v/2``, with ``v_{n_v/2 + l} > 0``,
``q = (g(v) - g(-v))/2`` and ``p = (g(v) + g(-v))/2``, each ``n_x × n_v/2`` in x-fastest order.
The map is linear and ``1/\\sqrt 2`` times an orthogonal map.
"""
function parity_split!(w::AbstractVector, g::AbstractVector, nx::Integer, nv::Integer)
    h = nv ÷ 2
    N = nx * h
    for l in 1:h
        jp = h + l
        jm = mirror(jp, nv)
        for i in 1:nx
            a, b = g[(jp - 1) * nx + i], g[(jm - 1) * nx + i]
            w[(l - 1) * nx + i] = (a - b) / 2
            w[N + (l - 1) * nx + i] = (a + b) / 2
        end
    end
    return w
end

"""
    parity_join!(g, w, nx, nv)

The inverse of [`parity_split!`](@ref): write the fluctuation ``g(v) = p + q``,
``g(-v) = p - q`` of the parity vector `w` into the ``n_x n_v`` values of `g`.
"""
function parity_join!(g::AbstractVector, w::AbstractVector, nx::Integer, nv::Integer)
    h = nv ÷ 2
    N = nx * h
    for l in 1:h
        jp = h + l
        jm = mirror(jp, nv)
        for i in 1:nx
            q, p = w[(l - 1) * nx + i], w[N + (l - 1) * nx + i]
            g[(jp - 1) * nx + i] = p + q
            g[(jm - 1) * nx + i] = p - q
        end
    end
    return g
end

"""
    parity_split(f, f0)

The canonical vector ``[q; p]`` of the `nx × nv` state `f` around the background `f0`: `q` the
odd and `p` the even part of ``f - f_0`` in ``v``, each on the ``n_v/2`` nodes with ``v > 0``,
``N = n_x n_v / 2`` values each (see [`parity_split!`](@ref)). It throws an `ArgumentError` for an
odd `nv` and a `DimensionMismatch` when `f0` does not hold `nv` values.
"""
function parity_split(f::AbstractMatrix, f0::AbstractVector)
    nx, nv = size(f)
    iseven(nv) ||
        throw(ArgumentError("the parity split takes an even number of v nodes, got $nv"))
    length(f0) == nv ||
        throw(DimensionMismatch("the background has $(length(f0)) values for $nv v nodes"))
    g = vec(fluctuation(f, f0))
    return parity_split!(similar(g), g, nx, nv)
end

"""
    parity_join(w, f0)

The `nx × nv` state ``f_0 + g`` of the parity vector `w` of [`parity_split`](@ref) and the
background `f0` of `nv` values, with ``n_x = `` `length(w) ÷ nv`. It throws a `DimensionMismatch`
when `nv` is zero or odd or `length(w)` is no multiple of `nv`.
"""
function parity_join(w::AbstractVector, f0::AbstractVector)
    nv = length(f0)
    (nv > 0 && iseven(nv) && length(w) % nv == 0) || throw(DimensionMismatch(
        "a parity vector of $(length(w)) values does not fit $nv v nodes"))
    nx = length(w) ÷ nv
    g = parity_join!(similar(w), w, nx, nv)
    return reshape(g, nx, nv) .+ reshape(f0, 1, :)
end

"""
    parity_snapshots(S, f0)

The `nx nv × m` matrix whose column `s` is [`parity_split`](@ref)`(S[:, :, s], f0)`, the
vertically stacked ``[Q; P]`` of the `nx × nv × m` snapshots `S`.
"""
function parity_snapshots(S::AbstractArray{<:Any, 3}, f0::AbstractVector)
    return reduce(hcat, [parity_split(view(S, :, :, s), f0) for s in axes(S, 3)])
end
