"""
    relative_l2_errors(reduced, full)

The relative ``L^2`` error ``\\|f_r - f\\|_2 / \\|f\\|_2`` of each of the `nx × nv × m` snapshots
`reduced` against the snapshots `full`, as nodal sums; the grid weights cancel.
"""
function relative_l2_errors(reduced::AbstractArray{<:Any, 3}, full::AbstractArray{<:Any, 3})
    size(reduced) == size(full) || throw(DimensionMismatch(
        "the reduced snapshots $(size(reduced)) and the full ones $(size(full)) differ"))
    return [sqrt(sum(abs2, view(reduced, :, :, s) .- view(full, :, :, s)) /
                 sum(abs2, view(full, :, :, s)))
            for s in axes(full, 3)]
end

"""
    l1_norm(f, hx, hv)

The ``L^1`` norm ``h_x h_v \\sum |f|`` of the nodal values `f`.
"""
l1_norm(f, hx, hv) = hx * hv * sum(abs, f)

"""
    max_drift(series)

The largest relative drift ``\\max_t |s(t) - s(0)| / |s(0)|`` of `series`.
"""
max_drift(series::AbstractVector) = maximum(abs, (series .- series[1]) ./ abs(series[1]))

"""
    invariants(f, G, k, x, v, hx, hv)

The [`diagnostics`](@ref) of the `nx × nv` state `f`, with its [`l1_norm`](@ref) as `l1`.
"""
function invariants(f::AbstractMatrix, G, k, x, v, hx, hv)
    return merge(diagnostics(vec(f), G, k, x, v, hx, hv), (l1 = l1_norm(f, hx, hv),))
end

"""
    fitted_rate(name, t, E1, landau_window)

The rate of the case `name` fitted to `log.(E1)` at the times `t`: for linear Landau damping the
damping rate of [`landau_rate_fit`](@ref) over `landau_window`, for the two-stream instability the
growth rate of [`growth_rate_fit`](@ref) over ``[5, 25]``, and `nothing` for any other case.
"""
function fitted_rate(name::AbstractString, t, E1, landau_window)
    name == "linear_landau" && return landau_rate_fit(t, log.(E1), landau_window).γ
    name == "two_stream" && return growth_rate_fit(t, log.(E1), (5.0, 25.0))
    return nothing
end

"""
    trajectory_metrics(reduced, full, series)

The metrics of a reduced run against its full-order run: the [`relative_l2_errors`](@ref) of
the decoded snapshots `reduced` against `full` and their maximum, the [`max_drift`](@ref) of the
`energy`, `l1`, `l2` and `entropy` of `series`, and the largest count of nodes where
``f ≤ 0``.
"""
function trajectory_metrics(reduced, full, series)
    errors = relative_l2_errors(reduced, full)
    return (l2_errors = errors, l2_error_max = maximum(errors),
        energy_drift = max_drift(series.energy), l1_drift = max_drift(series.l1),
        l2_drift = max_drift(series.l2), entropy_drift = max_drift(series.entropy),
        nonpositive_max = maximum(series.nonpositive))
end
