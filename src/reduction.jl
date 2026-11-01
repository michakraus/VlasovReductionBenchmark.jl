"The coordinates of a fluctuation in which a [`LinearReduction`](@ref) holds its basis."
abstract type Coordinates end

"The ``n_x n_v`` nodal values of the fluctuation, in x-fastest order."
struct GridCoordinates <: Coordinates end

"The parity vector ``[q; p]`` of the fluctuation, as [`parity_split`](@ref) gives it."
struct ParityCoordinates <: Coordinates end

"""
    LinearReduction(coordinates, basis, f0, nx, nv)

The linear reduction ``f = f_0 + C^{-1} B z`` of the `nx × nv` state `f`, with the background
`f0` of `nv` values, the `basis` ``B`` of ``n_x n_v`` rows and the map ``C`` from the
fluctuation to the `coordinates`: the identity for [`GridCoordinates`](@ref), the parity split
for [`ParityCoordinates`](@ref). It throws a `DimensionMismatch` when the basis or the
background does not fit the grid.
"""
struct LinearReduction{C <: Coordinates, BT <: AbstractMatrix, OT <: AbstractVector}
    coordinates::C
    basis::BT
    f0::OT
    nx::Int
    nv::Int

    function LinearReduction(coordinates::C, basis::BT, f0::OT, nx::Integer,
            nv::Integer) where {
            C <: Coordinates, BT <: AbstractMatrix, OT <: AbstractVector}
        size(basis, 1) == nx * nv || throw(DimensionMismatch(
            "a basis of $(size(basis, 1)) rows does not fit a $nx × $nv grid"))
        length(f0) == nv ||
            throw(DimensionMismatch("the background has $(length(f0)) values for $nv v nodes"))
        return new{C, BT, OT}(coordinates, basis, f0, nx, nv)
    end
end

"Write the coordinates of the fluctuation `g` into `w`."
to_coordinates!(w, ::GridCoordinates, g, nx, nv) = copyto!(w, g)
to_coordinates!(w, ::ParityCoordinates, g, nx, nv) = parity_split!(w, g, nx, nv)

"Write the fluctuation of the coordinates `w` into `g`."
from_coordinates!(g, ::GridCoordinates, w, nx, nv) = copyto!(g, w)
from_coordinates!(g, ::ParityCoordinates, w, nx, nv) = parity_join!(g, w, nx, nv)

"""
    coordinate_snapshots(coordinates, S, f0)

The `nx nv × m` matrix of the fluctuations of the `nx × nv × m` snapshots `S` around `f0`, one
snapshot per column, in `coordinates`.
"""
function coordinate_snapshots(::GridCoordinates, S::AbstractArray{<:Any, 3}, f0)
    nx, nv, m = size(S)
    return reshape(fluctuation(S, f0), nx * nv, m)
end
function coordinate_snapshots(::ParityCoordinates, S::AbstractArray{<:Any, 3}, f0)
    parity_snapshots(S, f0)
end

"""
    encode(r, f)

The reduced state ``z = B^{\\mathsf T} C(f - f_0)`` of the `nx × nv` state `f` under the
[`LinearReduction`](@ref) `r`.
"""
function encode(r::LinearReduction, f::AbstractMatrix)
    g = vec(fluctuation(f, r.f0))
    return r.basis' * to_coordinates!(similar(g), r.coordinates, g, r.nx, r.nv)
end

"""
    decode(r, z)

The `nx × nv` state ``f_0 + C^{-1} B z`` of the reduced state `z` under the
[`LinearReduction`](@ref) `r`.
"""
function decode(r::LinearReduction, z::AbstractVector)
    w = r.basis * z
    g = from_coordinates!(similar(w), r.coordinates, w, r.nx, r.nv)
    return reshape(g, r.nx, r.nv) .+ reshape(r.f0, 1, :)
end

"""
The parameter space that every basis of this package records: the bases are built per run,
or per set of runs, and not over a sampled parameter space.
"""
unsampled_space() = ParameterSpace(Parameter(:α, [0.0]))

"""
    pod_reduction(S, f0, n)

POD in grid coordinates: the basis ``V`` of the `n` leading left singular vectors of the
fluctuation snapshots of the `nx × nv × m` snapshots `S` around `f0`, from ReducedBasisMethods'
`PODSVD`.
"""
function pod_reduction(S::AbstractArray{<:Any, 3}, f0::AbstractVector, n::Integer)
    nx, nv, _ = size(S)
    G = coordinate_snapshots(GridCoordinates(), S, f0)
    V = ReducedBasis(PODSVD(), G, NamedTuple(), unsampled_space(); k = n).Φ
    return LinearReduction(GridCoordinates(), V, f0, nx, nv)
end

"""
    psd_basis(W, n)

The factor ``Φ`` of `n ÷ 2` columns of the cotangent lift of the vertically stacked parity
snapshots `W` ``= [Q; P]``, from ReducedBasisMethods' `CotangentLiftSVD`: the leading left
singular vectors of ``[Q\\; P]``.
"""
function psd_basis(W::AbstractMatrix, n::Integer)
    return ReducedBasis(CotangentLiftSVD(), W, NamedTuple(), unsampled_space(); k = n ÷ 2).Φ
end

"""
    psd_reduction(S, f0, n)

The cotangent-lift PSD in parity coordinates: the basis ``A = \\operatorname{diag}(Φ, Φ)`` of
`n` columns, with ``Φ`` from [`psd_basis`](@ref) on the parity snapshots of `S` around `f0`.
"""
function psd_reduction(S::AbstractArray{<:Any, 3}, f0::AbstractVector, n::Integer)
    nx, nv, _ = size(S)
    Φ = psd_basis(coordinate_snapshots(ParityCoordinates(), S, f0), n)
    return LinearReduction(ParityCoordinates(), [Φ zero(Φ); zero(Φ) Φ], f0, nx, nv)
end

"""
    relative_projection_error(r, S)

The relative snapshot projection error ``\\|W - B B^{\\mathsf T} W\\|_F / \\|W\\|_F`` of the
`nx × nv × m` snapshots `S` under the [`LinearReduction`](@ref) `r`, where ``W`` holds the
fluctuations of `S` around `r.f0` in the coordinates of `r`. The parity split is ``1/\\sqrt 2``
times an orthogonal map, so the ratio is the same in grid and in parity coordinates.
"""
function relative_projection_error(r::LinearReduction, S::AbstractArray{<:Any, 3})
    W = coordinate_snapshots(r.coordinates, S, r.f0)
    return norm(W - r.basis * (r.basis' * W)) / norm(W)
end

"The slack of the comparison of the POD error with the PSD error."
const POD_OPTIMALITY_TOLERANCE = 1e-12

"""
    check_pod_optimality(name, ns, pod, psd)

Throw an error naming the case `name` when the POD projection errors `pod` at the basis sizes
`ns` rise with `n`, or when one exceeds the PSD error `psd` at the same `n` by more than
[`POD_OPTIMALITY_TOLERANCE`](@ref). By Eckart–Young neither happens to the aggregate errors of
[`relative_projection_error`](@ref): ``\\operatorname{diag}(Φ, Φ)`` has `n` orthonormal columns.
"""
function check_pod_optimality(name, ns, pod, psd)
    for i in 2:length(ns)
        pod[i] ≤ pod[i - 1] || error(
            "$name: the POD projection error rises from $(pod[i - 1]) at n = $(ns[i - 1]) " *
            "to $(pod[i]) at n = $(ns[i])")
    end
    for i in eachindex(ns)
        pod[i] ≤ psd[i] + POD_OPTIMALITY_TOLERANCE || error(
            "$name: the POD projection error $(pod[i]) at n = $(ns[i]) exceeds the PSD " *
            "error $(psd[i]) by more than $POD_OPTIMALITY_TOLERANCE")
    end
    return nothing
end

"""
    checked_projection_errors(name, S, f0, ns; pod = pod_reduction, psd = psd_reduction)

The [`relative_projection_error`](@ref)s `(pod = …, psd = …)` of the snapshots `S` around `f0`
at each basis size in `ns`, with the bases that `pod` and `psd` build, after
[`check_pod_optimality`](@ref). It returns them with the `reductions` `(pod = …, psd = …)`
themselves and the `seconds` that each build took.
"""
function checked_projection_errors(name, S, f0, ns; pod = pod_reduction,
        psd = psd_reduction)
    built = (pod = [@timed(pod(S, f0, n)) for n in ns],
        psd = [@timed(psd(S, f0, n)) for n in ns])
    reductions = map(b -> [t.value for t in b], built)
    errors = map(rs -> [relative_projection_error(r, S) for r in rs], reductions)
    check_pod_optimality(name, ns, errors.pod, errors.psd)

    return (
        errors..., reductions = reductions, seconds = map(b -> [t.time for t in b], built))
end

"""
    full_order_field(case, nx, nv)

VlasovMethods' Arakawa vector field `F(ḟ, t, f, params)` of `case` on the `nx × nv` grid and its
`params`, taken from the integrator that VlasovMethods builds. That integrator steps with
`RK4()`, which builds no Newton solver; the field is the same for every method.
"""
function full_order_field(case::VlasovCase, nx::Integer, nv::Integer)
    int = VlasovMethods.GeometricIntegrator(vlasov_model(case, nx, nv), (0.0, case.tend),
        TIMESTEP, RK4())
    return functions(int.equation).v, parameters(int.equation)
end

"""
    GalerkinField

The Galerkin projection ``\\dot z = B^{\\mathsf T} C F(f_0 + C^{-1} B z)`` of the full-order
vector field `F` under a [`LinearReduction`](@ref), as a callable `(ż, t, z, params)`; build it
with [`galerkin_field`](@ref). A `Float64` state uses its buffers; a state of another element
type, such as the dual numbers of a Newton Jacobian, allocates its own. It subtypes `Function`
because GeometricEquations' `ODEProblem` takes a vector field of that type.
"""
struct GalerkinField{RT <: LinearReduction, FT, PT} <: Function
    reduction::RT
    F::FT
    params::PT
    f::Vector{Float64}
    ḟ::Vector{Float64}
    w::Vector{Float64}
end

"""
    galerkin_field(r, F, params)

The [`GalerkinField`](@ref) of the full-order field `F` with its `params` under `r`.
"""
function galerkin_field(r::LinearReduction, F, params)
    M = r.nx * r.nv
    return GalerkinField(r, F, params, zeros(M), zeros(M), zeros(M))
end

function (g::GalerkinField)(ż::AbstractVector, t, z::AbstractVector, _)
    if eltype(ż) === eltype(z) === Float64
        return galerkin!(ż, t, z, g, g.f, g.ḟ, g.w)
    end
    T = promote_type(eltype(ż), eltype(z))
    M = length(g.f)
    return galerkin!(ż, t, z, g, Vector{T}(undef, M), Vector{T}(undef, M),
        Vector{T}(undef, M))
end

function galerkin!(ż, t, z, g::GalerkinField, f, ḟ, w)
    r = g.reduction
    mul!(w, r.basis, z)
    from_coordinates!(f, r.coordinates, w, r.nx, r.nv)
    for j in 1:(r.nv)
        view(f, ((j - 1) * r.nx + 1):(j * r.nx)) .+= r.f0[j]
    end
    g.F(ḟ, t, f, g.params)
    to_coordinates!(w, r.coordinates, ḟ, r.nx, r.nv)
    mul!(ż, transpose(r.basis), w)

    return ż
end

"""
    reduced_integrator(r, case, z0)

The `ImplicitMidpoint` integrator of the [`GalerkinField`](@ref) of `case` under `r`, from the
reduced state `z0` over ``[0, t_\\mathrm{end}]`` with step [`TIMESTEP`](@ref), and its first
solution step.
"""
function reduced_integrator(r::LinearReduction, case::VlasovCase, z0::AbstractVector)
    F, params = full_order_field(case, r.nx, r.nv)
    problem = ODEProblem(galerkin_field(r, F, params), (0.0, case.tend), TIMESTEP, z0)
    int = GeometricIntegrator(problem, ImplicitMidpoint())

    return int, solutionstep(int, initialstate(problem))
end

"Whether the nonlinear solve of the last step of `int` converged."
step_converged(int) = isconverged(status(int.solver, int.solverstate))

"""
    run_reduced(r, case, nx, nv, f_init; perturb = nothing)

Integrate the reduced model of `case` under the [`LinearReduction`](@ref) `r` with
`ImplicitMidpoint` and step [`TIMESTEP`](@ref) from the encoded `f_init` over
``[0, t_\\mathrm{end}]``. It returns the reduced `states` and the `times` of every step, the
decoded `snapshots` every [`SNAPSHOT_INTERVAL`](@ref), and every step the `mass`, `energy`,
`E1`, `l1`, `l2`, `entropy` and `nonpositive` count of [`invariants`](@ref) of the decoded state.

A step whose `ImplicitMidpoint` solve does not converge or throws SimpleSolvers'
`NonlinearSolverException`, or that leaves a non-finite state, ends the run: its `status` is
`"failed at step <s>"` and `failed_step` is `s`; otherwise the status is `"ok"` and
`failed_step` is `nothing`. Any other exception of a step propagates. `perturb(step, z)`, where
given, is called on the state after each step and before these checks; it lets a test break a
run.
"""
function run_reduced(r::LinearReduction, case::VlasovCase, nx::Integer, nv::Integer,
        f_init::AbstractMatrix; perturb = nothing)
    model = vlasov_model(case, nx, nv)
    x, v = model.distribution.x, model.distribution.v
    hx, hv = step(x), step(v)
    G = model.work
    int, solstep = reduced_integrator(r, case, encode(r, f_init))

    nsteps = round(Int, case.tend / TIMESTEP)
    stride = round(Int, SNAPSHOT_INTERVAL / TIMESTEP)
    times = [s * TIMESTEP for s in 0:nsteps]
    snapshots = zeros(nx, nv, nsteps ÷ stride + 1)
    states = Vector{Vector{Float64}}()
    series = (mass = Float64[], energy = Float64[], E1 = Float64[], l1 = Float64[],
        l2 = Float64[], entropy = Float64[], nonpositive = Int[])
    status, failed_step = "ok", nothing
    for s in 0:nsteps
        if s > 0
            reset!(solstep, times[s + 1])
            solved = try
                integrate!(solstep, int)
                true
            catch e
                e isa NonlinearSolverException || rethrow()
                false
            end
            perturb === nothing || perturb(s, solstep.q)
            if !(solved && step_converged(int) && all(isfinite, solstep.q))
                status, failed_step = "failed at step $s", s
                break
            end
        end
        push!(states, copy(solstep.q))
        f = decode(r, solstep.q)
        d = invariants(f, G, case.k, x, v, hx, hv)
        for name in keys(series)
            push!(series[name], d[name])
        end
        s % stride == 0 && (snapshots[:, :, s ÷ stride + 1] .= f)
    end

    return (status = status, failed_step = failed_step, times = times, states = states,
        snapshots = snapshots, series...)
end

"""
    reduced_step_time(r, case, f_init; steps = 100, warmup = 10)

The wall time in seconds of one `ImplicitMidpoint` step of the reduced model of `case` under `r`
from the encoded `f_init`, averaged over `steps` steps after `warmup` steps.
"""
function reduced_step_time(r::LinearReduction, case::VlasovCase, f_init::AbstractMatrix;
        steps::Integer = 100, warmup::Integer = 10)
    int, solstep = reduced_integrator(
        r, VlasovCase(case.name, case.profile, case.α, case.k, (warmup + steps) * TIMESTEP),
        encode(r, f_init))
    timed_steps!(solstep, int, 1, warmup)

    return timed_steps!(solstep, int, warmup + 1, warmup + steps) / steps
end

"""
    timed_steps!(solstep, int, first, last)

The wall time of the steps `first:last` of `int` on `solstep`. It is a function barrier: the
integrator's type is concrete inside it, so no step dispatches at run time.
"""
function timed_steps!(solstep::SST, int::IT, first::Integer, last::Integer) where {SST, IT}
    return @elapsed for s in first:last
        reset!(solstep, s * TIMESTEP)
        integrate!(solstep, int)
    end
end
