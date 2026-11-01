"The time step of every DLRA run and of its full-rank solver, with a classical RK4 per substep."
const DLRA_TIMESTEP = 0.025

"The number of DLRA steps between two stored snapshots, the [`SNAPSHOT_INTERVAL`](@ref) of the
full-order runs."
const DLRA_STRIDE = round(Int, SNAPSHOT_INTERVAL / DLRA_TIMESTEP)

"The largest relative difference of a DLRA run at ``r = n_x`` from the full-rank solver."
const FULL_RANK_TOLERANCE = 1e-10

"The number of steps over which a DLRA run at ``r = n_x`` is compared with the full-rank solver."
const AGREEMENT_STEPS = 20

"The rank of the DLRA run whose damping rate of linear Landau damping is checked."
const CHECK_RANK = 16

"The columns of `dlra.csv`, in order. The column `n` holds the rank ``r``."
const DLRA_COLUMNS = ("method", "kind", "case", "alpha", "n", "state_size", "status",
    "l2_error_max", "l2_errors", "max_l2_error_fluct", "l2_error_fluct", "rate", "rate_status",
    "rate_full_order", "energy_drift", "l1_drift", "l2_drift", "entropy_drift", "mass_drift",
    "nonpositive_nodes", "full_rank_l2_error_max", "offline_seconds",
    "online_seconds_per_step", "online_ratio")

"""
    dlra_mode(smoke)

The settings of [`FULL_MODE`](@ref), or of [`SMOKE_MODE`](@ref) for `smoke = true`, with the
DLRA `ranks` of the mode: ``r ∈ \\{4, 8, 16, 32\\}``, or ``\\{4, 8\\}`` in the smoke mode.
"""
function dlra_mode(smoke::Bool)
    smoke ? merge(SMOKE_MODE, (ranks = (4, 8),)) :
    merge(FULL_MODE, (ranks = (4, 8, 16, 32),))
end

"""
    dlra_state_size(r, nx, nv)

The number of values of the factored state ``(X, S, V)`` of rank `r` on the `nx × nv` grid,
``r (n_x + n_v) + r^2``.
"""
dlra_state_size(r::Integer, nx::Integer, nv::Integer) = r * (nx + nv) + r^2

"""
    spectral_derivative(nx, k)

The `nx × nx` matrix of the spectral derivative on `nx` equidistant nodes of the periodic
interval ``[0, 2π/k)``: ``∂_x e^{imkx} = imk\\, e^{imkx}`` for ``|m| < n_x/2``, and zero for
the Nyquist mode ``m = n_x/2``.
"""
function spectral_derivative(nx::Integer, k::Real)
    return [-2k / nx *
            sum(m * sinpi(2m * (a - b) / nx) for m in 1:(cld(nx, 2) - 1); init = 0.0)
            for a in 1:nx, b in 1:nx]
end

"""
    centred_derivative(nv, hv)

The `nv × nv` matrix of the centred difference ``(g_{j+1} - g_{j-1}) / 2h_v`` with spacing `hv`,
periodic over the `nv` nodes: node `nv` neighbours node 1, as in VlasovMethods' Arakawa scheme.
"""
function centred_derivative(nv::Integer, hv::Real)
    D = zeros(nv, nv)
    for j in 1:nv
        D[j, mod1(j + 1, nv)] += 1 / (2hv)
        D[j, mod1(j - 1, nv)] -= 1 / (2hv)
    end
    return D
end

@doc raw"""
    dlra_discretisation(case, nx, nv)

The discretisation of `case` on the `nx × nv` grid of [`grid`](@ref) that the DLRA solver and
its full-rank solver share: the nodes `x`, `v`, the spacings `hx`, `hv`, the Green's matrix `G`
(``φ = Gρ``) of the FFT Poisson solver of [`vlasov_model`](@ref), the
[`spectral_derivative`](@ref) `Dx` and the [`centred_derivative`](@ref) `Dv`, the initial
condition `f_init`, its [`background`](@ref) `f0`, the charge ``ρ_0 = h_v \sum_j f_{0,j}`` of
`f0` and ``D_v f_0`` as `Dvf0`.
"""
function dlra_discretisation(case::VlasovCase, nx::Integer, nv::Integer)
    model = vlasov_model(case, nx, nv)
    x, v = model.distribution.x, model.distribution.v
    hx, hv = step(x), step(v)
    f_init = initial_condition(case, x, v)
    f0 = background(f_init)
    Dv = centred_derivative(nv, hv)

    return (case = case, nx = nx, nv = nv, x = x, v = collect(v), hx = hx, hv = hv,
        G = model.work, Dx = spectral_derivative(nx, case.k), Dv = Dv, f_init = f_init,
        f0 = f0, ρ0 = hv * sum(f0), Dvf0 = Dv * f0)
end

"One classical RK4 step of size `h` of ``\\dot y = g(y)`` from `y`."
function rk4(g::G, y, h) where {G}
    k1 = g(y)
    k2 = g(y .+ h / 2 .* k1)
    k3 = g(y .+ h / 2 .* k2)
    k4 = g(y .+ h .* k3)
    return y .+ h / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

@doc raw"""
    full_rank_field(d, f)

The right-hand side ``\dot f = -v\, D_x f + (D_x φ) ⊙ (f D_v^{\mathsf T})`` of the `nx × nv` state
`f` on the [`dlra_discretisation`](@ref) `d`, with ``φ = Gρ`` and ``ρ_i = h_v \sum_j f_{ij}``:
VlasovMethods' sign convention, charge ``+1``.
"""
function full_rank_field(d, f::AbstractMatrix)
    φx = d.Dx * (d.G * (d.hv .* vec(sum(f; dims = 2))))
    return -(d.Dx * f) .* d.v' .+ φx .* (f * transpose(d.Dv))
end

"""
    full_rank_step(d, f, h)

One RK4 step of size `h` of the full-rank solver of [`full_rank_field`](@ref) from `f`.
"""
full_rank_step(d, f::AbstractMatrix, h) = rk4(g -> full_rank_field(d, g), f, h)

"""
    weighted_orthonormal(A, h)

The `min(size(A)...)` columns ``Q``, orthonormal in the inner product with weight `h`,
``h Q^{\\mathsf T} Q = I``, of the thin QR factorisation of ``\\sqrt h A``. They span the column
space of `A`, and more where `A` has fewer independent columns.
"""
weighted_orthonormal(A::AbstractMatrix, h::Real) = Matrix(qr(sqrt(h) .* A).Q) ./ sqrt(h)

@doc raw"""
    initial_factors(d, r)

The factors ``(X, S, V)`` of rank `r` of the initial fluctuation ``f(0) - f_0 = X S V^{\mathsf
T}`` of the [`dlra_discretisation`](@ref) `d`, with ``h_x X^{\mathsf T} X = I`` and
``h_v V^{\mathsf T} V = I``: the truncated SVD in the weighted inner product, of the singular
values above ``\max(n_x, n_v)\, ε\, \|f(0)\|``, the round-off of the subtraction of ``f_0`` (the
norm is the weighted spectral norm). Where these are fewer than `r`, both bases are completed
with orthonormal columns drawn from `Xoshiro(42)`.
"""
function initial_factors(d, r::Integer)
    w = sqrt(d.hx * d.hv)
    D = d.f_init .- d.f0'
    F = svd(w .* D)
    q = min(r, count(>(max(d.nx, d.nv) * eps() * opnorm(w .* d.f_init)), F.S))
    rng = Xoshiro(42)
    X = complete_basis(F.U[:, 1:q] ./ sqrt(d.hx), r, d.hx, rng)
    V = complete_basis(F.V[:, 1:q] ./ sqrt(d.hv), r, d.hv, rng)

    return X, d.hx * d.hv .* (X' * D * V), V
end

"The basis `Q` with weight `h`, completed to `r` columns drawn from `rng`."
function complete_basis(Q::AbstractMatrix, r::Integer, h::Real, rng)
    R = randn(rng, size(Q, 1), r - size(Q, 2))
    R .-= Q * (h .* (Q' * R))
    return hcat(Q, weighted_orthonormal(R, h))
end

@doc raw"""
    electric_field(d, A, c)

The derivative ``D_x φ`` of the potential ``φ = Gρ`` of the state ``f_0 + A B^{\mathsf T}``,
whose charge is ``ρ = ρ_0 + A c`` with ``c = h_v B^{\mathsf T} \mathbb 1``.
"""
electric_field(d, A::AbstractMatrix, c::AbstractVector) = d.Dx * (d.G * (d.ρ0 .+ A * c))

@doc raw"""
    bug_step(d, X, S, V, h)

One step of size `h` of the augmented BUG integrator, truncated back to the rank `r` of `S`, for
the state ``f_0 + X S V^{\mathsf T}`` on the [`dlra_discretisation`](@ref) `d`; each substep is
one RK4 step of the Galerkin projection of the field ``F`` of [`full_rank_field`](@ref).

1. The K-step ``\dot K = h_v F(f_0 + K V^{\mathsf T}) V`` from ``K = XS``, and the L-step
   ``\dot L = h_x F(f_0 + X L^{\mathsf T})^{\mathsf T} X`` from ``L = V S^{\mathsf T}``.
2. The augmented bases ``\hat X`` of ``[K, X]`` and ``\hat V`` of ``[L, V]``, of up to ``2r``
   columns, and the S-step ``\dot S = h_x h_v \hat X^{\mathsf T} F(f_0 + \hat X S \hat
   V^{\mathsf T}) \hat V`` from ``(h_x \hat X^{\mathsf T} X)\, S\, (h_v V^{\mathsf T} \hat V)``.
3. The truncation of the new ``S`` to its `r` largest singular values.

The transport term ``-v\, D_x f_0`` vanishes, because ``f_0`` is constant in ``x``. Where the
S-step leaves a non-finite value, it returns the untruncated ``(\hat X, S, \hat V)``.
"""
function bug_step(d, X::AbstractMatrix, S::AbstractMatrix, V::AbstractMatrix, h)
    r = size(S, 1)
    hx, hv = d.hx, d.hv

    # the K-step on the fixed basis V
    Mv = hv .* (V' * (d.v .* V))
    Md = hv .* ((d.Dv * V)' * V)
    b = hv .* (V' * d.Dvf0)
    c = hv .* vec(sum(V; dims = 1))
    K = rk4(X * S, h) do K
        -(d.Dx * K) * Mv .+ electric_field(d, K, c) .* (b' .+ K * Md)
    end

    # the L-step on the fixed basis X
    Mx = hx .* ((d.Dx * X)' * X)
    L = rk4(V * S', h) do L
        φx = electric_field(d, X, hv .* vec(sum(L; dims = 1)))
        -(d.v .* L) * Mx .+ d.Dvf0 * (hx .* (φx' * X)) .+
        (d.Dv * L) * (hx .* (X' * (φx .* X)))
    end

    # the Galerkin S-step on the augmented bases
    X̂ = weighted_orthonormal(hcat(K, X), hx)
    V̂ = weighted_orthonormal(hcat(L, V), hv)
    Cx = hx .* (X̂' * (d.Dx * X̂))
    M̂v = hv .* (V̂' * (d.v .* V̂))
    M̂d = hv .* ((d.Dv * V̂)' * V̂)
    b̂ = hv .* (V̂' * d.Dvf0)
    ĉ = hv .* vec(sum(V̂; dims = 1))
    Ŝ = rk4((hx .* (X̂' * X)) * S * (hv .* (V' * V̂)), h) do S
        φx = electric_field(d, X̂ * S, ĉ)
        -Cx * S * M̂v .+ (hx .* (X̂' * φx)) * b̂' .+ (hx .* (X̂' * (φx .* X̂))) * S * M̂d
    end
    all(isfinite, Ŝ) || return X̂, Ŝ, V̂

    # the truncation back to rank r
    F = svd(Ŝ)
    return X̂ * F.U[:, 1:r], Matrix(Diagonal(F.S[1:r])), V̂ * F.V[:, 1:r]
end

"The `nx × nv` state ``f_0 + X S V^{\\mathsf T}`` of the factors on `d`."
factored_state(d, X, S, V) = d.f0' .+ X * S * V'

"""
    full_rank_agreement(case, nx, nv, r, steps)

The largest relative difference ``\\|f_r - f\\|_2 / \\|f\\|_2`` over `steps` steps of
[`DLRA_TIMESTEP`](@ref) between the DLRA run of rank `r` of `case` on the `nx × nv` grid and the
full-rank solver, both from the initial condition. It is `NaN` when either run leaves a
non-finite value.
"""
function full_rank_agreement(case::VlasovCase, nx::Integer, nv::Integer, r::Integer,
        steps::Integer)
    d = dlra_discretisation(case, nx, nv)
    X, S, V = initial_factors(d, r)
    f = d.f_init
    err = 0.0
    for _ in 1:steps
        X, S, V = bug_step(d, X, S, V, DLRA_TIMESTEP)
        f = full_rank_step(d, f, DLRA_TIMESTEP)
        all(isfinite, S) && all(isfinite, f) || return NaN
        err = max(err, norm(factored_state(d, X, S, V) .- f) / norm(f))
    end
    return err
end

"""
    check_full_rank_agreement(case, nx, nv)

Print the [`full_rank_agreement`](@ref) of `case` at ``r = n_x`` over
[`AGREEMENT_STEPS`](@ref) steps, and throw an error when it is above
[`FULL_RANK_TOLERANCE`](@ref) or not finite.
"""
function check_full_rank_agreement(case::VlasovCase, nx::Integer, nv::Integer)
    err = full_rank_agreement(case, nx, nv, nx, AGREEMENT_STEPS)
    @printf("full-rank check %s on %d × %d: relative difference %.2e\n", case.name, nx, nv,
        err)
    flush(stdout)
    err ≤ FULL_RANK_TOLERANCE || error(
        "full-rank check $(case.name) on $nx × $nv: the DLRA run at r = $nx differs from " *
        "the full-rank solver by $err relative over $AGREEMENT_STEPS steps, above " *
        "$FULL_RANK_TOLERANCE")
    return err
end

"""
    run_full_rank(case, nx, nv)

The full-rank solver of [`full_rank_field`](@ref) for `case` on the `nx × nv` grid with step
[`DLRA_TIMESTEP`](@ref) over ``[0, t_\\mathrm{end}]``: the `times` of every step, ``|E_1|`` of
[`diagnostics`](@ref) at each as `E1`, and the `snapshots` every [`DLRA_STRIDE`](@ref) steps. It
throws an error when the state holds a non-finite value.
"""
function run_full_rank(case::VlasovCase, nx::Integer, nv::Integer)
    d = dlra_discretisation(case, nx, nv)
    nsteps = round(Int, case.tend / DLRA_TIMESTEP)
    times = [s * DLRA_TIMESTEP for s in 0:nsteps]
    snapshots = zeros(nx, nv, nsteps ÷ DLRA_STRIDE + 1)
    E1 = zeros(nsteps + 1)
    f = d.f_init
    for s in 0:nsteps
        s > 0 && (f = full_rank_step(d, f, DLRA_TIMESTEP))
        all(isfinite, f) || error(
            "the full-rank solver of $(case.name), α = $(case.α) holds a non-finite value " *
            "after step $s")
        E1[s + 1] = diagnostics(vec(f), d.G, case.k, d.x, d.v, d.hx, d.hv).E1
        s % DLRA_STRIDE == 0 && (snapshots[:, :, s ÷ DLRA_STRIDE + 1] .= f)
    end
    return (times = times, E1 = E1, snapshots = snapshots)
end

"""
    run_dlra(case, nx, nv, r; perturb = nothing)

The DLRA run of rank `r` of `case` on the `nx × nv` grid: [`bug_step`](@ref) with step
[`DLRA_TIMESTEP`](@ref) from [`initial_factors`](@ref) over ``[0, t_\\mathrm{end}]``. It returns
the `times` of every step, the decoded `snapshots` every [`DLRA_STRIDE`](@ref) steps, the final
`factors` ``(X, S, V)``, the `offline_seconds` of the initial factorisation, and every step the
`mass`, `energy`, `E1`, `l1`, `l2`, `entropy` and `nonpositive` count of [`invariants`](@ref)
of the decoded state.

A step that leaves a non-finite ``S`` ends the run: its `status` is `"failed at step <s>"` and
`failed_step` is `s`; otherwise the status is `"ok"` and `failed_step` is `nothing`.
`perturb(step, S)`, where given, is called on ``S`` after each step and before this check; it
lets a test break a run. It throws an `ArgumentError` unless ``1 ≤ r ≤ \\min(n_x, n_v)``.
"""
function run_dlra(case::VlasovCase, nx::Integer, nv::Integer, r::Integer; perturb = nothing)
    1 ≤ r ≤ min(nx, nv) || throw(ArgumentError(
        "the rank $r is not in 1:$(min(nx, nv)) for the $nx × $nv grid"))
    d = dlra_discretisation(case, nx, nv)
    timed = @timed initial_factors(d, r)
    X, S, V = timed.value

    nsteps = round(Int, case.tend / DLRA_TIMESTEP)
    times = [s * DLRA_TIMESTEP for s in 0:nsteps]
    snapshots = zeros(nx, nv, nsteps ÷ DLRA_STRIDE + 1)
    series = (mass = Float64[], energy = Float64[], E1 = Float64[], l1 = Float64[],
        l2 = Float64[], entropy = Float64[], nonpositive = Int[])
    status, failed_step = "ok", nothing
    for s in 0:nsteps
        if s > 0
            X, S, V = bug_step(d, X, S, V, DLRA_TIMESTEP)
            perturb === nothing || perturb(s, S)
            if !all(isfinite, S)
                status, failed_step = "failed at step $s", s
                break
            end
        end
        f = factored_state(d, X, S, V)
        i = invariants(f, d.G, case.k, d.x, d.v, d.hx, d.hv)
        for name in keys(series)
            push!(series[name], i[name])
        end
        s % DLRA_STRIDE == 0 && (snapshots[:, :, s ÷ DLRA_STRIDE + 1] .= f)
    end

    return (
        status = status, failed_step = failed_step, times = times, snapshots = snapshots,
        factors = (X, S, V), offline_seconds = timed.time, series...)
end

"""
    check_dlra_rates(mode; perturb = nothing)

The checks of the damping rate of linear Landau damping at ``α = 0.01`` on the grid of `mode`,
a [`dlra_mode`](@ref), fitted by [`landau_rate_fit`](@ref) over its `landau_window` against
[`landau_root`](@ref) with its `landau_rtol`: first the full-rank solver, so that a failure of
the DLRA run is the rank's and not the discretisation's, then the DLRA run of rank
[`CHECK_RANK`](@ref), with `perturb`. Each throws the error of [`check_rate`](@ref), and the
second an error of its own when the DLRA run stops.
"""
function check_dlra_rates(mode; perturb = nothing)
    case, (nx, nv) = linear_landau(), mode.grid
    oracle = imag(landau_root(case.k))
    full = run_full_rank(case, nx, nv)
    check_rate("full-rank solver linear Landau γ",
        landau_rate_fit(full.times, log.(full.E1), mode.landau_window).γ, oracle,
        mode.landau_rtol)
    dlra = run_dlra(case, nx, nv, CHECK_RANK; perturb)
    dlra.status == "ok" ||
        error("the DLRA run of rank $CHECK_RANK of linear Landau damping " *
              "on $nx × $nv stopped: $(dlra.status)")
    check_rate("DLRA r = $CHECK_RANK linear Landau γ",
        landau_rate_fit(dlra.times, log.(dlra.E1), mode.landau_window).γ, oracle,
        mode.landau_rtol)
    return nothing
end

"A row of `dlra.csv` with the given columns, every other column empty."
function dlra_row(; kwargs...)
    merge(Dict(c => "" for c in DLRA_COLUMNS),
        Dict(string(k) => string(v) for (k, v) in kwargs))
end

"""
    dlra_data_row(run, kind, r, full_rank_error, mode; perturb = nothing)

The row of the DLRA run of rank `r` of the full-order `run`, as [`read_run`](@ref) returns it:
the [`run_dlra`](@ref), with `perturb`, and its [`trajectory_metrics`](@ref) against the run's
snapshots, around the background of the run's first snapshot, the rate and its status of
[`reduced_rate`](@ref), the rate of [`fitted_rate`](@ref) of the run, the drift of the mass, the
[`dlra_state_size`](@ref), the `full_rank_error` and the `kind`, `"data"` or `"sweep"`. A run that
stops leaves every metric of the trajectory empty.
"""
function dlra_data_row(run, kind, r, full_rank_error, mode; perturb = nothing)
    case = run.case
    dlra = run_dlra(case, run.nx, run.nv, r; perturb)
    common = (method = "DLRA", kind = kind, case = case.name, alpha = case.α, n = r,
        state_size = dlra_state_size(r, run.nx, run.nv), status = dlra.status,
        full_rank_l2_error_max = full_rank_error, offline_seconds = dlra.offline_seconds)
    dlra.status == "ok" || return dlra_row(; common...)

    m = trajectory_metrics(dlra.snapshots, run.snapshots, dlra,
        background(run.snapshots[:, :, 1]))
    rate, rate_status = reduced_rate(case.name, dlra.times, dlra.E1, mode.landau_window)
    rate_full = fitted_rate(case.name, run.times, run.E1, mode.landau_window)

    return dlra_row(; common..., l2_error_max = m.l2_error_max,
        l2_errors = join(m.l2_errors, ';'), max_l2_error_fluct = m.l2_error_fluct_max,
        l2_error_fluct = join(m.l2_errors_fluct, ';'), rate = something(rate, ""),
        rate_status = rate_status, rate_full_order = something(rate_full, ""),
        energy_drift = m.energy_drift, l1_drift = m.l1_drift, l2_drift = m.l2_drift,
        entropy_drift = m.entropy_drift, mass_drift = max_drift(dlra.mass),
        nonpositive_nodes = m.nonpositive_max)
end

"""
    dlra_step_time(case, nx, nv, r; steps = 100, warmup = 10)

The wall time in seconds of one [`bug_step`](@ref) of rank `r` of `case` on the `nx × nv`
grid, averaged over `steps` steps after `warmup` steps from [`initial_factors`](@ref).
"""
function dlra_step_time(case::VlasovCase, nx::Integer, nv::Integer, r::Integer;
        steps::Integer = 100, warmup::Integer = 10)
    d = dlra_discretisation(case, nx, nv)
    X, S, V = initial_factors(d, r)
    _, X, S, V = timed_bug_steps(d, X, S, V, warmup)
    return first(timed_bug_steps(d, X, S, V, steps)) / steps
end

"""
    timed_bug_steps(d, X, S, V, n)

The wall time of `n` steps of [`bug_step`](@ref) from ``(X, S, V)``, and the factors after
them. It is a function barrier: the type of `d` is concrete inside it.
"""
function timed_bug_steps(d::D, X, S, V, n::Integer) where {D}
    elapsed = @elapsed for _ in 1:n
        X, S, V = bug_step(d, X, S, V, DLRA_TIMESTEP)
    end
    return elapsed, X, S, V
end

"""
    dlra_step_times(runs_dir, specs)

Print the id and the thread count of this process, then for each `(file, r)` of `specs` the
[`dlra_step_time`](@ref) of rank `r` of the case of that run file in `runs_dir`.
[`dlra_costs`](@ref) calls it in a fresh process.
"""
function dlra_step_times(runs_dir::AbstractString, specs)
    println(getpid(), ' ', Threads.nthreads())
    for (file, r) in specs
        run = read_run(joinpath(runs_dir, file))
        println(file, ' ', r, ' ', dlra_step_time(run.case, run.nx, run.nv, r))
    end
    return nothing
end

"""
    dlra_costs(results_dir, runs_dir, specs, full_order_seconds)

Measure [`dlra_step_times`](@ref) of `specs` in a fresh Julia process on the active project, and
write them with the machine's `uptime`, the id `pid` and the `threads` of the measuring process
and the full-order cost `full_order_seconds` per step to `<results_dir>/dlra_cost.toml`. It
returns the seconds per step by `(file, r)`.
"""
function dlra_costs(results_dir::AbstractString, runs_dir::AbstractString, specs,
        full_order_seconds::Real)
    code = "using VlasovReductionBenchmark; VlasovReductionBenchmark.dlra_step_times(" *
           "$(repr(runs_dir)), $(repr(collect(specs))))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    lines = readlines(cmd)
    pid, threads = parse.(Int, split(lines[end - length(specs)]))
    seconds = Dict{Tuple{String, Int}, Float64}()
    for line in lines[(end - length(specs) + 1):end]
        file, r, s = split(line)
        seconds[(file, parse(Int, r))] = parse(Float64, s)
    end
    uptime = Sys.isunix() ? readchomp(`uptime`) :
             @sprintf("up %.0f s, load averages %s", Sys.uptime(), join(Sys.loadavg(), " "))

    open(joinpath(results_dir, "dlra_cost.toml"), "w") do io
        TOML.print(io,
            Dict("steps" => 100, "warmup" => 10, "dt" => DLRA_TIMESTEP,
                "method" => "augmented BUG, RK4 substeps", "uptime" => uptime, "pid" => pid,
                "threads" => threads, "julia" => string(VERSION),
                "full_order_seconds_per_step" => full_order_seconds,
                "rows" => [Dict("file" => f, "n" => r, "seconds_per_step" => s)
                 for ((f, r), s) in sort(collect(seconds))]))
    end

    return seconds
end

"""
    write_dlra(path, rows)

Write the `rows` of [`dlra_row`](@ref) to the CSV file `path`, with the header
[`DLRA_COLUMNS`](@ref).
"""
function write_dlra(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(DLRA_COLUMNS, ','))
        for row in rows
            println(io, join((csv_field(row[c]) for c in DLRA_COLUMNS), ','))
        end
    end
    return path
end

"""
    run_dlra_benchmark(; runs_dir, results_dir, smoke = false, perturb = nothing)

The DLRA runs of the full-order runs in `runs_dir`, as `scripts/full_order.jl` writes them, with
the settings of [`dlra_mode`](@ref): the three cases of [`DATA_RUNS`](@ref), each at the amplitude
of its run, and linear Landau damping at the [`TEST_AMPLITUDES`](@ref) of the sweep, for each rank
of the mode.

First it checks the solver: [`check_full_rank_agreement`](@ref) of the three cases on the smoke
grid ``32 × 64``, and [`check_dlra_rates`](@ref) on the grid of the mode. Then for each run it
runs the full-rank solver, whose largest relative ``L^2`` error against the run each of its rows
holds, and writes a [`dlra_data_row`](@ref) of kind `"data"` (the cases) or `"sweep"` (the test
amplitudes) per rank. The online cost of each row that completes comes from
[`dlra_costs`](@ref), as a ratio to `<results_dir>/full_order_cost.toml`; the ratio is per step,
and a DLRA step is half the full-order step. It writes `<results_dir>/dlra.csv`, prints the
number of rows and of failed runs, and returns the rows. `perturb` goes to every
[`run_dlra`](@ref) of a row; it lets a test break the runs.

It throws an error when a check fails, or when the grid of `full_order_cost.toml` or of a run file
is not the grid of the mode.
"""
function run_dlra_benchmark(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false, perturb = nothing)
    mode = dlra_mode(smoke)
    full_cost = TOML.parsefile(joinpath(results_dir, "full_order_cost.toml"))
    (full_cost["nx"], full_cost["nv"]) == mode.grid || error(
        "the full-order cost is measured on $(full_cost["nx"]) × $(full_cost["nv"]), " *
        "not on the $(mode.grid[1]) × $(mode.grid[2]) grid of this mode")
    @printf("DLRA on %d × %d%s, r ∈ %s\n", mode.grid..., smoke ? " (smoke)" : "",
        mode.ranks)
    flush(stdout)

    for case in (linear_landau(), nonlinear_landau(), two_stream())
        check_full_rank_agreement(case, SMOKE_MODE.grid...)
    end
    check_dlra_rates(mode)

    files = vcat([file => "data" for file in DATA_RUNS],
        [run_file(linear_landau(α)) => "sweep" for α in TEST_AMPLITUDES])
    rows, row_files = Dict{String, String}[], String[]
    for (file, kind) in files
        run = read_run(joinpath(runs_dir, file))
        (run.nx, run.nv) == mode.grid || error(
            "$file holds a $(run.nx) × $(run.nv) run, not the $(mode.grid[1]) × " *
            "$(mode.grid[2]) grid of this mode")
        full = run_full_rank(run.case, run.nx, run.nv)
        full_rank_error = maximum(relative_l2_errors(full.snapshots, run.snapshots))
        for r in mode.ranks
            push!(rows, dlra_data_row(run, kind, r, full_rank_error, mode; perturb))
            push!(row_files, file)
            @printf("  DLRA, %s, r = %d: %s\n", file, r, rows[end]["status"])
            flush(stdout)
        end
    end

    specs = [(row_files[i], parse(Int, rows[i]["n"]))
             for i in eachindex(rows) if rows[i]["status"] == "ok"]
    seconds = dlra_costs(results_dir, runs_dir, specs, full_cost["seconds_per_step"])
    for (i, file) in enumerate(row_files)
        key = (file, parse(Int, rows[i]["n"]))
        haskey(seconds, key) || continue
        rows[i]["online_seconds_per_step"] = string(seconds[key])
        rows[i]["online_ratio"] = string(seconds[key] / full_cost["seconds_per_step"])
    end

    path = write_dlra(joinpath(results_dir, "dlra.csv"), rows)
    nfailed = count(r -> r["status"] != "ok", rows)
    @printf("DLRA: %d rows, %d failed, written to %s\n", length(rows), nfailed, path)
    flush(stdout)

    return rows
end
