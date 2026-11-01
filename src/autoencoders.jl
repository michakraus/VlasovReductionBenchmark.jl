"The number of PSD coordinates on which the untied autoencoder acts: 64 for `q`, 64 for `p`."
const PSD_COORDINATES = 128

"The modes `(m_x, m_v)` per parity of the `BilinearStiefelLayer` at each latent dimension."
const PARITY_MODES = Dict(4 => (2, 2), 8 => (2, 4), 16 => (4, 4), 32 => (4, 8))

"The step size of the training of both networks."
const TRAINING_STEP_SIZE = 1e-3

"The relative singular value of a decoder's Jacobian below which its rank is deficient."
const RANK_TOLERANCE = 1e-12

"""
    autoencoder_mode(smoke)

The grid mode of the full-order runs, the latent dimensions `ns`, the number of training
`steps` and the `seeds` of the full mode, or of the smoke mode with `smoke = true`.
"""
function autoencoder_mode(smoke::Bool)
    return smoke ?
           (grid_mode = SMOKE_MODE, ns = BASELINE_SIZES.smoke, steps = 100, seeds = (42,)) :
           (grid_mode = FULL_MODE, ns = BASELINE_SIZES.full, steps = 2000,
        seeds = (42, 43, 44))
end

"""
    AutoencoderReduction(encoder, decoder, lift, f0, nx, nv)

The reduction ``f = f_0 + Ψ(z)`` of the `nx × nv` state `f` with the decoder
``Ψ(z) = L ψ(z)``: the network's `decoder` ``ψ`` to `k` coordinates, and the fixed linear
`lift` ``L`` of ``n_x n_v`` rows and `k` columns to the grid fluctuation in x-fastest order.
`encoder` maps an `nx × nv` fluctuation to the latent state ``z``. It holds the thin QR factors
`Q` and `R` of ``L``, so that a least-squares solve against the Jacobian
``∇Ψ = Q (R ∇ψ)`` is one against the `k × n` matrix ``R ∇ψ``.
"""
struct AutoencoderReduction{ET, DT, LT <: AbstractMatrix, QT <: AbstractMatrix,
    RT <: AbstractMatrix, OT <: AbstractVector}
    encoder::ET
    decoder::DT
    lift::LT
    Q::QT
    R::RT
    f0::OT
    nx::Int
    nv::Int

    function AutoencoderReduction(encoder::ET, decoder::DT, lift::LT, f0::OT, nx::Integer,
            nv::Integer) where {ET, DT, LT <: AbstractMatrix, OT <: AbstractVector}
        size(lift, 1) == nx * nv || throw(DimensionMismatch(
            "a lift of $(size(lift, 1)) rows does not fit a $nx × $nv grid"))
        factors = qr(lift)
        Q, R = Matrix(factors.Q), Matrix(factors.R)
        return new{ET, DT, LT, typeof(Q), typeof(R), OT}(
            encoder, decoder, lift, Q, R, f0, nx,
            nv)
    end
end

"""
    encode(r::AutoencoderReduction, f)

The latent state ``z`` of the `nx × nv` state `f`: the network's encoder of ``f - f_0``.
"""
encode(r::AutoencoderReduction, f::AbstractMatrix) = r.encoder(fluctuation(f, r.f0))

"""
    decode(r::AutoencoderReduction, z)

The `nx × nv` state ``f_0 + L ψ(z)`` of the latent state `z`.
"""
function decode(r::AutoencoderReduction, z::AbstractVector)
    return reshape(r.lift * r.decoder(z), r.nx, r.nv) .+ reshape(r.f0, 1, :)
end

"""
    decoder_jacobian(r, z)

The Jacobian ``∇Ψ(z) = L\\, ∇ψ(z)`` of the decoder of `r` at `z`, ``n_x n_v × n``, with ``∇ψ``
from ForwardDiff.
"""
decoder_jacobian(r::AutoencoderReduction, z) = r.lift * ForwardDiff.jacobian(r.decoder, z)

"""
    relative_projection_error(r::AutoencoderReduction, S)

The relative reconstruction error ``\\|G - Ψ(\\mathrm{enc}(G))\\|_F / \\|G\\|_F`` of the
fluctuations ``G`` of the `nx × nv × m` snapshots `S` around `r.f0`.
"""
function relative_projection_error(r::AutoencoderReduction, S::AbstractArray{<:Any, 3})
    G = fluctuation(S, r.f0)
    err = sum(axes(G, 3)) do s
        g = view(G, :, :, s)
        sum(abs2, decode(r, r.encoder(g)) .- reshape(r.f0, 1, :) .- g)
    end
    return sqrt(err / sum(abs2, G))
end

"The grid encoder of a multi-modal network: its `encode` of the `:grid` modality."
struct GridEncoder{NT}
    network::NT
end
(e::GridEncoder)(g::AbstractMatrix) = GeometricMachineLearning.encode(e.network, :grid, g)

"""
    PSDCoordinateEncoder(encoder, Φ, nx, nv)

The encoder of the untied network on PSD coordinates: the coordinates
``(Φ^{\\mathsf T} q, Φ^{\\mathsf T} p)`` of the parity vector of a fluctuation, mapped by the
network's `encoder`.
"""
struct PSDCoordinateEncoder{ET, BT <: AbstractMatrix}
    encoder::ET
    Φ::BT
    nx::Int
    nv::Int
end
function (e::PSDCoordinateEncoder)(g::AbstractMatrix)
    return e.encoder(psd_coordinates(e.Φ, parity_split!(similar(vec(g)), vec(g), e.nx, e.nv)))
end

"""
    psd_coordinates(Φ, W)

The PSD coordinates ``[Φ^{\\mathsf T} Q; Φ^{\\mathsf T} P]`` of the parity vectors `W`
``= [Q; P]``, a vector or one per column.
"""
function psd_coordinates(Φ::AbstractMatrix, W::AbstractMatrix)
    N = size(Φ, 1)
    return vcat(Φ' * W[1:N, :], Φ' * W[(N + 1):(2N), :])
end
function psd_coordinates(Φ::AbstractMatrix, w::AbstractVector)
    N = size(Φ, 1)
    return vcat(Φ' * w[1:N], Φ' * w[(N + 1):(2N)])
end

"The matrix whose column `i` is `f` of the `i`-th unit vector of length `k`."
lift_matrix(f, k::Integer) = reduce(hcat, [f(Float64.(1:k .== i)) for i in 1:k])

"""
    charbonnier_reconstruction(x̂, x, ε)

The reconstruction term ``(\\sqrt{\\|x̂ - x\\|_F^2 + ε^2} - ε) / \\sqrt{\\|x\\|_F^2 + ε^2}`` of
GeometricMachineLearning's `CharbonnierRelativeLoss`, which has no method for the untied network.
"""
function charbonnier_reconstruction(x̂, x, ε)
    return (sqrt(sum(abs2, x̂ .- x) + ε^2) - ε) / sqrt(sum(abs2, x) + ε^2)
end

"""
    train_network!(loss, ps, steps; step_size = TRAINING_STEP_SIZE)

Take `steps` steps of `TrainingOptimizer` with
`CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam())` and `step_size` on
`loss(ps)`, with the gradient from Zygote. It returns the `status`, the loss `first` before the
first step and the loss `last` after the last step.

A loss that is not finite before step `s + 1` ends the training with the status
`"failed at step <s>"`. A last loss that is not below the first gives the status
`"failed: the loss did not fall (<first> → <last>)"`: a training that does not move is a failed
training. Otherwise the status is `"ok"`.
"""
function train_network!(loss, ps, steps::Integer; step_size::Real = TRAINING_STEP_SIZE)
    optimizer = TrainingOptimizer(ps;
        algorithm = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam()),
        linesearch = step_size)
    first_loss = NaN
    for s in 0:(steps - 1)
        result = Zygote.withgradient(loss, ps)
        s == 0 && (first_loss = result.val)
        isfinite(result.val) ||
            return (status = "failed at step $s", first = first_loss, last = result.val)
        optimization_step!(ps, optimizer, result.grad[1])
    end
    last_loss = loss(ps)
    isfinite(last_loss) ||
        return (status = "failed at step $steps", first = first_loss, last = last_loss)
    status = last_loss < first_loss ? "ok" :
             "failed: the loss did not fall ($first_loss → $last_loss)"

    return (status = status, first = first_loss, last = last_loss)
end

"""
    weight_tied_network(G, n; seed)

The multi-modal network of one `:grid` modality on the `nx × nv × m` fluctuation snapshots `G`
at latent dimension `n`: `BilinearStiefelLayer(nx, nv, m_x, m_v)` with the modes of
[`PARITY_MODES`](@ref) and the core `WeightTiedSymplecticAutoencoder(2n, n;
n_ambient_layers = 0)`, built with `Xoshiro(seed)`. The layer's bases are set by its `solve!`
on `G`, and the core's last layer, the PSD lift, by `PSDArch(2n, n)` `solve!` on the layer's
coordinates of `G`.
"""
function weight_tied_network(G::AbstractArray{<:Any, 3}, n::Integer; seed::Integer)
    nx, nv, _ = size(G)
    mx, mv = PARITY_MODES[n]
    frontend = BilinearStiefelLayer(nx, nv, mx, mv)
    core = WeightTiedSymplecticAutoencoder(2n, n; n_ambient_layers = 0)
    nn = NeuralNetwork(MultiModalSymplecticAutoencoder((grid = frontend,), core), CPU(),
        Float64; rng = Xoshiro(seed))
    ps = nn.params
    solve!(frontend, ps.frontends.grid, G)
    coordinates = frontend(G, ps.frontends.grid)
    psd = NeuralNetwork(PSDArch(2n, n), CPU(), Float64; rng = Xoshiro(seed))
    solve!(psd, vcat(coordinates.q, coordinates.p))
    copyto!(parent(ps.core[end].weight), parent(psd.params[1].weight))

    return nn
end

"""
    weight_tied_reduction(nn, f0, nx, nv)

The [`AutoencoderReduction`](@ref) of the multi-modal network `nn`: its `encode` of the `:grid`
modality, its core decoder, and the inverse of its `BilinearStiefelLayer` as the lift.
"""
function weight_tied_reduction(nn, f0::AbstractVector, nx::Integer, nv::Integer)
    inverse = nn.model.inverse_frontends.grid
    ps = nn.params.frontends.grid
    N = nn.architecture.full_dim ÷ 2
    lift = lift_matrix(e -> vec(inverse((q = e[1:N], p = e[(N + 1):end]), ps)), 2N)
    return AutoencoderReduction(
        GridEncoder(nn), GeometricMachineLearning.decoder(nn), lift,
        f0, nx, nv)
end

"""
    untied_reduction(nn, Φ, f0, nx, nv)

The [`AutoencoderReduction`](@ref) of the untied network `nn` on the PSD coordinates of the
factor `Φ`: the encoder of [`PSDCoordinateEncoder`](@ref), the network's decoder, and the lift
of the PSD coordinates through ``\\operatorname{diag}(Φ, Φ)`` and [`parity_join!`](@ref).
"""
function untied_reduction(nn, Φ::AbstractMatrix, f0::AbstractVector, nx::Integer,
        nv::Integer)
    lift = lift_matrix(PSD_COORDINATES) do y
        w = vcat(Φ * y[1:(end ÷ 2)], Φ * y[(end ÷ 2 + 1):end])
        parity_join!(similar(w), w, nx, nv)
    end
    return AutoencoderReduction(
        PSDCoordinateEncoder(GeometricMachineLearning.encoder(nn), Φ, nx, nv),
        GeometricMachineLearning.decoder(nn), lift, f0, nx, nv)
end

"""
    train_weight_tied(S, f0, n; seed, steps, step_size = TRAINING_STEP_SIZE)

Build the [`weight_tied_network`](@ref) of the fluctuations of the `nx × nv × m` snapshots `S`
around `f0` and train it with [`train_network!`](@ref) on GeometricMachineLearning's
`CharbonnierRelativeLoss` of the `:grid` modality. It returns the `reduction` of
[`weight_tied_reduction`](@ref), the `training` result and the `network`.
"""
function train_weight_tied(S::AbstractArray{<:Any, 3}, f0::AbstractVector, n::Integer;
        seed::Integer, steps::Integer, step_size::Real = TRAINING_STEP_SIZE)
    nx, nv, _ = size(S)
    G = fluctuation(S, f0)
    nn = weight_tied_network(G, n; seed)
    loss = CharbonnierRelativeLoss()
    training = train_network!(ps -> loss(nn.model, ps, :grid, G), nn.params, steps;
        step_size)

    return (reduction = weight_tied_reduction(nn, f0, nx, nv), training = training,
        network = nn)
end

"""
    train_untied(S, f0, n; seed, steps, step_size = TRAINING_STEP_SIZE)

Build GeometricMachineLearning's `SymplecticAutoencoder(128, n)` with `Xoshiro(seed)` on the
[`PSD_COORDINATES`](@ref) coordinates of the factor ``Φ`` of [`psd_basis`](@ref) of the parity
snapshots of `S` around `f0`, and train it with [`train_network!`](@ref) on the
[`charbonnier_reconstruction`](@ref) of those coordinates, with the `ε` of
`CharbonnierRelativeLoss()`. It returns the `reduction` of [`untied_reduction`](@ref), the
`training` result and the `network`.
"""
function train_untied(S::AbstractArray{<:Any, 3}, f0::AbstractVector, n::Integer;
        seed::Integer, steps::Integer, step_size::Real = TRAINING_STEP_SIZE)
    nx, nv, _ = size(S)
    W = parity_snapshots(S, f0)
    Φ = psd_basis(W, PSD_COORDINATES)
    Y = psd_coordinates(Φ, W)
    nn = NeuralNetwork(SymplecticAutoencoder(PSD_COORDINATES, n), CPU(), Float64;
        rng = Xoshiro(seed))
    ε = CharbonnierRelativeLoss().ε
    training = train_network!(ps -> charbonnier_reconstruction(nn.model(Y, ps), Y, ε),
        nn.params, steps; step_size)

    return (reduction = untied_reduction(nn, Φ, f0, nx, nv), training = training,
        network = nn)
end

"The two networks, by name, with the function that trains each."
const NETWORKS = ("untied" => train_untied, "weight-tied" => train_weight_tied)

"The decoder's Jacobian has a relative singular value below [`RANK_TOLERANCE`](@ref)."
struct RankDeficientDecoder <: Exception
    ratio::Float64
end
function Base.showerror(io::IO, e::RankDeficientDecoder)
    print(io, "the decoder's Jacobian has the relative singular value $(e.ratio), below ",
        RANK_TOLERANCE)
end

"The `Float64` value of `x`, through every level of ForwardDiff's dual numbers."
real_value(x::Real) = x
real_value(x::ForwardDiff.Dual) = real_value(ForwardDiff.value(x))

"""
    check_rank(M)

Throw a [`RankDeficientDecoder`](@ref) when the smallest singular value of the values of `M` is
below [`RANK_TOLERANCE`](@ref) times its largest, or when `M` is zero. A matrix with a value
that is not finite is not checked: the solve of the step that evaluates it fails on the
non-finite field instead, as it does for a `LinearReduction`.
"""
function check_rank(M::AbstractMatrix)
    V = real_value.(M)
    all(isfinite, V) || return nothing
    s = svdvals(V)
    s[1] > 0 && s[end] ≥ RANK_TOLERANCE * s[1] ||
        throw(RankDeficientDecoder(s[1] > 0 ? s[end] / s[1] : 0.0))
    return nothing
end

"""
    ManifoldGalerkinField(reduction, F, params)

The manifold Galerkin projection ``\\dot z = (∇Ψ(z))^{+} F(f_0 + Ψ(z))`` of the full-order
field `F` with its `params` under the [`AutoencoderReduction`](@ref) `reduction`, as a callable
`(ż, t, z, params)`: the least-squares solve against the Jacobian ``∇Ψ = Q (R ∇ψ)``, which is the
solve of ``R ∇ψ\\, \\dot z = Q^{\\mathsf T} F``. It throws a [`RankDeficientDecoder`](@ref) when
the Jacobian has numerical rank below `n` (see [`check_rank`](@ref)). It subtypes `Function`
because GeometricEquations' `ODEProblem` takes a vector field of that type.
"""
struct ManifoldGalerkinField{RT <: AutoencoderReduction, FT, PT} <: Function
    reduction::RT
    F::FT
    params::PT
end

function (m::ManifoldGalerkinField)(ż::AbstractVector, t, z::AbstractVector, _)
    r = m.reduction
    M = r.R * ForwardDiff.jacobian(r.decoder, z)
    check_rank(M)
    f = r.lift * r.decoder(z)
    for j in 1:(r.nv)
        view(f, ((j - 1) * r.nx + 1):(j * r.nx)) .+= r.f0[j]
    end
    ḟ = similar(f)
    m.F(ḟ, t, f, m.params)
    ż .= qr(M) \ (transpose(r.Q) * ḟ)

    return ż
end

"""
    reduced_integrator(r::AutoencoderReduction, case, z0)

The `ImplicitMidpoint` integrator of the [`ManifoldGalerkinField`](@ref) of `case` under `r`,
from `z0` over ``[0, t_\\mathrm{end}]`` with step [`TIMESTEP`](@ref), and its first solution
step.
"""
function reduced_integrator(r::AutoencoderReduction, case::VlasovCase, z0::AbstractVector)
    F, params = full_order_field(case, r.nx, r.nv)
    problem = ODEProblem(ManifoldGalerkinField(r, F, params), (0.0, case.tend), TIMESTEP, z0)
    int = GeometricIntegrator(problem, ImplicitMidpoint())

    return int, solutionstep(int, initialstate(problem))
end

"""
    run_reduced(r::AutoencoderReduction, case, nx, nv, f_init; perturb = nothing)

Integrate the [`ManifoldGalerkinField`](@ref) of `case` under `r` as the method of a
`LinearReduction` does, with the same results. A step whose field throws a
[`RankDeficientDecoder`](@ref) ends the run with the status
`"failed at step <s>: rank-deficient decoder"`; at the initial state, which the first solution
step evaluates the field at, `s` is 0 and the run has no states.
"""
function run_reduced(r::AutoencoderReduction, case::VlasovCase, nx::Integer, nv::Integer,
        f_init::AbstractMatrix; perturb = nothing)
    model = vlasov_model(case, nx, nv)
    x, v = model.distribution.x, model.distribution.v
    hx, hv = step(x), step(v)
    G = model.work
    nsteps = round(Int, case.tend / TIMESTEP)
    stride = round(Int, SNAPSHOT_INTERVAL / TIMESTEP)
    times = [s * TIMESTEP for s in 0:nsteps]
    snapshots = zeros(nx, nv, nsteps ÷ stride + 1)
    states = Vector{Vector{Float64}}()
    series = (mass = Float64[], energy = Float64[], E1 = Float64[], l1 = Float64[],
        l2 = Float64[], entropy = Float64[], nonpositive = Int[])
    status, failed_step = "ok", nothing
    # the first solution step evaluates the field at the initial state
    built = try
        reduced_integrator(r, case, encode(r, f_init))
    catch e
        e isa RankDeficientDecoder || rethrow()
        status, failed_step = "failed at step 0: rank-deficient decoder", 0
        nothing
    end
    int, solstep = something(built, (nothing, nothing))
    for s in 0:(built === nothing ? -1 : nsteps)
        if s > 0
            reset!(solstep, times[s + 1])
            reason = ""
            solved = try
                integrate!(solstep, int)
                true
            catch e
                e isa NonlinearSolverException || e isa RankDeficientDecoder || rethrow()
                e isa RankDeficientDecoder && (reason = ": rank-deficient decoder")
                false
            end
            perturb === nothing || perturb(s, solstep.q)
            if !(solved && step_converged(int) && all(isfinite, solstep.q))
                status, failed_step = "failed at step $s$reason", s
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
    reduced_step_time(r::AutoencoderReduction, case, f_init; steps = 100, warmup = 10)

The wall time in seconds of one step of the reduced model of `case` under `r`, as the method of
a `LinearReduction` measures it.
"""
function reduced_step_time(
        r::AutoencoderReduction, case::VlasovCase, f_init::AbstractMatrix;
        steps::Integer = 100, warmup::Integer = 10)
    int, solstep = reduced_integrator(
        r, VlasovCase(case.name, case.profile, case.α, case.k, (warmup + steps) * TIMESTEP),
        encode(r, f_init))
    timed_steps!(solstep, int, 1, warmup)

    return timed_steps!(solstep, int, warmup + 1, warmup + steps) / steps
end

"""
    canonical_flow_comparison(case, f)

The cosine and the norm ratio between the canonical flow ``(\\partial H/\\partial p,
-\\partial H/\\partial q)`` of the discrete energy ``H`` of [`diagnostics`](@ref), pulled back to
the parity coordinates ``(q, p)`` of [`parity_split`](@ref), and the Arakawa field of
VlasovMethods in the same coordinates, at the `nx × nv` state `f` of `case`. The gradient of
``H`` is ``h_x h_v (v_j^2/2 + φ_i)``, and the transpose of [`parity_join!`](@ref) is twice
[`parity_split!`](@ref).
"""
function canonical_flow_comparison(case::VlasovCase, f::AbstractMatrix)
    nx, nv = size(f)
    model = vlasov_model(case, nx, nv)
    v = model.distribution.v
    hx, hv = step(model.distribution.x), step(v)
    φ = model.work * (hv .* vec(sum(f; dims = 2)))
    ∇H = vec(hx * hv .* (φ .+ reshape(v .^ 2 ./ 2, 1, :)))
    ∇w = 2 .* parity_split!(similar(∇H), ∇H, nx, nv)
    N = length(∇w) ÷ 2
    flow = vcat(∇w[(N + 1):end], -∇w[1:N])
    F, params = full_order_field(case, nx, nv)
    ḟ = similar(∇H)
    F(ḟ, 0.0, vec(copy(f)), params)
    arakawa = parity_split!(similar(ḟ), ḟ, nx, nv)

    return (cosine = dot(flow, arakawa) / (norm(flow) * norm(arakawa)),
        norm_ratio = norm(flow) / norm(arakawa))
end

"The columns of `autoencoders.csv`, in order."
const AUTOENCODER_COLUMNS = (BASELINE_COLUMNS[1:5]..., "seed", BASELINE_COLUMNS[6:end]...,
    "loss_stage_I", "loss_final")

"""
    write_autoencoder_rows(path, rows)

Write the `rows` to the CSV file `path`, with the header [`AUTOENCODER_COLUMNS`](@ref).
"""
function write_autoencoder_rows(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(AUTOENCODER_COLUMNS, ','))
        for row in rows
            println(io, join((csv_field(get(row, c, "")) for c in AUTOENCODER_COLUMNS), ','))
        end
    end
    return path
end

"""
    autoencoder_step_times(path)

Print the id and the thread count of this process, then for each `(key, reduction, case,
f_init)` serialized in `path` the `key` and the wall time per step of
[`reduced_step_time`](@ref). [`autoencoder_costs`](@ref) calls it in a fresh process.
"""
function autoencoder_step_times(path::AbstractString)
    println(getpid(), ' ', Threads.nthreads())
    for (key, r, case, f_init) in deserialize(path)
        println(join(key, ' '), ' ', reduced_step_time(r, case, f_init))
    end
    return nothing
end

"""
    autoencoder_costs(results_dir, specs, full_order_seconds)

Measure [`autoencoder_step_times`](@ref) of the `specs`, each `(key, reduction, case, f_init)`
with the key `(file, network, n, seed)`, in a fresh Julia process on the active project, and
write them with the machine's `uptime`, the id `pid` and the `threads` of the measuring process
and the full-order cost `full_order_seconds` per step to `<results_dir>/autoencoders_cost.toml`.
It returns the seconds per step by key.
"""
function autoencoder_costs(results_dir::AbstractString, specs, full_order_seconds::Real)
    seconds = Dict{Tuple{String, String, Int, Int}, Float64}()
    pid, threads = 0, 0
    mktempdir() do dir
        path = joinpath(dir, "reductions.jls")
        serialize(path, specs)
        code = "using VlasovReductionBenchmark; " *
               "VlasovReductionBenchmark.autoencoder_step_times($(repr(path)))"
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        lines = readlines(cmd)
        pid, threads = parse.(Int, split(lines[end - length(specs)]))
        for line in lines[(end - length(specs) + 1):end]
            file, network, n, seed, s = split(line)
            seconds[(file, network, parse(Int, n), parse(Int, seed))] = parse(Float64, s)
        end
    end
    uptime = Sys.isunix() ? readchomp(`uptime`) :
             @sprintf("up %.0f s, load averages %s", Sys.uptime(), join(Sys.loadavg(), " "))

    open(joinpath(results_dir, "autoencoders_cost.toml"), "w") do io
        TOML.print(io,
            Dict("steps" => 100, "warmup" => 10, "dt" => TIMESTEP,
                "method" => "ImplicitMidpoint", "uptime" => uptime, "pid" => pid,
                "threads" => threads, "julia" => string(VERSION),
                "full_order_seconds_per_step" => full_order_seconds,
                "rows" => [Dict("file" => f, "network" => w, "n" => n, "seed" => s,
                     "seconds_per_step" => t)
                 for ((f, w, n, s), t) in sort(collect(seconds))]))
    end

    return seconds
end

"""
    run_autoencoders(; runs_dir, results_dir, smoke = false, perturb = nothing,
        step_size = TRAINING_STEP_SIZE)

The untied and the weight-tied autoencoder of the full-order runs in `runs_dir`, as
`scripts/full_order.jl` writes them, with the mode of [`autoencoder_mode`](@ref).

For each run of [`DATA_RUNS`](@ref), each network of [`NETWORKS`](@ref), each `n` and each
seed it trains the network on the run's fluctuation snapshots around the background of its
first snapshot, integrates the [`ManifoldGalerkinField`](@ref) over the run's time span and
writes a [`data_row`](@ref) with the seed, the loss `loss_stage_I` before the first training
step (for the weight-tied network, after the `solve!` of its bases and its lift) and the loss
`loss_final` after the last step, and the time of the build and the training as the offline
cost. For the amplitude sweep it trains on the three training runs together, around the
background of the first, and writes the largest relative error at each test amplitude. A
training that fails, whose loss becomes non-finite or does not fall (see
[`train_network!`](@ref)), is a row with its status and empty metrics, as is a reduced run that
stops; the other rows go on. The online cost of each data row that completes comes from
[`autoencoder_costs`](@ref).

It writes `<results_dir>/autoencoders.csv`, and the [`canonical_flow_comparison`](@ref) of the
first snapshot of the linear Landau run to `<results_dir>/autoencoders_hamiltonian.toml`,
prints the number of rows and of failed rows, and returns the rows. `perturb` goes to every
[`run_reduced`](@ref) and `step_size` to every training; they let a test break the runs.

It throws an error when the grid of `full_order_cost.toml` or of a run file is not the grid of
the mode.
"""
function run_autoencoders(; runs_dir::AbstractString, results_dir::AbstractString,
        smoke::Bool = false, perturb = nothing, step_size::Real = TRAINING_STEP_SIZE)
    mode = autoencoder_mode(smoke)
    grid_mode = mode.grid_mode
    full_cost = TOML.parsefile(joinpath(results_dir, "full_order_cost.toml"))
    (full_cost["nx"], full_cost["nv"]) == grid_mode.grid || error(
        "the full-order cost is measured on $(full_cost["nx"]) × $(full_cost["nv"]), " *
        "not on the $(grid_mode.grid[1]) × $(grid_mode.grid[2]) grid of this mode")
    @printf("autoencoders on %d × %d%s, n ∈ %s, %d steps, seeds %s\n", grid_mode.grid...,
        smoke ? " (smoke)" : "", mode.ns, mode.steps, mode.seeds)
    flush(stdout)

    rows, specs = Dict{String, String}[], []
    for file in DATA_RUNS
        run = read_run(joinpath(runs_dir, file))
        (run.nx, run.nv) == grid_mode.grid || error(
            "$file holds a $(run.nx) × $(run.nv) run, not the $(grid_mode.grid[1]) × " *
            "$(grid_mode.grid[2]) grid of this mode")
        S = run.snapshots
        f0 = background(S[:, :, 1])
        if run.case.name == "linear_landau"
            comparison = canonical_flow_comparison(run.case, S[:, :, 1])
            open(joinpath(results_dir, "autoencoders_hamiltonian.toml"), "w") do io
                TOML.print(io,
                    Dict("case" => run.case.name, "alpha" => run.case.α, "t" => 0.0,
                        "nx" => run.nx, "nv" => run.nv, "cosine" => comparison.cosine,
                        "norm_ratio" => comparison.norm_ratio))
            end
        end
        for (network, build) in NETWORKS, n in mode.ns, seed in mode.seeds
            label = "$network, $file, n = $n, seed $seed"
            timed = @timed build(S, f0, n; seed, steps = mode.steps, step_size)
            trained, offline = timed.value, timed.time
            losses = (seed = seed, loss_stage_I = trained.training.first,
                loss_final = trained.training.last)
            row = if trained.training.status == "ok"
                r = trained.reduction
                merge(
                    data_row(network, run, n, r, offline,
                        relative_projection_error(r, S), grid_mode; perturb),
                    Dict(string(k) => string(v) for (k, v) in pairs(losses)))
            else
                baseline_row(; method = network, kind = "data", case = run.case.name,
                    alpha = run.case.α, n = n, state_size = n,
                    status = trained.training.status, offline_seconds = offline, losses...)
            end
            push!(rows, row)
            row["status"] == "ok" &&
                push!(specs, (
                    (file, network, n, seed), trained.reduction, run.case, S[:, :, 1]))
            @printf("  %s: %s\n", label, row["status"])
            flush(stdout)
        end
    end

    training_runs = [read_run(joinpath(runs_dir, run_file(linear_landau(α))))
                     for α in TRAIN_AMPLITUDES]
    test_runs = [read_run(joinpath(runs_dir, run_file(linear_landau(α))))
                 for α in TEST_AMPLITUDES]
    S = cat((r.snapshots for r in training_runs)...; dims = 3)
    f0 = background(training_runs[1].snapshots[:, :, 1])
    for (network, build) in NETWORKS, n in mode.ns, seed in mode.seeds
        label = "$network, sweep, n = $n, seed $seed"
        timed = @timed build(S, f0, n; seed, steps = mode.steps, step_size)
        trained, offline = timed.value, timed.time
        losses = (seed = seed, loss_stage_I = trained.training.first,
            loss_final = trained.training.last)
        for run in test_runs
            common = (method = network, kind = "sweep", case = run.case.name,
                alpha = run.case.α, n = n, state_size = n, offline_seconds = offline,
                losses...)
            if trained.training.status != "ok"
                push!(rows, baseline_row(; common..., status = trained.training.status))
                continue
            end
            reduced = run_reduced(trained.reduction, run.case, run.nx, run.nv,
                run.snapshots[:, :, 1]; perturb)
            reduced.status == "ok" ||
                (push!(rows, baseline_row(; common..., status = reduced.status)); continue)
            f0_run = background(run.snapshots[:, :, 1])
            push!(rows,
                baseline_row(; common..., status = reduced.status,
                    l2_error_max = maximum(relative_l2_errors(reduced.snapshots,
                        run.snapshots)),
                    max_l2_error_fluct = maximum(relative_l2_errors(reduced.snapshots,
                        run.snapshots, f0_run))))
        end
        @printf("  %s: %s\n", label, trained.training.status)
        flush(stdout)
    end

    seconds = autoencoder_costs(results_dir, specs, full_cost["seconds_per_step"])
    for row in rows
        row["kind"] == "data" || continue
        key = (row["case"] * "_alpha" * row["alpha"] * ".h5", row["method"],
            parse(Int, row["n"]), parse(Int, row["seed"]))
        haskey(seconds, key) || continue
        row["online_seconds_per_step"] = string(seconds[key])
        row["online_ratio"] = string(seconds[key] / full_cost["seconds_per_step"])
    end

    path = write_autoencoder_rows(joinpath(results_dir, "autoencoders.csv"), rows)
    nfailed = count(r -> r["status"] != "ok", rows)
    @printf("autoencoders: %d rows, %d failed, written to %s\n", length(rows), nfailed,
        path)
    flush(stdout)

    return rows
end
