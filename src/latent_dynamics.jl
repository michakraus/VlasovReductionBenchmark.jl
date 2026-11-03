"""
    latent_snapshots(r, S)

The `n × m` matrix whose column `s` is the latent vector `encode(r, S[:, :, s])` of the
`nx × nv × m` snapshots `S` under the reduction `r`. The PSD reductions and the weight-tied
network write the latent vector as ``[q; p]``, `q` first, as [`qp_split`](@ref) reads it.
"""
function latent_snapshots(r, S::AbstractArray{<:Any, 3})
    return reduce(hcat, [encode(r, S[:, :, s]) for s in axes(S, 3)])
end

"""
    qp_split(Z)

The named tuple `(q = Z[1:n÷2, :], p = Z[n÷2+1:n, :])` of the `n × m` latent snapshots `Z`. It
throws an `ArgumentError` for an odd `n`.
"""
function qp_split(Z::AbstractMatrix)
    n = size(Z, 1)
    iseven(n) || throw(ArgumentError("a latent vector [q; p] has an even length, got $n"))
    return (q = Z[1:(n ÷ 2), :], p = Z[(n ÷ 2 + 1):n, :])
end

"""
    train_latent_sympnet(Z, m_train; epochs, seed, n_layers = 4, batch_size = 32)

Train a `Float64` `GSympNet(n; upscaling_dimension = 2n, n_layers)` on the one-step pairs
`(Z[:, s], Z[:, s + 1])` of the columns `1:m_train` of the `n × m` latent snapshots `Z`, with
`Adam` at the step size [`TRAINING_STEP_SIZE`](@ref) and `Batch(batch_size)`, for `epochs`
epochs. It returns the `network` and the loss `history` of GeometricMachineLearning's `train!`.

The parameters are drawn with `Xoshiro(seed)`. `train!` shuffles the minibatches with the
global random number generator, so the training runs with that generator seeded with `seed` and
restores its state afterwards: the same `seed` gives the same network and history.

It throws an `ArgumentError` for an odd `n`, for `m_train` outside `2:m`, for `epochs < 1` and
for a non-finite value in the columns `1:m_train`.
"""
function train_latent_sympnet(Z::AbstractMatrix, m_train::Integer; epochs::Integer,
        seed::Integer, n_layers::Integer = 4, batch_size::Integer = 32)
    n, m = size(Z)
    check_training_columns(m_train, m)
    data = qp_split(Z[:, 1:m_train])
    all(isfinite, view(Z, :, 1:m_train)) ||
        throw(ArgumentError("the training columns 1:$m_train of Z hold a non-finite value"))
    nn = NeuralNetwork(GSympNet(n; upscaling_dimension = 2n, n_layers), CPU(), Float64;
        rng = Xoshiro(seed))
    optimizer = TrainingOptimizer(nn.params; algorithm = Adam(),
        linesearch = TRAINING_STEP_SIZE)
    loader = DataLoader(data; suppress_info = true)
    state = copy(default_rng())
    seed!(seed)
    history = try
        train!(nn, optimizer, loader, Batch(batch_size), epochs; show_progress = false)
    finally
        copy!(default_rng(), state)
    end

    return (network = nn, history = history)
end

"Throw an `ArgumentError` unless `2 ≤ m_train ≤ m`, so that the training has a one-step pair."
function check_training_columns(m_train::Integer, m::Integer)
    2 ≤ m_train ≤ m || throw(ArgumentError(
        "m_train = $m_train is outside 2:$m, the training columns of $m snapshots"))
    return nothing
end

"""
    latent_rollout(nn, z0, m)

The `n × m` trajectory of the network `nn` from the latent vector `z0`: its first column is
`z0`, and its column `s + 1` is `nn` of its column `s`.
"""
function latent_rollout(nn, z0::AbstractVector, m::Integer)
    Z = similar(z0, length(z0), m)
    Z[:, 1] = z0
    for s in 1:(m - 1)
        Z[:, s + 1] = nn(Z[:, s])
    end
    return Z
end

"""
    latent_run(r, case, S, nn, m_train)

The rollout of the network `nn` from the encoded first snapshot of the `nx × nv × m` snapshots
`S` under the reduction `r`, over `m` snapshots, against `S`. It returns

- `snapshots`, the decoded rollout;
- `l2_error_in_sample` and `l2_error_extrapolation`, the root mean square of the
  [`relative_l2_errors`](@ref) relative to the fluctuation around `r.f0` over the snapshots
  `1:m_train` and `m_train+1:m`; the second is `NaN` for `m_train = m`;
- `energy_drift`, `l1_drift`, `l2_drift` and `mass_drift`, the [`max_drift`](@ref) of the
  series of [`invariants`](@ref) of the decoded snapshots on the grid of `case`;
- `status` `"ok"` and `diverged_at` `nothing`.

A rollout with a non-finite value has the status `"diverged"` and the first column with one as
`diverged_at`; its snapshots from that column on, its errors and its drifts are `NaN`. It
throws an `ArgumentError` for `m_train` outside `2:m`.
"""
function latent_run(r, case::VlasovCase, S::AbstractArray{<:Any, 3}, nn, m_train::Integer)
    m = size(S, 3)
    check_training_columns(m_train, m)
    Z = latent_rollout(nn, encode(r, S[:, :, 1]), m)
    snapshots = fill(NaN, r.nx, r.nv, m)
    diverged_at = findfirst(s -> !all(isfinite, view(Z, :, s)), 1:m)
    for s in 1:(something(diverged_at, m + 1) - 1)
        snapshots[:, :, s] = decode(r, Z[:, s])
    end
    if diverged_at !== nothing
        return (status = "diverged", diverged_at = diverged_at, snapshots = snapshots,
            l2_error_in_sample = NaN, l2_error_extrapolation = NaN, energy_drift = NaN,
            l1_drift = NaN, l2_drift = NaN, mass_drift = NaN)
    end

    errors = relative_l2_errors(snapshots, S, r.f0)
    rms(e) = sqrt(sum(abs2, e) / length(e))
    model = vlasov_model(case, r.nx, r.nv)
    x, v = model.distribution.x, model.distribution.v
    series = [invariants(
                  view(snapshots, :, :, s), model.work, case.k, x, v, step(x), step(v))
              for s in 1:m]
    drift(name) = max_drift(Float64[getfield(d, name) for d in series])

    return (status = "ok", diverged_at = nothing, snapshots = snapshots,
        l2_error_in_sample = rms(errors[1:m_train]),
        l2_error_extrapolation = rms(errors[(m_train + 1):m]), energy_drift = drift(:energy),
        l1_drift = drift(:l1), l2_drift = drift(:l2), mass_drift = drift(:mass))
end
