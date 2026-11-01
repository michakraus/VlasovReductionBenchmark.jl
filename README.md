# VlasovReductionBenchmark

A benchmark of reduced models for the 1D1V Vlasov–Poisson system. It asks whether a reduced model
with a shared canonical latent space reproduces a full-order simulation better than the linear and
low-rank baselines.

This repository holds the code that produces the data and the results, not the data and the
results themselves: `runs/` and `results/` are not tracked, and every file in them comes from a
script in `scripts/`.

## The test cases

All three run with VlasovMethods' grid scheme, with `RK4()` and a time step of 0.05, on
$x \in [0, 4\pi)$ and $v \in [-6, 6]$, with the initial condition
$f(x, v) = (1 + \alpha \cos kx)\, g(v)$ and $k = 0.5$:

| case | $g(v)$ | $\alpha$ | $t$ |
|:--|:--|:--|:--|
| `linear_landau` | $e^{-v^2/2}/\sqrt{2\pi}$ | 0.01 | $[0, 40]$ |
| `nonlinear_landau` | $e^{-v^2/2}/\sqrt{2\pi}$ | 0.5 | $[0, 60]$ |
| `two_stream` | $v^2 e^{-v^2/2}/\sqrt{2\pi}$ | 0.05 | $[0, 60]$ |

Linear Landau damping also runs at the amplitudes of the generalisation sweep: $\alpha \in \{0.005,
0.01, 0.02\}$ for training and $\alpha \in \{0.015, 0.03\}$ for testing.

`landau_root(k)` and `two_stream_root(k)` give the complex root $\omega(k)$ of the dispersion
relation of each equilibrium, which the full-order runs are checked against.

## The full-order data

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project scripts/full_order.jl --runs-dir runs --results-dir results
```

The grid is $64 \times 128$; `--smoke` runs the same cases and checks on $32 \times 64$. The script
writes one HDF5 file per run to `runs/`, and the cost per step to
`results/full_order_cost.toml`. It stops with an error when a run breaks its invariants or when a
fitted rate misses the dispersion relation.

## The linear baselines

```sh
julia --project scripts/linear_baselines.jl --runs-dir runs --results-dir results
```

POD-Galerkin and the cotangent-lift PSD-Galerkin reduce the fluctuation around the spatial
average $f_0(v)$ of each run's initial condition at the latent dimensions
$n \in \{4, 8, 16, 32\}$ (`--smoke`: $\{4, 8\}$, on the smoke runs), and integrate the Galerkin
projection of VlasovMethods' vector field with `ImplicitMidpoint`. PSD works in the parity
coordinates $q$, the odd, and $p$, the even part of $f - f_0$ in $v$. The script writes one row per
method, case and $n$, and per test amplitude of the sweep, to `results/linear_baselines.csv`, with
each relative $L^2$ error both against $f$ and against the fluctuation $f - f_0$ (the columns that
end in `_fluct`), and the online cost per reduced step, measured in a fresh process, to
`results/linear_baselines_cost.toml`. It stops with an error when the POD projection error rises
with $n$ or exceeds the PSD error.

## The autoencoders

```sh
julia --project scripts/autoencoders.jl --runs-dir runs --results-dir results
```

Two symplectic autoencoders of GeometricMachineLearning reduce the same fluctuations at the same
latent dimensions: the untied `SymplecticAutoencoder` on the 128 leading PSD coordinates of each
run, and a multi-modal autoencoder with one grid modality, a `BilinearStiefelLayer` with
$m_x m_v = n$ modes per parity and a weight-tied core from $2n$ to $n$, whose bases and core lift
start from their SVD solutions. Each trains for 2000 steps with three seeds (`--smoke`: 100 steps,
one seed) on the Charbonnier relative reconstruction loss. The reduced model is the manifold
Galerkin projection $\dot z = (\nabla\Psi)^{+} F(f_0 + \Psi(z))$ of VlasovMethods' vector field
through the decoder $\Psi$, integrated with `ImplicitMidpoint`. The script writes one row per
network, case, $n$ and seed, and per test amplitude of the sweep, to `results/autoencoders.csv`,
with the losses before and after training, the online cost per reduced step to
`results/autoencoders_cost.toml`, and the cosine and the norm ratio between the canonical flow of
the energy in parity coordinates and VlasovMethods' vector field to
`results/autoencoders_hamiltonian.toml`. A training whose loss does not fall, or becomes
non-finite, is a row with a `failed` status, and the script goes on.

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The suite runs the smoke mode of `scripts/full_order.jl` in the `core` group, and its full mode,
the smoke mode of `scripts/linear_baselines.jl` and the full-basis checks of the Galerkin models in
the `slow` group; `Pkg.test()` runs both. The `slow` group also runs the smoke mode of
`scripts/autoencoders.jl` and the checks of the two networks; its full mode is a script run and
no test.
