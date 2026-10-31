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

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The suite runs the script's smoke mode in the `core` group and its full mode in the `slow` group;
`Pkg.test()` runs both.
