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

## The DLRA solver

```sh
julia --project scripts/dlra.jl --runs-dir runs --results-dir results
```

A dynamical low-rank solver integrates the factored state $f = f_0 + X S V^{\mathsf T}$ of rank
$r \in \{4, 8, 16, 32\}$ (`--smoke`: $\{4, 8\}$, on the smoke runs), with $X$ and $V$
orthonormal in the inner products weighted by $h_x$ and $h_v$. Each step is the BUG step with
augmented bases (K-step, L-step, then the Galerkin S-step on the bases of up to $2r$ columns),
truncated back to rank $r$; each substep is one RK4 step. The discretisation is spectral in $x$
and centred differences in $v$, periodic in $v$, with the same Poisson solve as the full-order
runs. The time step is 0.025, half the full-order step, because RK4 with the spectral derivative
is unstable at 0.05 on $64 \times 128$. The solver does not conserve the mass exactly; the drift
is written, not asserted.

Before it measures, the script checks the solver. At full rank in $x$ it agrees with a full-rank
solver of the same discretisation to $10^{-10}$ over 20 steps on $32 \times 64$, and the
full-rank solver and the run of rank 16 meet the rate check of linear Landau damping (2 % over
$[0, 40]$ on $64 \times 128$; `--smoke`: 10 % over $[0, 25]$ on $32 \times 64$). It writes one row
per case and rank, and per test amplitude of the sweep and rank, to `results/dlra.csv`, with the
metrics of the linear baselines against the full-order runs, the state size
$r(N_x + N_v) + r^2$, the error of the full-rank solver against the same full-order run (the
difference of the two discretisations), and the online cost per DLRA step, measured in a fresh
process, to `results/dlra_cost.toml`. The column `n` holds the rank $r$.

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The suite runs the smoke mode of `scripts/full_order.jl` in the `core` group, and its full mode,
the smoke mode of `scripts/linear_baselines.jl` and the full-basis checks of the Galerkin models in
the `slow` group; `Pkg.test()` runs both. The checks of the DLRA solver run in the `core` group,
and the smoke and the full mode of `scripts/dlra.jl` in the `slow` group.
