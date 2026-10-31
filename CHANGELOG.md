# Changelog

All notable changes to VlasovReductionBenchmark.jl are recorded here. Nothing here is registered,
so `[Unreleased]` has no target version.

## [Unreleased]

### New Features

- The project, its tests (Aqua and ExplicitImports in the suite) and the shared git hooks and CI.
  VlasovMethods comes from its `main` through `[sources]`, because no registered release binds
  ReducedComplexityModeling 0.5.
- The three test cases on the grid $x \in [0, 4\pi)$, $v \in [-6, 6]$: linear Landau damping
  (`linear_landau`, $\alpha = 0.01$, $t \le 40$), nonlinear Landau damping (`nonlinear_landau`,
  $\alpha = 0.5$, $t \le 60$) and the two-stream instability (`two_stream`, $\alpha = 0.05$,
  $t \le 60$), with `vlasov_model` to build VlasovMethods' grid model of each on a given grid.
- The oracle of the linear rates: the roots of the dielectric function of the Maxwellian
  (`landau_root`, through the plasma dispersion function on the Landau contour) and of the
  two-stream profile (`two_stream_root`, through the plasma dispersion function or by quadrature
  over the real line).
- `scripts/full_order.jl --runs-dir <d> --results-dir <r> [--smoke]`: the full-order data on
  $64 \times 128$ (smoke: $32 \times 64$), one HDF5 file per run, with the snapshots every 0.5,
  the field energy and $|E_1|$ every step, the drifts of the mass, the energy, the $L^2$ norm and
  the entropy, and the train/test label of the amplitude sweep. The script asserts the mass to
  $10^{-12}$ and the energy to $10^{-4}$ relative in every run, checks the fitted rates of linear
  Landau damping and of the two-stream instability against the oracle, and writes the cost per
  step, measured in a fresh process, to `<r>/full_order_cost.toml`. The test suite runs its smoke
  mode, and its full mode in the `slow` group.
