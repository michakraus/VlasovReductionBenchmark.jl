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
- The velocity-parity split (`parity_split`, `parity_join`): the fluctuation $f - f_0$ around the
  spatial average $f_0(v)$ (`background`) as the canonical vector $(q, p)$, the odd and the even
  part in $v$ on the nodes with $v > 0$.
- The linear baselines: POD-Galerkin in grid coordinates (`pod_reduction`, ReducedBasisMethods'
  `PODSVD`) and the cotangent-lift PSD-Galerkin in parity coordinates (`psd_reduction`,
  `CotangentLiftSVD` on the stacked parity snapshots), each reducing VlasovMethods' Arakawa
  vector field around $f_0$ and integrated with `ImplicitMidpoint` (`run_reduced`). A run whose
  Newton solve does not converge, or whose state is not finite, stops with its status. With a
  full orthogonal basis both reproduce a full-order `ImplicitMidpoint` run to $10^{-10}$; the
  `slow` group checks this on $32 \times 64$.
- The metrics of a reduced run (`trajectory_metrics`, `fitted_rate`): the relative $L^2$ error at
  each snapshot and its maximum, the fitted damping or growth rate, and the drifts of the
  discrete energy, of the $L^1$ and $L^2$ norms and of the entropy, with the count of nodes where
  $f \le 0$. Each error is relative to $f$ and also relative to the fluctuation $f - f_0$: where
  the fluctuation is small against $f$, only the second shows that a reduced model lost it.
- `scripts/linear_baselines.jl --runs-dir <d> --results-dir <r> [--smoke]`: POD and PSD at
  $n \in \{4, 8, 16, 32\}$ (smoke: $\{4, 8\}$) on the three data runs of `scripts/full_order.jl`
  and on the amplitude sweep, one row each in `<r>/linear_baselines.csv`, with every error both
  relative to $f$ and relative to the fluctuation (`max_l2_error_fluct`, `l2_error_fluct`). It
  asserts that the POD projection error of the fluctuation snapshots does not rise with $n$ and is
  at most the PSD error, and it records the
  online cost per reduced step, measured in a fresh process, as a ratio to the full-order cost.
  A Landau fit that finds too few maxima in a reduced run leaves the rate empty, with the reason
  in the row's `rate_status`. The test suite runs its smoke mode in the `slow` group.
- ReducedBasisMethods, ReducedComplexityModeling (for `ParameterSpace`), GeometricEquations (for
  the vector field of VlasovMethods' integrator) and SimpleSolvers (for the convergence of a
  step) are direct dependencies.
- A DLRA solver (`run_dlra`, `bug_step`): the factored state $f_0 + X S V^{\mathsf T}$ with bases
  orthonormal in the weighted inner products, stepped by the BUG step with augmented bases and a
  truncation back to rank $r$ (the rank-adaptive BUG of Ceruti, Kusch and Lubich at a fixed rank),
  with an RK4 step per substep, the spectral derivative in $x$, centred differences periodic in $v$,
  and the Poisson solve of the full-order runs, at the time step 0.025. A full-rank solver of the
  same discretisation (`run_full_rank`) checks it: at full rank in $x$ the two agree to $10^{-10}$
  over 20 steps, and the full-rank solver and the DLRA run of rank 16 meet the 2 % rate check of
  linear Landau damping on $64 \times 128$. The fixed-rank BUG of Ceruti and Lubich, without the
  augmentation, fails both checks.
- `scripts/dlra.jl --runs-dir <d> --results-dir <r> [--smoke]`: the DLRA solver at
  $r \in \{4, 8, 16, 32\}$ (smoke: $\{4, 8\}$) on the three data runs and the two test amplitudes
  of the sweep, after the checks of the solver, one row each in `<r>/dlra.csv` with the metrics of
  the linear baselines, the drift of the mass, the state size $r(N_x + N_v) + r^2$, the error of
  the full-rank solver against the full-order run, and the online cost per step, measured in a
  fresh process. A run that blows up is a row with its status. The test suite runs the solver's
  checks in the `core` group, and the smoke and the full mode of the script in the `slow` group.
- Random is a direct dependency, for the seeded completion of the initial bases.
